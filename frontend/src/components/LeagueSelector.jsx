import React, { useEffect, useRef, useState } from 'react'
import { getLeagueLogo } from '../utils/premierLeagueLogos'

function LeagueDropdown({ leagues, value, onChange }) {
  const [isOpen, setIsOpen] = useState(false)
  const wrapperRef = useRef(null)
  const selectedLeague = leagues.find(l => l.id === value)

  useEffect(() => {
    function handleOutsideClick(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick)
      document.addEventListener('touchstart', handleOutsideClick)
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick)
      document.removeEventListener('touchstart', handleOutsideClick)
    }
  }, [isOpen])

  function handleSelect(id) {
    onChange(id)
    setIsOpen(false)
  }

  const logoUrl = selectedLeague ? getLeagueLogo(selectedLeague.id) : null

  return (
    <div className="league-dropdown" ref={wrapperRef}>
      <button
        type="button"
        className={`league-dropdown-trigger${isOpen ? ' open' : ''}`}
        onClick={() => setIsOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span className="league-dropdown-icon">
          {logoUrl
            ? <img src={logoUrl} alt={selectedLeague?.name} className="league-dropdown-logo" />
            : <span className="league-dropdown-flag">{selectedLeague?.flag || ''}</span>
          }
        </span>
        <span className="league-dropdown-label">{selectedLeague?.name ?? 'Select league'}</span>
        <span className={`league-dropdown-chevron${isOpen ? ' open' : ''}`} aria-hidden="true">
          <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
            <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </span>
      </button>

      {isOpen && (
        <ul className="league-dropdown-menu" role="listbox" aria-label="Leagues">
          {leagues.map(l => {
            const logo = getLeagueLogo(l.id)
            const isActive = l.id === value
            return (
              <li
                key={l.id}
                role="option"
                aria-selected={isActive}
                className={`league-dropdown-item${isActive ? ' active' : ''}`}
                onClick={() => handleSelect(l.id)}
              >
                <span className="league-dropdown-item-icon">
                  {logo
                    ? <img src={logo} alt={l.name} className="league-dropdown-logo" />
                    : <span className="league-dropdown-flag">{l.flag || ''}</span>
                  }
                </span>
                <span className="league-dropdown-item-name">{l.name}</span>
                {isActive && (
                  <span className="league-dropdown-item-check" aria-hidden="true">
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export default function LeagueSelector({ leagues = [], value, onChange }) {
  const tabsRef = useRef(null)
  const [scrollClass, setScrollClass] = useState('scroll-at-start')

  useEffect(() => {
    const el = tabsRef.current
    if (!el) return

    function update() {
      const atStart = el.scrollLeft <= 4
      const atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 4
      const cls = [atStart && 'scroll-at-start', atEnd && 'scroll-at-end']
        .filter(Boolean)
        .join(' ')
      setScrollClass(cls)
    }

    update()
    el.addEventListener('scroll', update, { passive: true })
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => {
      el.removeEventListener('scroll', update)
      ro.disconnect()
    }
  }, [leagues])

  if (!leagues.length) return null

  return (
    <>
      {/* Mobile: custom dropdown */}
      <div className="league-select-wrapper">
        <LeagueDropdown leagues={leagues} value={value} onChange={onChange} />
      </div>

      {/* Desktop: scrollable tabs */}
      <div className={`league-tabs-wrapper ${scrollClass}`}>
        <div className="league-tabs" ref={tabsRef}>
          {leagues.map(l => {
            const logoUrl = getLeagueLogo(l.id)
            return (
              <button
                key={l.id}
                className={`league-tab${value === l.id ? ' league-tab--active' : ''}`}
                onClick={() => onChange(l.id)}
                title={l.name}
              >
                {logoUrl
                  ? <img src={logoUrl} alt={l.name} className="league-tab-logo" />
                  : <span className="league-tab-flag">{l.flag || ''}</span>
                }
                <span className="league-tab-name">{l.name}</span>
              </button>
            )
          })}
        </div>
      </div>
    </>
  )
}
