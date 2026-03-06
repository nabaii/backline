import React, { useEffect, useRef, useState } from 'react'
import { ragStream } from '../api/backendApi'

export default function ChatWindow({ selectedFixture }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const textareaRef = useRef(null)
  const messagesEndRef = useRef(null)
  const abortRef = useRef(false)

  const hasMessages = messages.length > 0
  const homeTeamId = selectedFixture?.home_team_id || null
  const awayTeamId = selectedFixture?.away_team_id || null
  const fixtureLabel = selectedFixture
    ? `${selectedFixture.home_team_name} vs ${selectedFixture.away_team_name}`
    : null

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    const userMsg = { role: 'user', text }
    const assistantMsg = { role: 'assistant', text: '' }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
    setIsStreaming(true)
    abortRef.current = false

    try {
      await ragStream(
        {
          query: text,
          home_team_id: homeTeamId,
          away_team_id: awayTeamId,
          extra_context: fixtureLabel ? `Fixture context: ${fixtureLabel}` : '',
        },
        (chunk) => {
          if (abortRef.current) return
          setMessages(prev => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, text: last.text + chunk }
            }
            return next
          })
        }
      )
    } catch (err) {
      if (!abortRef.current) {
        setMessages(prev => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === 'assistant') {
            next[next.length - 1] = { ...last, text: `Error: ${err.message}`, error: true }
          }
          return next
        })
      }
    } finally {
      setIsStreaming(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [input])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Abort any in-flight stream on unmount
  useEffect(() => () => { abortRef.current = true }, [])

  return (
    <div className="chat-window">
      {!hasMessages ? (
        <div className="chat-empty-state">
          <div className="chat-greeting">
            <span className="chat-greeting-star">✦</span>
            <h2 className="chat-greeting-text">How can I help you today?</h2>
          </div>
          {fixtureLabel && (
            <p className="chat-fixture-hint">
              Selected: <strong>{fixtureLabel}</strong>
            </p>
          )}
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`chat-message chat-message--${msg.role}${msg.error ? ' chat-message--error' : ''}`}
            >
              {msg.text
                ? msg.text
                : msg.role === 'assistant'
                  ? <span className="chat-typing-cursor" />
                  : null}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="chat-input-wrap">
        <div className="chat-input-box">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Message Backline..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isStreaming}
          />
          <div className="chat-input-footer">
            {fixtureLabel && (
              <span className="chat-input-context">{fixtureLabel}</span>
            )}
            <button
              className="chat-send-btn"
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              aria-label="Send message"
            >
              {isStreaming ? <span className="chat-send-spinner" /> : '↑'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
