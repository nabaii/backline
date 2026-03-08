import React, { useEffect, useRef, useState, useCallback } from 'react'
import { chatStream, analyzeBetSlip } from '../api/backendApi'
import ChatMiniChart from './ChatMiniChart'
import BetSlipThread from './BetSlipThread'

const CHART_DELIMITER = '\n---CHART_DATA---\n'

const GREETING_PHRASES = [
  'What are we cooking up today?',
  'What do you want to bet?',
  'Upload and analyse your bet slip',
]
const TYPE_SPEED = 60
const ERASE_SPEED = 35
const PAUSE_AFTER_TYPE = 2000
const PAUSE_AFTER_ERASE = 400

function useTypewriter(phrases) {
  const [display, setDisplay] = useState('')
  const [showCursor, setShowCursor] = useState(true)
  const phraseIdx = useRef(0)
  const charIdx = useRef(0)
  const isErasing = useRef(false)
  const timerRef = useRef(null)

  useEffect(() => {
    const tick = () => {
      const current = phrases[phraseIdx.current]

      if (!isErasing.current) {
        // Typing
        charIdx.current++
        setDisplay(current.slice(0, charIdx.current))

        if (charIdx.current >= current.length) {
          // Done typing — pause then erase
          timerRef.current = setTimeout(() => {
            isErasing.current = true
            tick()
          }, PAUSE_AFTER_TYPE)
          return
        }
        timerRef.current = setTimeout(tick, TYPE_SPEED)
      } else {
        // Erasing
        charIdx.current--
        setDisplay(current.slice(0, charIdx.current))

        if (charIdx.current <= 0) {
          // Done erasing — move to next phrase
          isErasing.current = false
          phraseIdx.current = (phraseIdx.current + 1) % phrases.length
          timerRef.current = setTimeout(tick, PAUSE_AFTER_ERASE)
          return
        }
        timerRef.current = setTimeout(tick, ERASE_SPEED)
      }
    }

    timerRef.current = setTimeout(tick, PAUSE_AFTER_ERASE)
    return () => clearTimeout(timerRef.current)
  }, [phrases])

  // Blink cursor
  useEffect(() => {
    const id = setInterval(() => setShowCursor(v => !v), 530)
    return () => clearInterval(id)
  }, [])

  return { display, showCursor }
}

export default function ChatWindow({ selectedFixture, onNavigateToKitchen }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const textareaRef = useRef(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const abortRef = useRef(false)
  // Buffer to accumulate chunks until we find the delimiter
  const bufferRef = useRef('')
  const chartParsedRef = useRef(false)
  const { display: greetingText, showCursor: greetingCursor } = useTypewriter(GREETING_PHRASES)

  const hasMessages = messages.length > 0
  const fixtureLabel = selectedFixture
    ? `${selectedFixture.home_team_name} vs ${selectedFixture.away_team_name}`
    : null

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    const userMsg = { role: 'user', text }
    const assistantMsg = { role: 'assistant', text: '', chartData: null }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
    setIsStreaming(true)
    abortRef.current = false
    bufferRef.current = ''
    chartParsedRef.current = false

    try {
      await chatStream(
        {
          query: text,
          home_team_id: selectedFixture?.home_team_id || undefined,
          away_team_id: selectedFixture?.away_team_id || undefined,
          home_team_name: selectedFixture?.home_team_name || '',
          away_team_name: selectedFixture?.away_team_name || '',
          league_id: selectedFixture?.league_id || '',
        },
        (chunk) => {
          if (abortRef.current) return

          // If we already parsed chart data, append directly as text
          if (chartParsedRef.current) {
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.role === 'assistant') {
                next[next.length - 1] = { ...last, text: last.text + chunk }
              }
              return next
            })
            return
          }

          // Buffer until we find the delimiter
          bufferRef.current += chunk
          const delimIdx = bufferRef.current.indexOf(CHART_DELIMITER)

          if (delimIdx === -1) return // keep buffering

          // Found delimiter — split into chart JSON and remaining text
          const jsonPart = bufferRef.current.slice(0, delimIdx)
          const textPart = bufferRef.current.slice(delimIdx + CHART_DELIMITER.length)
          chartParsedRef.current = true
          bufferRef.current = ''

          let chartData = null
          try {
            chartData = JSON.parse(jsonPart)
          } catch {
            // Not valid JSON — treat everything as text
          }

          setMessages(prev => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last?.role === 'assistant') {
              next[next.length - 1] = {
                ...last,
                text: textPart,
                chartData,
              }
            }
            return next
          })
        }
      )

      // If stream ended without a delimiter (no chart data), flush buffer as text
      if (!chartParsedRef.current && bufferRef.current) {
        const remaining = bufferRef.current
        bufferRef.current = ''
        setMessages(prev => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === 'assistant') {
            next[next.length - 1] = { ...last, text: last.text + remaining }
          }
          return next
        })
      }
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

  // ── Bet slip upload handler ──
  const handleBetSlipUpload = async (file) => {
    if (!file || isStreaming) return

    // Create a preview URL for the image
    const imageUrl = URL.createObjectURL(file)

    const userMsg = { role: 'user', text: '', imageUrl, imageName: file.name }
    const assistantMsg = {
      role: 'assistant',
      text: '',
      betSlip: { analyses: [], extractedCount: 0, isProcessing: true },
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsStreaming(true)
    abortRef.current = false

    try {
      await analyzeBetSlip(
        file,
        selectedFixture?.league_id || '',
        (event) => {
          if (abortRef.current) return

          if (event.type === 'bets_extracted') {
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.betSlip) {
                next[next.length - 1] = {
                  ...last,
                  betSlip: {
                    ...last.betSlip,
                    extractedCount: event.count,
                  },
                }
              }
              return next
            })
          } else if (event.type === 'analysis') {
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.betSlip) {
                next[next.length - 1] = {
                  ...last,
                  betSlip: {
                    ...last.betSlip,
                    analyses: [...last.betSlip.analyses, event.data],
                  },
                }
              }
              return next
            })
          } else if (event.type === 'done') {
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.betSlip) {
                next[next.length - 1] = {
                  ...last,
                  betSlip: { ...last.betSlip, isProcessing: false },
                }
              }
              return next
            })
          } else if (event.type === 'error') {
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.betSlip) {
                next[next.length - 1] = {
                  ...last,
                  text: `Error: ${event.message}`,
                  error: true,
                  betSlip: { ...last.betSlip, isProcessing: false },
                }
              }
              return next
            })
          }
        }
      )
    } catch (err) {
      if (!abortRef.current) {
        setMessages(prev => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === 'assistant') {
            next[next.length - 1] = {
              ...last,
              text: `Error: ${err.message}`,
              error: true,
              betSlip: last.betSlip
                ? { ...last.betSlip, isProcessing: false }
                : undefined,
            }
          }
          return next
        })
      }
    } finally {
      setIsStreaming(false)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      handleBetSlipUpload(file)
      e.target.value = '' // reset so same file can be re-uploaded
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
            <h2 className="chat-greeting-text">
              {greetingText}
              <span className={`chat-greeting-cursor${greetingCursor ? '' : ' chat-greeting-cursor--hidden'}`}>|</span>
            </h2>
          </div>
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <React.Fragment key={i}>
              {/* User image message */}
              {msg.imageUrl && (
                <div className="chat-message chat-message--user chat-message--image">
                  <img
                    src={msg.imageUrl}
                    alt={msg.imageName || 'Bet slip'}
                    className="chat-betslip-image"
                  />
                </div>
              )}
              {/* Text message */}
              {(msg.text || (!msg.imageUrl && !msg.betSlip)) && (
                <div
                  className={`chat-message chat-message--${msg.role}${msg.error ? ' chat-message--error' : ''}`}
                >
                  {msg.text
                    ? msg.text
                    : msg.role === 'assistant'
                      ? <span className="chat-typing-cursor" />
                      : null}
                </div>
              )}
              {/* Chart as a separate "message" below the text */}
              {msg.chartData?.recent_matches?.length > 0 && (
                <div className="chat-message chat-message--chart">
                  <ChatMiniChart
                    chartData={msg.chartData.recent_matches}
                    betType={msg.chartData.bet_type}
                    line={msg.chartData.line}
                    teamName={msg.chartData.home_team}
                    onNavigateToKitchen={onNavigateToKitchen}
                  />
                </div>
              )}
              {/* Bet slip thread */}
              {msg.betSlip && (
                <div className="chat-message chat-message--chart">
                  <BetSlipThread
                    analyses={msg.betSlip.analyses}
                    extractedCount={msg.betSlip.extractedCount}
                    isProcessing={msg.betSlip.isProcessing}
                    onNavigateToKitchen={onNavigateToKitchen}
                  />
                </div>
              )}
            </React.Fragment>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

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
            <div className="chat-input-actions">
              <button
                className="chat-attach-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={isStreaming}
                aria-label="Upload bet slip"
                title="Upload bet slip"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              </button>
              <button
                className="chat-send-btn"
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                aria-label="Send message"
              >
                {isStreaming ? <span className="chat-send-spinner" /> : '\u2191'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
