import React, { useEffect, useRef, useState } from 'react'
import { chatStream } from '../api/backendApi'
import ChatMiniChart from './ChatMiniChart'

const CHART_DELIMITER = '\n---CHART_DATA---\n'

export default function ChatWindow({ selectedFixture, onNavigateToKitchen }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const textareaRef = useRef(null)
  const messagesEndRef = useRef(null)
  const abortRef = useRef(false)
  // Buffer to accumulate chunks until we find the delimiter
  const bufferRef = useRef('')
  const chartParsedRef = useRef(false)

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
            <h2 className="chat-greeting-text">What do you want to bet?</h2>
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
            <React.Fragment key={i}>
              {/* Text message */}
              <div
                className={`chat-message chat-message--${msg.role}${msg.error ? ' chat-message--error' : ''}`}
              >
                {msg.text
                  ? msg.text
                  : msg.role === 'assistant'
                    ? <span className="chat-typing-cursor" />
                    : null}
              </div>
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
            </React.Fragment>
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
