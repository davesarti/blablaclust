import React, { useEffect, useRef, useState } from 'react'
import { useApp } from '../store/AppContext'
import { sendTurn, getActiveClusters } from '../api/client'
import type { ChatMessage, Cluster } from '../types'

const TYPING_MSGS = [
  'Thinking…', 'Analysing clusters…', 'Updating partition…',
  'Naming groups…', 'Checking coherence…',
]

function Message({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${isUser ? 'animate-slide-right' : 'animate-slide-left'}`}>
      <div className={`max-w-[88%] px-5 py-3.5 rounded-sm text-[15px] leading-relaxed ${
        isUser
          ? 'text-white'
          : 'border border-border text-ink'
      }`}
        style={isUser ? { background: 'var(--color-accent)' } : { background: 'var(--color-surface)' }}>
        {msg.text}
      </div>
    </div>
  )
}

function TypingIndicator({ visible }: { visible: boolean }) {
  const [msgIdx, setMsgIdx] = useState(0)
  useEffect(() => {
    if (!visible) { setMsgIdx(0); return }
    const id = setInterval(() => setMsgIdx(i => (i + 1) % TYPING_MSGS.length), 900)
    return () => clearInterval(id)
  }, [visible])
  if (!visible) return null
  return (
    <div className="flex justify-start animate-slide-left">
      <div className="flex items-center gap-3 px-5 py-3.5 rounded-sm border border-border text-[15px] text-faint"
        style={{ background: 'var(--color-surface)' }}>
        <span className="w-2.5 h-2.5 rounded-full animate-live-dot" style={{ background: 'var(--color-accent)' }} />
        {TYPING_MSGS[msgIdx]}
      </div>
    </div>
  )
}

export default function ChatPanel() {
  const { state, dispatch } = useApp()
  const sess = state.session!
  const [text, setText] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sess.chat, sess.isBusy])

  async function handleSend() {
    const raw = text.trim()
    if (!raw || sess.isBusy) return
    setText('')

    const pointIds = [...sess.selectedPoints.keys()]
    const clusterIds = [...sess.selectedClusterIds]
    const feedbackType: 'global' | 'cluster' | 'point' =
      pointIds.length > 0 ? 'point' : clusterIds.length > 0 ? 'cluster' : 'global'

    dispatch({ type: 'APPEND_CHAT', message: { role: 'user', text: raw } })
    dispatch({ type: 'SET_BUSY', busy: true })

    try {
      const turn = await sendTurn({
        session_id: sess.sessionId,
        raw_text: raw,
        feedback_type: feedbackType,
        target_cluster_ids: clusterIds,
        target_point_ids: pointIds,
        metadata: {},
      })

      const so = turn.system_output
      dispatch({ type: 'APPEND_CHAT', message: { role: 'system', text: so.display.content, turnNumber: turn.turn_number } })
      dispatch({ type: 'CLEAR_CLUSTER_SELECT' })
      dispatch({ type: 'CLEAR_POINT_SELECT' })
      dispatch({
        type: 'UPDATE_METRICS',
        turnNumber: turn.turn_number,
        tokenInput: so.token_usage?.input_tokens ?? 0,
        tokenOutput: so.token_usage?.output_tokens ?? 0,
        cost: so.cost_usd ?? 0,
        cogLoad: so.cognitive_load_score ?? 1,
      })

      if (so.clusters_updated) {
        const clusters = await getActiveClusters(sess.sessionId)
        dispatch({ type: 'SET_CLUSTERS', clusters })
      }

      if (so.action === 'stop' || sess.session.status !== 'active') {
        dispatch({ type: 'UPDATE_STATUS', status: 'closed' })
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      dispatch({ type: 'APPEND_CHAT', message: { role: 'system', text: `⚠ ${msg}` } })
    } finally {
      dispatch({ type: 'SET_BUSY', busy: false })
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const selectedClusters: Cluster[] = sess.clusters.filter(c => sess.selectedClusterIds.has(c.id))
  const pinnedPoints = [...sess.selectedPoints.entries()]
  const hasPoints = pinnedPoints.length > 0
  const destinationClear = hasPoints && selectedClusters.length === 1
  const destinationAmbiguous = hasPoints && selectedClusters.length > 1
  const destinationMissing = hasPoints && selectedClusters.length === 0
  const destCluster = destinationClear ? selectedClusters[0] : null
  const inputPlaceholder =
    sess.session.status !== 'active'
      ? 'Session closed'
      : destinationClear
      ? `Move ${pinnedPoints.length} pinned point${pinnedPoints.length === 1 ? '' : 's'} to "${destCluster!.name.slice(0, 24)}"… (Enter to send)`
      : destinationMissing
      ? `Name the destination for ${pinnedPoints.length} pinned point${pinnedPoints.length === 1 ? '' : 's'}, or pin a cluster… (Enter to send)`
      : 'Describe a change… (Enter to send)'

  return (
    <div className="flex flex-col h-full border-l border-border" style={{ background: 'var(--color-surface)' }}>
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-border shrink-0">
        <span className="w-2.5 h-2.5 rounded-full animate-live-dot" style={{ background: 'var(--color-accent)' }} />
        <span className="font-mono text-[13px] font-bold tracking-widest uppercase text-faint">Chat</span>
        {sess.chat.length > 0 && (
          <span className="ml-auto font-mono text-[12px] text-faint">{sess.chat.length} msgs</span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 flex flex-col gap-4">
        {sess.chat.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-[15px] text-faint text-center max-w-[260px] leading-relaxed">
              Describe how you'd like to cluster the data, or select clusters first.
            </p>
          </div>
        )}
        {sess.chat.map((m, i) => <Message key={i} msg={m} />)}
        <TypingIndicator visible={sess.isBusy} />
        <div ref={bottomRef} />
      </div>

      {/* Selected cluster chips */}
      {selectedClusters.length > 0 && (
        <div className="px-6 pb-2 flex flex-wrap gap-2 border-t border-border pt-3 items-center">
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase"
            style={{ color: destinationClear ? 'var(--color-accent)' : 'var(--color-faint)' }}>
            {destinationClear ? 'destination →' : `clusters · ${selectedClusters.length}`}
          </span>
          {selectedClusters.map((c, i) => {
            const isDest = destinationClear && c.id === destCluster!.id
            return (
              <span key={c.id}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-sm border text-[13px] font-mono"
                style={{
                  borderColor: isDest ? 'var(--color-accent)' : 'var(--color-border)',
                  background: isDest ? 'color-mix(in srgb, var(--color-accent) 10%, transparent)' : 'var(--color-surface2)',
                  color: isDest ? 'var(--color-accent)' : 'var(--color-muted)',
                }}>
                #{i + 1} {c.name.slice(0, 22)}
                <button onClick={() => dispatch({ type: 'TOGGLE_CLUSTER_SELECT', clusterId: c.id })}
                  className="text-faint hover:text-red-600 font-bold">×</button>
              </span>
            )
          })}
        </div>
      )}

      {/* Pinned points chips */}
      {hasPoints && (
        <div className={`px-6 pb-2 flex flex-wrap gap-2 items-center ${selectedClusters.length === 0 ? 'border-t border-border pt-3' : 'pt-1'}`}>
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase"
            style={{ color: 'var(--color-faint)' }}>
            points · {pinnedPoints.length}
          </span>
          {pinnedPoints.map(([id, p]) => (
            <span key={id}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-sm border text-[13px] font-mono text-muted max-w-[260px]"
              style={{ background: 'var(--color-surface2)', borderColor: 'var(--color-accent)' }}
              title={p.text}>
              <span className="truncate">{p.text.slice(0, 40)}{p.text.length > 40 ? '…' : ''}</span>
              <button onClick={() => dispatch({ type: 'TOGGLE_POINT_SELECT', pointId: id, text: p.text, clusterId: p.clusterId })}
                className="text-faint hover:text-red-600 font-bold shrink-0">×</button>
            </span>
          ))}
        </div>
      )}

      {/* Destination guidance */}
      {(destinationAmbiguous || destinationMissing) && (
        <div className="px-6 pb-2 flex items-center gap-2 text-[12px] font-mono text-muted">
          <span>ℹ</span>
          <span>
            {destinationAmbiguous
              ? 'Multiple clusters and points pinned — clarify intentions in your message'
              : 'No destination cluster pinned — name the destination in your message (e.g. "move to Reviews").'}
          </span>
        </div>
      )}

      {/* Input */}
      <div className="px-6 pb-6 pt-3 shrink-0 border-t border-border">
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={sess.isBusy || sess.session.status !== 'active'}
            placeholder={inputPlaceholder}
            rows={3}
            className="flex-1 resize-none rounded-sm border border-border px-4 py-3 text-[15px] leading-relaxed outline-none text-ink placeholder:text-faint disabled:opacity-50 focus:border-borders transition-colors scrollbar-thin"
            style={{ background: 'var(--color-bg)' }}
          />
          <button
            onClick={handleSend}
            disabled={!text.trim() || sess.isBusy || sess.session.status !== 'active'}
            className="shrink-0 px-5 py-3 rounded-sm text-white text-lg font-medium transition-all disabled:opacity-40 hover:opacity-90 active:scale-95"
            style={{ background: 'var(--color-accent)' }}>
            {sess.isBusy ? (
              <span className="inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin-slow" />
            ) : '↑'}
          </button>
        </div>
      </div>
    </div>
  )
}
