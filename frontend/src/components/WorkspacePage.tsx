import React, { useEffect, useState } from 'react'
import { useApp } from '../store/AppContext'
import { patchSessionState, getEvalCached } from '../api/client'
import ClusterCard from './ClusterCard'
import ChatPanel from './ChatPanel'
import AnalyticsPanel from './AnalyticsPanel'
import ExpandClusterModal from './modals/ExpandClusterModal'
import StopSessionModal from './modals/StopSessionModal'
import EvalModal from './modals/EvalModal'
import UmapModal from './modals/UmapModal'
import type { EvalResult } from '../types'

const LOAD_COLORS = ['#4a7c59','#7a8a3a','#9a7a30','#8a5030','#8a3030']
const STATUS_CLS: Record<string, string> = {
  active:    'bg-green-50 text-green-700 border border-green-200',
  converged: 'bg-blue-50 text-blue-700 border border-blue-200',
  closed:    'text-muted border border-border',
}

export default function WorkspacePage() {
  const { state, dispatch, goWelcome } = useApp()
  const sess = state.session!
  const [expandId, setExpandId] = useState<string | null>(null)
  const [showStop, setShowStop] = useState(false)
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [showUmap, setShowUmap] = useState(false)
  const [marking, setMarking] = useState(false)

  const totalSize = sess.clusters.reduce((s, c) => s + c.size, 0)

  useEffect(() => {
    getEvalCached(sess.sessionId).then(() => setShowEval(true)).catch(() => {})
  }, [sess.sessionId])

  async function markConverged() {
    setMarking(true)
    try {
      await patchSessionState(sess.sessionId, 'converged')
      dispatch({ type: 'UPDATE_STATUS', status: 'converged' })
    } finally {
      setMarking(false)
    }
  }

  const status = sess.session.status

  return (
    <div className="flex flex-col h-full">
      {/* ── Header ── */}
      <header className="h-16 shrink-0 flex items-center justify-between px-6 border-b border-border z-50"
        style={{ background: 'var(--color-surface)' }}>
        <div className="flex items-center">
          <button onClick={goWelcome} disabled={sess.isBusy}
            className="font-mono text-[14px] font-bold tracking-wider uppercase text-faint px-3 py-2 rounded-sm hover:text-muted hover:bg-surface2 transition-colors disabled:opacity-40">
            ← back
          </button>
          <div className="w-px h-6 mx-5" style={{ background: 'var(--color-borders)' }} />
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[13px] font-bold tracking-widest uppercase text-faint">
              {sess.session.dataset_name}
            </span>
            <span className="text-borders">/</span>
            <span className="font-serif text-[19px] text-ink">{sess.session.name || sess.sessionId.slice(0, 8)}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="font-mono text-[14px] text-faint">T{sess.turnNumber}</span>
          <div className="w-px h-4 bg-border" />
          <span className="font-mono text-[14px] text-faint">${sess.costUsd.toFixed(4)}</span>
          <div className="w-px h-4 bg-border" />
          <div className="flex items-center gap-1">
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i} className="w-2.5 h-2.5 rounded-full transition-colors"
                style={{ background: i < sess.cognitiveLoad ? LOAD_COLORS[i] : 'var(--color-surface3)' }} />
            ))}
          </div>
          <div className="w-px h-4 bg-border" />
          <span className={`font-mono text-[13px] font-bold tracking-widest uppercase px-3 py-1 rounded-sm ${STATUS_CLS[status] ?? ''}`}>
            {status}
          </span>
          <div className="w-px h-4 bg-border" />
          {status === 'active' && (
            <button onClick={markConverged} disabled={marking}
              className="font-mono text-[13px] font-bold tracking-wider uppercase px-3.5 py-2 rounded-sm border border-blue-200 text-blue-700 bg-blue-50 hover:bg-blue-100 transition-colors disabled:opacity-50">
              {marking ? '…' : '✓ converged'}
            </button>
          )}
          {status === 'active' && (
            <button onClick={() => setShowStop(true)}
              className="font-mono text-[13px] font-bold tracking-wider uppercase px-3.5 py-2 rounded-sm border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 transition-colors">
              stop
            </button>
          )}
        </div>
      </header>

      {/* ── Main layout ── */}
      <div className="flex flex-1 min-h-0">
        {/* Left column: cluster grid + analytics strip */}
        <div className="flex flex-col flex-1 min-h-0 min-w-0">
          {/* Cluster grid */}
          <div className="flex-1 overflow-y-auto scrollbar-thin p-7">
            {sess.clusters.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <p className="text-faint text-lg">No clusters yet. Initialising…</p>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="font-mono text-[14px] font-bold tracking-widest uppercase text-faint">
                    {sess.clusters.length} clusters · {totalSize} points
                  </h2>
                  {sess.selectedClusterIds.size > 0 && (
                    <button onClick={() => dispatch({ type: 'CLEAR_CLUSTER_SELECT' })}
                      className="font-mono text-[14px] text-faint hover:text-muted transition-colors">
                      clear selection
                    </button>
                  )}
                </div>
                <div className="grid gap-5"
                  style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
                  {sess.clusters.map((c, i) => (
                    <ClusterCard
                      key={c.id}
                      cluster={c}
                      index={i}
                      selected={sess.selectedClusterIds.has(c.id)}
                      totalSize={totalSize}
                      onSelect={() => dispatch({ type: 'TOGGLE_CLUSTER_SELECT', clusterId: c.id })}
                      onExpand={() => setExpandId(c.id)}
                    />
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Analytics strip */}
          <AnalyticsPanel
            onOpenUmap={() => setShowUmap(true)}
            onOpenEval={(result) => setEvalResult(result)}
          />
        </div>

        {/* Chat panel */}
        <div className="w-[500px] shrink-0 flex flex-col min-h-0">
          <ChatPanel />
        </div>
      </div>

      {/* Modals */}
      {expandId && <ExpandClusterModal clusterId={expandId} onClose={() => setExpandId(null)} />}
      {showStop && <StopSessionModal onClose={() => setShowStop(false)} onClosed={goWelcome} />}
      {evalResult && <EvalModal result={evalResult} onClose={() => setEvalResult(null)} />}
      {showUmap && <UmapModal sessionId={sess.sessionId} onClose={() => setShowUmap(false)} />}
    </div>
  )
}
