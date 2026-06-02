import React, { useEffect, useState } from 'react'
import { useApp, makeSession } from '../store/AppContext'
import { getSessions, getActiveClusters, getTurns, deleteSession } from '../api/client'
import type { Session } from '../types'
import NewSessionModal from './modals/NewSessionModal'
import DatasetsModal from './modals/DatasetsModal'

const STATUS_LABEL: Record<string, string> = { active: 'ACTIVE', converged: 'CONVERGED', closed: 'CLOSED' }
const STATUS_CLS: Record<string, string> = {
  active:    'bg-green-50 text-green-700 border border-green-200',
  converged: 'bg-blue-50 text-blue-700 border border-blue-200',
  closed:    'bg-surface2 text-muted border border-border',
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`font-mono text-[12px] font-bold tracking-widest uppercase px-2 py-0.5 rounded-sm ${STATUS_CLS[status] ?? ''}`}>
      {STATUS_LABEL[status] ?? status}
    </span>
  )
}

export default function WelcomePage() {
  const { goWorkspace, dispatch } = useApp()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [resumingId, setResumingId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [showDatasets, setShowDatasets] = useState(false)

  async function loadSessions() {
    try {
      setSessions(await getSessions())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadSessions() }, [])

  async function handleResume(s: Session) {
    setResumingId(s.id)
    try {
      const [clusters, turns] = await Promise.all([getActiveClusters(s.id), getTurns(s.id)])
      goWorkspace(makeSession(s, clusters, turns))
    } finally {
      setResumingId(null)
    }
  }

  async function handleDelete(id: string) {
    setDeletingId(id)
    try {
      await deleteSession(id)
      setSessions(prev => prev.filter(s => s.id !== id))
    } finally {
      setDeletingId(null)
      setConfirmDeleteId(null)
    }
  }

  return (
    <>
      <div className="flex-1 flex flex-col items-center justify-start pt-[14dvh] relative overflow-y-auto scrollbar-thin">
        {/* radial glow */}
        <div className="absolute inset-0 pointer-events-none"
          style={{ background: 'radial-gradient(ellipse 65% 55% at 50% 42%, rgba(74,124,89,.12) 0%, transparent 68%)' }} />

        <div className="relative z-10 w-full max-w-xl px-5 text-center">
          {/* Animated logo */}
          <div className="flex gap-2 items-end justify-center mb-8 h-[52px]">
            {[52, 33, 18].map((h, i) => (
              <div key={i} className="w-3.5 rounded-sm animate-bar-breathe"
                style={{ height: h, background: 'var(--color-accent)', opacity: i === 0 ? 1 : i === 1 ? 0.52 : 0.26, animationDelay: `${i * 0.26}s` }} />
            ))}
          </div>

          <h1 className="font-serif text-[56px] font-normal tracking-tight mb-4 leading-tight">BlaBlaClust</h1>
          <p className="text-[19px] text-muted leading-relaxed mb-10 max-w-md mx-auto">
            Conversational clustering — refine how your data is grouped through natural language.
          </p>

          <div className="flex items-center gap-3 justify-center">
            <button
              onClick={() => dispatch({ type: 'OPEN_MODAL', modal: 'new-session' })}
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-sm font-medium text-[17px] text-white transition-all duration-150 hover:opacity-90 active:scale-95"
              style={{ background: 'var(--color-accent)' }}>
              + New session
            </button>
            <button
              onClick={() => setShowDatasets(true)}
              className="inline-flex items-center gap-2 px-5 py-3.5 rounded-sm font-medium text-[17px] border border-border text-muted hover:bg-surface2 hover:border-borders transition-all duration-150 active:scale-95"
              style={{ background: 'var(--color-surface)' }}>
              Datasets
            </button>
          </div>
        </div>

        {/* Session list */}
        {!loading && sessions.length > 0 && (
          <div className="relative z-10 w-full max-w-xl px-5 mt-10">
            <p className="font-mono text-[13px] font-bold tracking-[0.14em] uppercase text-faint mb-4">Recent sessions</p>
            <div className="flex flex-col gap-2 max-h-[40dvh] overflow-y-auto scrollbar-thin pr-1">
              {sessions.map(s => (
                <div key={s.id}
                  className="flex items-center gap-4 px-5 py-4 rounded-sm border border-border bg-surface/70 hover:bg-surface2 hover:border-borders transition-all duration-100">
                  <div className="flex-1 min-w-0 text-left">
                    <div className="font-mono text-[13px] text-faint truncate uppercase tracking-wider">{s.dataset_name}</div>
                    <div className="text-[17px] text-muted mt-1 truncate">{s.name || s.id.slice(0, 8)}</div>
                  </div>
                  <StatusBadge status={s.status} />
                  <div className="flex items-center gap-2 shrink-0">
                    <button onClick={() => handleResume(s)} disabled={resumingId === s.id}
                      className="font-mono text-[13px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border transition-colors disabled:opacity-50"
                      style={{ borderColor: 'var(--color-accent)', color: 'var(--color-accent)' }}>
                      {resumingId === s.id ? '…' : 'resume'}
                    </button>
                    {confirmDeleteId === s.id ? (
                      <>
                        <button onClick={() => handleDelete(s.id)} disabled={deletingId === s.id}
                          className="font-mono text-[13px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 transition-colors">
                          {deletingId === s.id ? '…' : 'confirm'}
                        </button>
                        <button onClick={() => setConfirmDeleteId(null)}
                          className="font-mono text-[13px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted transition-colors">
                          cancel
                        </button>
                      </>
                    ) : (
                      <button onClick={() => setConfirmDeleteId(s.id)}
                        className="font-mono text-[13px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-red-700 hover:border-red-200 transition-colors">
                        del
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="relative z-10 mt-8 text-faint text-sm">loading sessions…</div>
        )}
      </div>

      <NewSessionModal onCreated={s => { loadSessions(); handleResume(s) }} />
      {showDatasets && <DatasetsModal onClose={() => setShowDatasets(false)} />}
    </>
  )
}
