import React, { useEffect, useState } from 'react'
import { useApp } from '../../store/AppContext'
import { getClusterPoints } from '../../api/client'
import type { ClusterPoint } from '../../types'
import Modal from './Modal'

interface Props { clusterId: string; onClose: () => void }

export default function ExpandClusterModal({ clusterId, onClose }: Props) {
  const { state } = useApp()
  const cluster = state.session?.clusters.find(c => c.id === clusterId)
  const [points, setPoints] = useState<ClusterPoint[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [showAll, setShowAll] = useState(false)

  useEffect(() => {
    setLoading(true)
    getClusterPoints(clusterId)
      .then(r => setPoints(r.points))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [clusterId])

  const filtered = points.filter(p => {
    const text = String((p.data as Record<string,unknown>)?.text ?? p.id).toLowerCase()
    return !search || text.includes(search.toLowerCase())
  })
  const shown = showAll ? filtered : filtered.slice(0, 10)

  function highlight(text: string) {
    if (!search) return text
    const idx = text.toLowerCase().indexOf(search.toLowerCase())
    if (idx < 0) return text
    return (
      <>
        {text.slice(0, idx)}
        <mark className="bg-yellow-100 text-ink rounded-sm">{text.slice(idx, idx + search.length)}</mark>
        {text.slice(idx + search.length)}
      </>
    )
  }

  return (
    <Modal title={cluster ? `${cluster.name}  ·  ${cluster.size} pts` : 'Cluster points'} onClose={onClose} width="max-w-2xl">
      <div className="p-5 flex flex-col gap-3">
        <input
          value={search} onChange={e => { setSearch(e.target.value); setShowAll(false) }}
          placeholder="Search points…"
          className="w-full px-3 py-2 rounded-sm border border-border text-[13px] outline-none focus:border-borders transition-colors"
          style={{ background: 'var(--color-surface)' }} />

        {loading ? (
          <div className="py-8 text-center text-faint text-sm">Loading…</div>
        ) : (
          <>
            <div className="flex flex-col gap-1">
              {shown.map(p => {
                const text = String((p.data as Record<string,unknown>)?.text ?? p.id)
                return (
                  <div key={p.id} className="px-3 py-2.5 rounded-sm border border-border text-[13px] text-ink leading-relaxed hover:bg-surface2 transition-colors"
                    style={{ background: 'var(--color-surface)' }}>
                    {highlight(text.slice(0, 240))}
                    {text.length > 240 && <span className="text-faint">…</span>}
                  </div>
                )
              })}
            </div>
            {!showAll && filtered.length > 10 && (
              <button onClick={() => setShowAll(true)}
                className="text-[12px] font-mono font-bold tracking-wider uppercase text-faint hover:text-muted transition-colors text-center py-1">
                Show all {filtered.length} items
              </button>
            )}
            {filtered.length === 0 && (
              <div className="py-4 text-center text-faint text-sm">No points match "{search}"</div>
            )}
          </>
        )}
      </div>
    </Modal>
  )
}
