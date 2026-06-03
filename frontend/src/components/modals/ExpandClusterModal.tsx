import React, { useEffect, useMemo, useState } from 'react'
import { useApp } from '../../store/AppContext'
import { getClusterPoints } from '../../api/client'
import type { ClusterPoint } from '../../types'
import Modal from './Modal'

interface Props { clusterId: string; onClose: () => void }

export default function ExpandClusterModal({ clusterId, onClose }: Props) {
  const { state, dispatch } = useApp()
  const cluster = state.session?.clusters.find(c => c.id === clusterId)
  const selectedPoints = state.session?.selectedPoints ?? new Map()
  const pinnedFromThisCluster = useMemo(
    () => Array.from(selectedPoints.values()).filter(p => p.clusterId === clusterId).length,
    [selectedPoints, clusterId],
  )
  const [points, setPoints] = useState<ClusterPoint[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [showAll, setShowAll] = useState(false)
  const [pinnedOnly, setPinnedOnly] = useState(false)

  useEffect(() => {
    setLoading(true)
    getClusterPoints(clusterId)
      .then(r => setPoints(r.points.slice().sort((a, b) => b.probability - a.probability)))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [clusterId])

  useEffect(() => {
    if (pinnedFromThisCluster === 0 && pinnedOnly) setPinnedOnly(false)
  }, [pinnedFromThisCluster, pinnedOnly])

  const filtered = useMemo(() => {
    let result = points
    if (pinnedOnly) result = result.filter(p => selectedPoints.has(p.id))
    if (search) {
      const q = search.toLowerCase()
      result = result.filter(p => (p.text || p.id).toLowerCase().includes(q))
    }
    return result
  }, [points, search, pinnedOnly, selectedPoints])

  const shown = showAll ? filtered : filtered.slice(0, 12)

  function highlight(text: string) {
    if (!search) return <>{text}</>
    const idx = text.toLowerCase().indexOf(search.toLowerCase())
    if (idx < 0) return <>{text}</>
    return (
      <>
        {text.slice(0, idx)}
        <mark className="bg-amber-100 text-ink">{text.slice(idx, idx + search.length)}</mark>
        {text.slice(idx + search.length)}
      </>
    )
  }

  const maxProb = points.length > 0 ? Math.max(...points.map(p => p.probability)) : 1

  return (
    <Modal title={cluster?.name ?? 'Cluster'} onClose={onClose} width="max-w-3xl">
      <div className="flex flex-col">

        {/* Cluster meta */}
        <div className="px-6 pt-4 pb-5" style={{ borderBottom: '1px solid var(--color-border)' }}>
          {cluster?.description && (
            <p className="text-[15px] text-muted leading-relaxed mb-3">{cluster.description}</p>
          )}
          <div className="flex items-center gap-3">
            <span className="font-mono text-[11px] tracking-widest uppercase text-faint">
              {cluster?.size ?? points.length} pts
            </span>
            {pinnedFromThisCluster > 0 && (
              <>
                <span className="text-faint text-[11px]">·</span>
                <button
                  onClick={() => setPinnedOnly(v => !v)}
                  title={pinnedOnly ? 'Show all points' : 'Show only pinned points'}
                  className="font-mono text-[11px] tracking-widest uppercase px-2 py-0.5 rounded-sm border transition-colors"
                  style={{
                    color: 'var(--color-accent)',
                    borderColor: pinnedOnly ? 'var(--color-accent)' : 'var(--color-border)',
                    background: pinnedOnly ? 'color-mix(in srgb, var(--color-accent) 12%, transparent)' : 'transparent',
                  }}>
                  {pinnedFromThisCluster} pinned {pinnedOnly ? '×' : '↗'}
                </button>
              </>
            )}
            <span className="ml-auto font-mono text-[11px] text-faint italic">
              click a point to pin it for move
            </span>
          </div>
        </div>

        {/* Search */}
        <div className="px-6 py-3 flex items-center gap-3" style={{ borderBottom: '1px solid var(--color-border)' }}>
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setShowAll(false) }}
            placeholder="Search points…"
            className="flex-1 text-[14px] outline-none bg-transparent placeholder:text-faint text-ink"
          />
          {search && (
            <span className="font-mono text-[11px] text-faint shrink-0">
              {filtered.length} / {points.length}
            </span>
          )}
        </div>

        {/* Points list */}
        <div className="px-6 py-2 flex flex-col overflow-y-auto scrollbar-thin" style={{ maxHeight: 480 }}>
          {loading ? (
            <div className="py-12 text-center text-faint text-[14px]">Loading…</div>
          ) : filtered.length === 0 ? (
            <div className="py-10 text-center text-faint text-[14px]">No points match "{search}"</div>
          ) : (
            <>
              {shown.map((p, idx) => {
                const text = p.text || p.id
                const barW = Math.round((p.probability / maxProb) * 100)
                const isPinned = selectedPoints.has(p.id)
                return (
                  <div
                    key={p.id}
                    onClick={() => dispatch({ type: 'TOGGLE_POINT_SELECT', pointId: p.id, text, clusterId })}
                    className="group flex items-start gap-4 py-3 cursor-pointer transition-colors"
                    style={{
                      borderBottom: '1px solid var(--color-border)',
                      background: isPinned ? 'color-mix(in srgb, var(--color-accent) 8%, transparent)' : undefined,
                      borderLeft: isPinned ? '3px solid var(--color-accent)' : '3px solid transparent',
                      paddingLeft: isPinned ? 9 : 12,
                    }}>
                    <span className="font-mono text-[12px] shrink-0 w-6 text-right pt-0.5 select-none"
                      style={{ color: isPinned ? 'var(--color-accent)' : 'var(--color-faint)' }}>
                      {isPinned ? '●' : idx + 1}
                    </span>
                    <span className="flex-1 text-[14px] text-ink leading-relaxed">
                      {highlight(text)}
                    </span>
                    <div className="shrink-0 flex flex-col items-end gap-1 pt-1">
                      <span className="font-mono text-[11px] text-faint tabular-nums">{p.probability.toFixed(2)}</span>
                      <div className="w-14 rounded-full overflow-hidden" style={{ height: 2, background: 'var(--color-border)' }}>
                        <div style={{ width: `${barW}%`, height: '100%', background: 'var(--color-accent)', opacity: 0.7 }} />
                      </div>
                    </div>
                  </div>
                )
              })}

              {!showAll && filtered.length > 12 && (
                <button
                  onClick={() => setShowAll(true)}
                  className="py-4 font-mono text-[11px] tracking-widest uppercase text-faint hover:text-muted transition-colors text-center">
                  Show all {filtered.length} points
                </button>
              )}
            </>
          )}
        </div>

      </div>
    </Modal>
  )
}
