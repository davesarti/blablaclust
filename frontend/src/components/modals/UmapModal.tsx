import React, { useEffect, useState, useRef, useCallback } from 'react'
import { getUmap } from '../../api/client'
import type { UmapData } from '../../types'
import Modal from './Modal'

const UMAP_COLORS = [
  '#4a7c59','#c06b3e','#3e6fa3','#a3543e','#6b8e23','#8a5a9e',
  '#2e8b8b','#b07a2e','#9e3e5a','#5a6e8a','#7a9e3e','#3e8a5a',
]

interface Props { sessionId: string; onClose: () => void }

export default function UmapModal({ sessionId, onClose }: Props) {
  const [data, setData] = useState<UmapData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [turnIdx, setTurnIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [geomAware, setGeomAware] = useState(false)
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const plotRef = useRef<HTMLDivElement>(null)

  async function fetchData(geom: boolean) {
    setLoading(true); setError('')
    try {
      const d = await getUmap(sessionId, geom)
      setData(d); setTurnIdx(0)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData(false) }, [])

  const renderTurn = useCallback(async (ti: number) => {
    if (!data || !plotRef.current) return
    const Plotly = (await import('plotly.js-dist-min')).default as typeof import('plotly.js')
    const turn = data.turns[ti]
    const turnStr = String(turn)

    // Build color map
    const clusterIds = Object.keys(data.clusters)
    const colorMap: Record<string, string> = {}
    clusterIds.forEach((id, i) => { colorMap[id] = UMAP_COLORS[i % UMAP_COLORS.length] })

    const assigns = (data.assignments[turnStr] ?? []) as (string | null)[]
    const geomTurn = data.geometry_aware?.[turnStr]
    const pts = geomTurn ? geomTurn.points : data.points

    // Group points by cluster
    const byCluster: Record<string, { x: number[]; y: number[]; text: string[] }> = {}
    pts.forEach((p, i) => {
      const cid = assigns[i] ?? '__none__'
      if (!byCluster[cid]) byCluster[cid] = { x: [], y: [], text: [] }
      byCluster[cid].x.push(p.x)
      byCluster[cid].y.push(p.y)
      byCluster[cid].text.push(data.points[i]?.text ?? '')
    })

    const traces: Plotly.Data[] = Object.entries(byCluster).map(([cid, d]) => ({
      type: 'scattergl' as const,
      x: d.x, y: d.y,
      text: d.text,
      hovertemplate: '%{text}<extra></extra>',
      mode: 'markers',
      name: cid === '__none__' ? 'unassigned' : (data.clusters[cid]?.name ?? cid),
      marker: { color: cid === '__none__' ? '#ccc' : colorMap[cid], size: 5, opacity: 0.75 },
    }))

    // Centroids
    const cents = data.centroids_by_turn[turnStr]
    if (cents && !geomTurn) {
      Object.entries(cents).forEach(([cid, [cx, cy]]) => {
        traces.push({
          type: 'scatter', x: [cx], y: [cy], mode: 'markers',
          showlegend: false, hoverinfo: 'skip',
          marker: { color: colorMap[cid] ?? '#888', size: 12, symbol: 'circle', line: { color: 'white', width: 2 } },
        } as Plotly.Data)
      })
    }

    // Axis arrow
    const arrow = !geomTurn && data.axis_arrows[turnStr]
    if (arrow) {
      traces.push({
        type: 'scatter',
        x: [arrow.from[0], arrow.to[0]], y: [arrow.from[1], arrow.to[1]],
        mode: 'lines+markers',
        name: arrow.label || 'axis',
        line: { color: '#000', width: 2 },
        marker: { symbol: ['circle', 'arrow-up'], size: [6, 14], angleref: 'previous', color: '#000' },
        hoverinfo: 'name',
      } as Plotly.Data)
    }

    const silh = data.silhouette_by_turn[turnStr]
    const title = [silh != null ? `silhouette ${silh.toFixed(3)}` : '', geomTurn ? geomTurn.axis_label : ''].filter(Boolean).join('  ·  ')

    await Plotly.react(plotRef.current, traces, {
      title: { text: title, font: { size: 11, color: '#787868', family: 'ui-monospace' } },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { t: 32, r: 16, b: 24, l: 0 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false, domain: [0, 0.60] },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
      legend: { x: 0.98, xanchor: 'right', y: 0.98, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 15, family: 'ui-monospace' } },
      hovermode: 'closest',
    }, { responsive: true, displayModeBar: false })
  }, [data])

  useEffect(() => { renderTurn(turnIdx) }, [turnIdx, data, renderTurn])

  useEffect(() => {
    if (playing && data) {
      playRef.current = setInterval(() => {
        setTurnIdx(i => { const n = (i + 1) % data.turns.length; if (n === 0) setPlaying(false); return n })
      }, 1300)
    } else if (playRef.current) {
      clearInterval(playRef.current); playRef.current = null
    }
    return () => { if (playRef.current) clearInterval(playRef.current) }
  }, [playing, data])

  async function toggleGeom(v: boolean) {
    setGeomAware(v)
    await fetchData(v)
  }

  return (
    <Modal title="UMAP clustering evolution" onClose={onClose} width="max-w-6xl">
      <div className="px-6 py-4 flex flex-col gap-3">
        {loading && <div className="py-12 text-center text-faint text-sm">Computing projection…</div>}
        {error && <div className="px-3 py-2.5 rounded-sm border border-red-200 bg-red-50 text-red-700 text-[15px]">{error}</div>}

        {data && !loading && (
          <>
            {/* History bar */}
            {(() => {
              const nTurns = data.turns.length
              const silhEntries = data.turns.map(t => data.silhouette_by_turn[String(t)] ?? null)
              const hasSilh = silhEntries.some(v => v != null)
              const maxSilh = Math.max(...silhEntries.map(v => v ?? 0), 0.001)
              const progressPct = nTurns > 1 ? (turnIdx / (nTurns - 1)) * 100 : 0
              const dotBase = nTurns > 40 ? 4 : nTurns > 20 ? 6 : 8
              return (
                <div className="flex flex-col gap-2">
                  <div className="flex flex-col gap-0.5">
                    {/* Silhouette sparkline */}
                    {hasSilh && (
                      <div className="grid items-end" style={{ gridTemplateColumns: `repeat(${nTurns}, 1fr)`, height: 28 }}>
                        {data.turns.map((t, i) => {
                          const s = silhEntries[i] ?? 0
                          const h = Math.max(2, Math.round((Math.max(0, s) / maxSilh) * 24))
                          return (
                            <div key={t} className="flex justify-center items-end" style={{ height: '100%' }}>
                              <div style={{
                                width: nTurns > 30 ? 2 : 3,
                                height: h,
                                background: i === turnIdx ? 'var(--color-accent)' : i < turnIdx ? 'rgba(74,124,89,0.4)' : 'var(--color-border)',
                                borderRadius: 1,
                                transition: 'background 0.15s',
                              }} />
                            </div>
                          )
                        })}
                      </div>
                    )}

                    {/* Track with dot markers */}
                    <div className="relative" style={{ height: dotBase + 8 }}>
                      {/* Background line */}
                      <div className="absolute left-0 right-0 bg-border" style={{ top: Math.floor((dotBase + 8) / 2), height: 1 }} />
                      {/* Progress fill */}
                      <div className="absolute bg-accent transition-all duration-150" style={{ top: Math.floor((dotBase + 8) / 2), left: 0, height: 1, width: `${progressPct}%` }} />
                      {/* Dots */}
                      <div className="grid absolute inset-0" style={{ gridTemplateColumns: `repeat(${nTurns}, 1fr)` }}>
                        {data.turns.map((t, i) => {
                          const s = silhEntries[i]
                          const tip = `Turn ${t}${s != null ? ` · silhouette ${s.toFixed(3)}` : ''}`
                          const isCur = i === turnIdx
                          const isPast = i < turnIdx
                          return (
                            <button
                              key={t}
                              onClick={() => { setPlaying(false); setTurnIdx(i) }}
                              title={tip}
                              className="flex items-center justify-center"
                            >
                              <div style={{
                                width: isCur ? dotBase + 2 : dotBase - 2,
                                height: isCur ? dotBase + 2 : dotBase - 2,
                                borderRadius: '50%',
                                background: isCur ? 'var(--color-accent)' : isPast ? 'rgba(74,124,89,0.5)' : 'var(--color-surface3)',
                                border: isCur ? '2px solid var(--color-accent)' : `1px solid ${isPast ? 'rgba(74,124,89,0.6)' : 'var(--color-borders)'}`,
                                boxShadow: isCur ? '0 0 0 3px rgba(74,124,89,0.18)' : 'none',
                                transition: 'all 0.15s',
                              }} />
                            </button>
                          )
                        })}
                      </div>
                    </div>

                    {/* Turn labels */}
                    <div className="flex justify-between font-mono text-[11px] text-faint mt-0.5">
                      <span>T{data.turns[0]}</span>
                      <span className="text-ink font-bold">Turn {data.turns[turnIdx]}</span>
                      <span>T{data.turns[nTurns - 1]}</span>
                    </div>
                  </div>

                  {/* Action row */}
                  <div className="flex items-center gap-3">
                    <button onClick={() => setPlaying(p => !p)}
                      className="font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-muted hover:bg-surface2 transition-colors">
                      {playing ? '⏸ pause' : '▶ play'}
                    </button>
                    {data.geometry_aware && (
                      <label className="flex items-center gap-1.5 font-mono text-[12px] font-bold tracking-wider uppercase text-faint cursor-pointer">
                        <input type="checkbox" checked={geomAware} onChange={e => toggleGeom(e.target.checked)} className="accent-accent" />
                        geom-aware
                      </label>
                    )}
                  </div>
                </div>
              )
            })()}

            {/* Plot */}
            <div ref={plotRef} className="w-full" style={{ height: 520 }} />

            {/* Reducer info */}
            <p className="font-mono text-[12px] text-faint text-right mt-3">
              {data.reducer} · {data.n_points} pts
            </p>
          </>
        )}
      </div>
    </Modal>
  )
}
