import React, { useEffect, useState, useRef, useCallback } from 'react'
import { getUmap } from '../../api/client'
import type { UmapData } from '../../types'
import Modal from './Modal'

const UMAP_COLORS = [
  '#4a7c59','#c06b3e','#3e6fa3','#a3543e','#6b8e23','#8a5a9e',
  '#2e8b8b','#b07a2e','#9e3e5a','#5a6e8a','#7a9e3e','#3e8a5a',
]

function wrapText(text: string, width = 52): string {
  const words = text.split(' ')
  const lines: string[] = []
  let line = ''
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word
    if (candidate.length > width && line) {
      lines.push(line)
      line = word
    } else {
      line = candidate
    }
  }
  if (line) lines.push(line)
  return lines.join('<br>')
}

interface Props { sessionId: string; onClose: () => void }

interface LegendItem { id: string; name: string; color: string; count: number }

export default function UmapModal({ sessionId, onClose }: Props) {
  const [data, setData] = useState<UmapData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [turnIdx, setTurnIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [legendItems, setLegendItems] = useState<LegendItem[]>([])
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const plotRef = useRef<HTMLDivElement>(null)

  async function fetchData() {
    setLoading(true); setError('')
    try {
      const d = await getUmap(sessionId)
      setData(d); setTurnIdx(0)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

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
      byCluster[cid].text.push(wrapText(data.points[i]?.text ?? ''))
    })

    // Build legend items sorted by count descending, unassigned last
    const items: LegendItem[] = Object.entries(byCluster)
      .filter(([cid]) => cid !== '__none__')
      .map(([cid, d]) => ({ id: cid, name: data.clusters[cid]?.name ?? cid, color: colorMap[cid], count: d.x.length }))
      .sort((a, b) => b.count - a.count)
    const noneCount = byCluster['__none__']?.x.length ?? 0
    if (noneCount > 0) items.push({ id: '__none__', name: 'unassigned', color: '#ccc', count: noneCount })
    setLegendItems(items)

    const traces: Plotly.Data[] = Object.entries(byCluster).map(([cid, d]) => ({
      type: 'scattergl' as const,
      x: d.x, y: d.y,
      text: d.text,
      hovertemplate: '<span style="font-weight:600;font-size:11px;letter-spacing:0.04em;text-transform:uppercase">%{fullData.name}</span><br><br>%{text}<extra></extra>',
      mode: 'markers',
      name: cid === '__none__' ? 'unassigned' : (data.clusters[cid]?.name ?? cid),
      marker: {
        color: cid === '__none__' ? '#ccc' : colorMap[cid],
        size: 6,
        opacity: 0.82,
        line: { color: 'rgba(255,255,255,0.55)', width: 0.8 },
      },
    }))

    // Centroids
    const cents = data.centroids_by_turn[turnStr]
    if (cents && !geomTurn) {
      Object.entries(cents).forEach(([cid, [cx, cy]]) => {
        traces.push({
          type: 'scatter', x: [cx], y: [cy], mode: 'markers',
          showlegend: false, hoverinfo: 'skip',
          marker: { color: colorMap[cid] ?? '#888', size: 12, symbol: 'circle', line: { color: '#111', width: 1.5 } },
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
      showlegend: false,
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
      hovermode: 'closest',
      hoverlabel: {
        bgcolor: '#f5f5f0',
        bordercolor: '#c8c8b4',
        font: { family: 'ui-monospace, monospace', size: 12, color: '#2d2d28' },
        align: 'left',
        namelength: 0,
      },
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
                  </div>
                </div>
              )
            })()}

            {/* Plot + Legend */}
            <div className="flex gap-4" style={{ height: 520 }}>
              <div ref={plotRef} className="flex-1 min-w-0 overflow-hidden" />

              {/* Custom legend panel */}
              <div className="flex flex-col w-72 shrink-0 border border-border rounded-sm overflow-hidden"
                style={{ background: 'var(--color-surface2)' }}>
                <div className="px-4 py-2.5 border-b border-border shrink-0"
                  style={{ background: 'var(--color-surface)' }}>
                  <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-faint">
                    Clusters · {legendItems.filter(l => l.id !== '__none__').length}
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto scrollbar-thin py-1.5">
                  {legendItems.map(item => (
                    <div key={item.id}
                      className="flex items-start gap-3 px-4 pr-5 py-2 hover:bg-surface transition-colors group">
                      <div className="shrink-0 mt-[4px] rounded-full" style={{
                        width: 10, height: 10,
                        background: item.color,
                        boxShadow: `0 0 0 2px ${item.color}40`,
                      }} />
                      <span className={`font-mono text-[13px] leading-snug flex-1 break-words ${item.id === '__none__' ? 'text-faint italic' : 'text-ink'}`}>
                        {item.name}
                      </span>
                      <span className="font-mono text-[12px] text-faint shrink-0 tabular-nums mt-px opacity-0 group-hover:opacity-100 transition-opacity">
                        {item.count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

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
