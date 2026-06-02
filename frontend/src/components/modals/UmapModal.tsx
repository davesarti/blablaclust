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
    const title = `Turn ${turn}${silh != null ? `  ·  silhouette ${silh.toFixed(3)}` : ''}${geomTurn ? `  ·  ${geomTurn.axis_label}` : ''}`

    await Plotly.react(plotRef.current, traces, {
      title: { text: title, font: { size: 11, color: '#787868', family: 'ui-monospace' } },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { t: 32, r: 12, b: 24, l: 32 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
      legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
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
    <Modal title="UMAP clustering evolution" onClose={onClose} width="max-w-4xl">
      <div className="p-4 flex flex-col gap-3">
        {loading && <div className="py-12 text-center text-faint text-sm">Computing projection…</div>}
        {error && <div className="px-3 py-2.5 rounded-sm border border-red-200 bg-red-50 text-red-700 text-[13px]">{error}</div>}

        {data && !loading && (
          <>
            {/* Controls */}
            <div className="flex items-center gap-3 flex-wrap">
              <button onClick={() => setPlaying(p => !p)}
                className="font-mono text-[10px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-muted hover:bg-surface2 transition-colors">
                {playing ? '⏸ pause' : '▶ play'}
              </button>
              <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                <span className="font-mono text-[10px] text-faint shrink-0">T{data.turns[0]}</span>
                <input type="range" min={0} max={data.turns.length - 1} value={turnIdx}
                  onChange={e => { setPlaying(false); setTurnIdx(+e.target.value) }}
                  className="flex-1 accent-accent" />
                <span className="font-mono text-[10px] text-faint shrink-0">T{data.turns[data.turns.length - 1]}</span>
              </div>
              <span className="font-mono text-[11px] text-muted font-bold">Turn {data.turns[turnIdx]}</span>
              {data.geometry_aware && (
                <label className="flex items-center gap-1.5 font-mono text-[10px] font-bold tracking-wider uppercase text-faint cursor-pointer">
                  <input type="checkbox" checked={geomAware} onChange={e => toggleGeom(e.target.checked)} className="accent-accent" />
                  geom-aware
                </label>
              )}
            </div>

            {/* Plot */}
            <div ref={plotRef} className="w-full" style={{ height: 440 }} />

            {/* Reducer info */}
            <p className="font-mono text-[10px] text-faint text-right">
              {data.reducer} · {data.n_points} pts
            </p>
          </>
        )}
      </div>
    </Modal>
  )
}
