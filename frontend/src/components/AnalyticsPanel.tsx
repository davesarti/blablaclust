import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useApp } from '../store/AppContext'
import { evalSession, getUmap } from '../api/client'
import type { EvalResult, UmapData } from '../types'

const UMAP_COLORS = [
  '#4a7c59','#c06b3e','#3e6fa3','#a3543e','#6b8e23','#8a5a9e',
  '#2e8b8b','#b07a2e','#9e3e5a','#5a6e8a','#7a9e3e','#3e8a5a',
]

function tier(v: number | undefined | null, invert = false): string {
  if (v == null) return 'var(--color-faint)'
  const good = invert ? v <= 0.30 : v >= 0.70
  const mid  = invert ? v <= 0.60 : v >= 0.50
  if (good) return '#3a6e4a'
  if (mid)  return '#9a7a30'
  return '#8a3030'
}

interface ScorePillProps { label: string; value?: number | null; invert?: boolean }
function ScorePill({ label, value, invert }: ScorePillProps) {
  if (value == null) return null
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className="font-mono text-[11px] font-bold" style={{ color: tier(value, invert) }}>
        {value.toFixed(2)}
      </span>
      <span className="font-mono text-[10px] text-faint tracking-wider uppercase">{label}</span>
    </div>
  )
}

interface Props {
  onOpenUmap: () => void
  onOpenEval: (result: import('../types').EvalResult) => void
}

export default function AnalyticsPanel({ onOpenUmap, onOpenEval }: Props) {
  const { state } = useApp()
  const sess = state.session!

  const [umap, setUmap]           = useState<UmapData | null>(null)
  const [umapLoading, setUmapLoading] = useState(false)
  const [umapError, setUmapError] = useState(false)

  const [ev, setEv]               = useState<EvalResult | null>(null)
  const [evLoading, setEvLoading] = useState(false)

  const plotRef = useRef<HTMLDivElement>(null)
  const prevTurnRef = useRef(-1)

  // Load on mount (resumed sessions) and after each new turn
  useEffect(() => {
    if (sess.turnNumber === prevTurnRef.current) return
    prevTurnRef.current = sess.turnNumber
    if (sess.turnNumber === 0) return
    loadUmap()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sess.sessionId, sess.turnNumber])

  async function loadUmap() {
    setUmapLoading(true)
    setUmapError(false)
    try {
      setUmap(await getUmap(sess.sessionId, false))
    } catch {
      setUmapError(true)
    } finally {
      setUmapLoading(false)
    }
  }

  async function runEval() {
    setEvLoading(true)
    try {
      setEv(await evalSession(sess.sessionId))
    } catch {
      // silent
    } finally {
      setEvLoading(false)
    }
  }

  const renderMini = useCallback(async () => {
    if (!umap || !plotRef.current) return
    const Plotly = (await import('plotly.js-dist-min')).default as typeof import('plotly.js')

    const lastTurn  = umap.turns[umap.turns.length - 1]
    const turnStr   = String(lastTurn)
    const assigns   = (umap.assignments[turnStr] ?? []) as (string | null)[]
    const clusterIds = Object.keys(umap.clusters)
    const colorMap: Record<string, string> = {}
    clusterIds.forEach((id, i) => { colorMap[id] = UMAP_COLORS[i % UMAP_COLORS.length] })

    const byCluster: Record<string, { x: number[]; y: number[] }> = {}
    umap.points.forEach((p, i) => {
      const cid = assigns[i] ?? '__none__'
      if (!byCluster[cid]) byCluster[cid] = { x: [], y: [] }
      byCluster[cid].x.push(p.x)
      byCluster[cid].y.push(p.y)
    })

    const traces: Plotly.Data[] = Object.entries(byCluster).map(([cid, d]) => ({
      type: 'scattergl' as const,
      x: d.x, y: d.y,
      mode: 'markers',
      showlegend: false,
      hoverinfo: 'skip' as const,
      marker: { color: cid === '__none__' ? '#bbb' : colorMap[cid], size: 3, opacity: 0.75 },
    }))

    await Plotly.react(plotRef.current, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor:  'rgba(0,0,0,0)',
      margin: { t: 0, r: 0, b: 0, l: 0 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false, fixedrange: true },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x', fixedrange: true },
    }, { responsive: true, displayModeBar: false, staticPlot: true })
  }, [umap])

  useEffect(() => { renderMini() }, [umap, renderMini])

  const silh = ev?.A1?.silhouette_final

  return (
    <div className="shrink-0 flex border-t border-border overflow-hidden"
      style={{ height: 200, background: 'var(--color-surface)' }}>

      {/* ── UMAP side ── */}
      <div className="flex flex-col" style={{ width: 280, borderRight: '1px solid var(--color-border)' }}>
        {/* header */}
        <div className="flex items-center justify-between px-4 pt-3 pb-2 shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-faint">UMAP</span>
            {umap && (
              <span className="font-mono text-[10px] text-faint">
                T{umap.turns[umap.turns.length - 1]}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            {(umap || umapError) && (
              <button onClick={loadUmap} disabled={umapLoading}
                title="Refresh"
                className="font-mono text-[11px] text-faint hover:text-muted transition-colors disabled:opacity-40 px-1">
                ↺
              </button>
            )}
            <button onClick={onOpenUmap}
              className="font-mono text-[11px] font-bold tracking-wider uppercase text-faint hover:text-muted transition-colors px-1">
              expand ↗
            </button>
          </div>
        </div>

        {/* plot area */}
        <div className="flex-1 relative mx-3 mb-3 rounded-sm overflow-hidden"
          style={{ background: 'var(--color-surface2)' }}>
          {!umap && !umapLoading && !umapError && (
            <div className="absolute inset-0 flex items-center justify-center">
              {sess.turnNumber === 0 ? (
                <span className="font-mono text-[11px] text-faint text-center px-4">
                  Run a turn to see the projection
                </span>
              ) : (
                <button onClick={loadUmap}
                  className="font-mono text-[11px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted hover:border-borders transition-colors"
                  style={{ background: 'var(--color-surface)' }}>
                  Load projection
                </button>
              )}
            </div>
          )}
          {umapLoading && (
            <div className="absolute inset-0 flex items-center justify-center gap-2">
              <span className="inline-block w-3 h-3 border-2 rounded-full animate-spin-slow"
                style={{ borderColor: 'var(--color-border)', borderTopColor: 'var(--color-accent)' }} />
              <span className="font-mono text-[10px] text-faint">computing…</span>
            </div>
          )}
          {umapError && (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="font-mono text-[10px] text-faint">failed to load</span>
            </div>
          )}
          <div ref={plotRef} className="w-full h-full" style={{ opacity: umap && !umapLoading ? 1 : 0 }} />
        </div>
      </div>

      {/* ── Eval side ── */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* header */}
        <div className="flex items-center justify-between px-4 pt-3 pb-2 shrink-0">
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-faint">Evaluation</span>
          {ev && (
            <button onClick={() => onOpenEval(ev)}
              className="font-mono text-[11px] font-bold tracking-wider uppercase text-faint hover:text-muted transition-colors px-1">
              expand ↗
            </button>
          )}
        </div>

        {/* body */}
        <div className="flex-1 flex flex-col justify-between px-4 pb-4 min-h-0">
          {!ev && !evLoading && (
            <div className="flex-1 flex flex-col items-start justify-center gap-3">
              <p className="font-mono text-[11px] text-faint leading-relaxed">
                Run LLM judges to score<br />coherence, compliance &amp; contradiction.
              </p>
              <button onClick={runEval}
                className="font-mono text-[11px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted hover:border-borders transition-colors"
                style={{ background: 'var(--color-surface)' }}>
                Run evaluation
              </button>
            </div>
          )}

          {evLoading && (
            <div className="flex-1 flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 rounded-full animate-spin-slow"
                style={{ borderColor: 'var(--color-border)', borderTopColor: 'var(--color-accent)' }} />
              <span className="font-mono text-[11px] text-faint">running judges…</span>
            </div>
          )}

          {ev && !evLoading && (
            <div className="flex flex-col gap-2.5">
              {/* Overall score */}
              {ev.B1 && (
                <div className="flex items-baseline gap-2.5">
                  <span className="font-serif leading-none" style={{ fontSize: 38, color: tier(ev.B1.overall_score) }}>
                    {ev.B1.overall_score.toFixed(2)}
                  </span>
                  <span className="font-mono text-[10px] text-faint tracking-widest uppercase">overall</span>
                  {silh != null && (
                    <>
                      <span className="text-faint text-[11px] ml-1">·</span>
                      <span className="font-mono text-[11px]" style={{ color: tier(silh) }}>
                        silh {silh.toFixed(3)}
                      </span>
                    </>
                  )}
                  <button onClick={runEval}
                    className="ml-auto font-mono text-[10px] text-faint hover:text-muted transition-colors">
                    ↺
                  </button>
                </div>
              )}
              {/* Score pills */}
              <div className="flex items-end gap-5">
                <ScorePill label="coh"  value={ev.B2?.coherence_mean} />
                <ScorePill label="comp" value={ev.B3?.compliance_score} />
                <ScorePill label="cont" value={ev.B4?.contradiction_score} invert />
                {ev.A2 && (
                  <div className="flex flex-col items-center gap-0.5 ml-2">
                    <span className="font-mono text-[11px] font-bold text-ink">{ev.A2.turns}</span>
                    <span className="font-mono text-[10px] text-faint tracking-wider uppercase">turns</span>
                  </div>
                )}
                {ev.A3?.mean_cognitive_load !== undefined && (
                  <div className="flex flex-col items-center gap-0.5">
                    <span className="font-mono text-[11px] font-bold text-ink">
                      {ev.A3.mean_cognitive_load.toFixed(1)}
                    </span>
                    <span className="font-mono text-[10px] text-faint tracking-wider uppercase">cog</span>
                  </div>
                )}
              </div>
              {/* Notes preview */}
              {ev.B1?.notes && (
                <p className="font-mono text-[11px] text-faint leading-relaxed line-clamp-2"
                  style={{ borderLeft: '2px solid var(--color-border)', paddingLeft: 8 }}>
                  {ev.B1.notes}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
