import React from 'react'
import type { EvalResult } from '../../types'
import Modal from './Modal'

function tierColor(v: number | undefined | null, invert = false): string {
  if (v == null) return 'var(--color-faint)'
  const good = invert ? v <= 0.30 : v >= 0.70
  const mid  = invert ? v <= 0.60 : v >= 0.50
  if (good) return 'var(--color-green)'
  if (mid)  return 'var(--color-amber)'
  return 'var(--color-red)'
}

function tierCls(v: number | undefined | null, invert = false): string {
  if (v == null) return 'text-faint'
  const good = invert ? v <= 0.30 : v >= 0.70
  const mid  = invert ? v <= 0.60 : v >= 0.50
  if (good) return 'text-green-700'
  if (mid)  return 'text-amber-600'
  return 'text-red-700'
}

function Bar({ value, invert = false }: { value: number; invert?: boolean }) {
  const pct = Math.round(Math.min(1, Math.max(0, invert ? 1 - value : value)) * 100)
  return (
    <div className="w-full rounded-full overflow-hidden" style={{ height: 2, background: 'var(--color-border)' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: tierColor(value, invert), transition: 'width 0.35s ease' }} />
    </div>
  )
}

function Metric({ label, value, invert }: { label: string; value?: number | null; invert?: boolean }) {
  if (value == null) return null
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[16px] text-muted">{label}</span>
        <span className={`font-mono text-[17px] font-bold tabular-nums ${tierCls(value, invert)}`}>{value.toFixed(3)}</span>
      </div>
      <Bar value={value} invert={invert} />
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="font-mono text-[13px] font-bold tracking-[0.15em] uppercase text-faint mb-4">{children}</p>
}

interface Props { result: EvalResult; onClose: () => void }

export default function EvalModal({ result, onClose }: Props) {
  const hasCoherence   = !!result.B2
  const hasAlignment   = !!(result.B3 || result.B4)
  const hasBoth        = hasCoherence && hasAlignment

  return (
    <Modal title="Evaluation results" onClose={onClose} width="max-w-4xl">
      <div className="px-8 py-7 flex flex-col gap-8">

        {/* ── Overall score ─────────────────────────────────────── */}
        {result.B1 && (
          <div className="flex items-start gap-10">
            <div className="shrink-0">
              <span className={`font-serif leading-none ${tierCls(result.B1.overall_score)}`} style={{ fontSize: 84 }}>
                {result.B1.overall_score.toFixed(2)}
              </span>
              <Bar value={result.B1.overall_score} />
              <p className="font-mono text-[13px] tracking-[0.15em] uppercase text-faint mt-1.5">Overall score</p>
            </div>
            <div className="flex-1 pt-2 pl-8" style={{ borderLeft: '1px solid var(--color-border)' }}>
              {result.B1.notes
                ? <p className="text-[16px] text-muted leading-relaxed">{result.B1.notes}</p>
                : <p className="text-[15px] text-faint italic">No notes.</p>
              }
            </div>
          </div>
        )}

        <div className="h-px bg-border" />

        {/* ── Coherence + User alignment ────────────────────────── */}
        {(hasCoherence || hasAlignment) && (
          <div className={hasBoth ? 'grid grid-cols-2 gap-12' : ''}>

            {/* Coherence */}
            {result.B2 && (
              <div className="flex flex-col gap-4">
                <SectionLabel>Coherence</SectionLabel>

                <Metric label="Mean coherence" value={result.B2.coherence_mean} />
                <Metric label="Min coherence"  value={result.B2.coherence_min} />

                {result.B2.per_cluster && result.B2.per_cluster.length > 0 && (
                  <div className="flex flex-col gap-4 pt-2" style={{ borderTop: '1px solid var(--color-border)' }}>
                    <p className="font-mono text-[13px] tracking-[0.15em] uppercase text-faint">Per cluster</p>
                    {result.B2.per_cluster.map((c, i) => (
                      <div key={i}>
                        <div className="flex items-baseline justify-between mb-1.5">
                          <span className="text-[16px] text-muted">Cluster {i + 1}</span>
                          <span className={`font-mono text-[17px] font-bold tabular-nums ${tierCls(c.coherence)}`}>{c.coherence.toFixed(3)}</span>
                        </div>
                        <Bar value={c.coherence} />
                        {c.reasoning && <p className="text-[15px] text-faint leading-snug mt-1.5">{c.reasoning}</p>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* User alignment */}
            {hasAlignment && (
              <div className="flex flex-col gap-4">
                <SectionLabel>User alignment</SectionLabel>

                <Metric label="Compliance with instructions" value={result.B3?.compliance_score} />
                <Metric label="Contradiction score" value={result.B4?.contradiction_score} invert />

                {(result.B3?.notes || result.B4?.notes) && (
                  <div className="flex flex-col gap-2 pt-1">
                    {result.B3?.notes && <p className="text-[15px] text-muted leading-relaxed">{result.B3.notes}</p>}
                    {result.B4?.notes && <p className="text-[15px] text-muted leading-relaxed">{result.B4.notes}</p>}
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        <div className="h-px bg-border" />

        {/* ── Process metrics ───────────────────────────────────── */}
        <div className="flex flex-col gap-5">
          <SectionLabel>Process</SectionLabel>

          {result.A1 && (
            <div>
              <div className="flex items-baseline justify-between mb-1.5">
                <span className="text-[16px] text-muted">Silhouette</span>
                <div className="flex items-center gap-2 font-mono text-[16px]">
                  <span className="text-faint tabular-nums">{result.A1.silhouette_initial.toFixed(3)}</span>
                  <span className="text-faint text-[11px]">→</span>
                  <span className={`font-bold tabular-nums ${tierCls(result.A1.silhouette_final)}`}>{result.A1.silhouette_final.toFixed(3)}</span>
                  <span className={`text-[14px] font-bold tabular-nums ${result.A1.silhouette_final > result.A1.silhouette_initial ? 'text-green-700' : 'text-red-700'}`}>
                    {result.A1.silhouette_final > result.A1.silhouette_initial ? '↑' : '↓'}
                    {Math.abs(result.A1.silhouette_final - result.A1.silhouette_initial).toFixed(3)}
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                <div className="flex-1 rounded-full overflow-hidden" style={{ height: 2, background: 'var(--color-border)' }}>
                  <div style={{ width: `${Math.min(1, Math.max(0, result.A1.silhouette_initial)) * 100}%`, height: '100%', background: 'var(--color-borders)' }} />
                </div>
                <div className="flex-1 rounded-full overflow-hidden" style={{ height: 2, background: 'var(--color-border)' }}>
                  <div style={{ width: `${Math.min(1, Math.max(0, result.A1.silhouette_final)) * 100}%`, height: '100%', background: tierColor(result.A1.silhouette_final) }} />
                </div>
              </div>
            </div>
          )}

          {result.A2 && (
            <div className="flex items-center justify-between">
              <span className="text-[16px] text-muted">Turns</span>
              <div className="flex items-center gap-2 font-mono text-[16px]">
                <span className="font-bold text-ink">{result.A2.turns}</span>
                {result.A2.weighted_turns != null && (
                  <span className="text-faint text-[14px]">({result.A2.weighted_turns.toFixed(1)} weighted)</span>
                )}
                <span className="text-[11px] px-1.5 py-0.5 border border-border text-faint capitalize" style={{ background: 'var(--color-surface2)' }}>
                  {result.A2.termination}
                </span>
              </div>
            </div>
          )}

          {result.A3?.mean_cognitive_load != null && (
            <div>
              <div className="flex items-baseline justify-between mb-1.5">
                <span className="text-[16px] text-muted">Mean cognitive load</span>
                <span className={`font-mono text-[17px] font-bold tabular-nums ${tierCls(1 - (result.A3.mean_cognitive_load - 1) / 4)}`}>
                  {result.A3.mean_cognitive_load.toFixed(1)}<span className="text-faint font-normal text-[11px]"> / 5</span>
                </span>
              </div>
              <Bar value={1 - (result.A3.mean_cognitive_load - 1) / 4} />
            </div>
          )}
        </div>

        {/* ── Cognitive load by turn ────────────────────────────── */}
        {result.A3?.cognitive_load_driver_by_turn && result.A3.cognitive_load_driver_by_turn.length > 0 && (
          <div>
            <SectionLabel>Cognitive load by turn</SectionLabel>
            <div className="grid grid-cols-2 gap-x-10 gap-y-3">
              {result.A3.cognitive_load_driver_by_turn.map((driver, i) => (
                <div key={i} className="flex gap-3">
                  <span className="font-mono text-[13px] text-faint shrink-0 w-6 pt-0.5">T{i + 1}</span>
                  <span className="text-[14px] text-muted leading-snug">{driver}</span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </Modal>
  )
}
