import React from 'react'
import type { EvalResult } from '../../types'
import Modal from './Modal'

function tierCls(v: number | undefined | null, invert = false): string {
  if (v == null) return 'text-faint'
  const good = invert ? v <= 0.30 : v >= 0.70
  const mid  = invert ? v <= 0.60 : v >= 0.50
  if (good) return 'text-green-700'
  if (mid)  return 'text-amber-600'
  return 'text-red-700'
}

function Row({ label, value, invert, extra }: { label: string; value?: number | null; invert?: boolean; extra?: React.ReactNode }) {
  if (value == null) return null
  return (
    <div className="flex items-start justify-between py-2.5 px-3 border-b border-border last:border-0 gap-3">
      <span className="text-[13px] text-muted leading-snug">{label}</span>
      <div className="flex flex-col items-end gap-1 shrink-0">
        <span className={`font-mono text-[13px] font-bold ${tierCls(value, invert)}`}>{value.toFixed(3)}</span>
        {extra}
      </div>
    </div>
  )
}

interface Props { result: EvalResult; onClose: () => void }

export default function EvalModal({ result, onClose }: Props) {
  return (
    <Modal title="Evaluation results" onClose={onClose} width="max-w-xl">
      <div className="p-5 flex flex-col gap-5">

        {/* Overall */}
        {result.B1 && (
          <div>
            <div className="flex items-baseline gap-3 mb-2">
              <span className={`font-serif text-5xl font-normal leading-none ${tierCls(result.B1.overall_score)}`}>
                {result.B1.overall_score.toFixed(2)}
              </span>
              <span className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint">Overall score</span>
            </div>
            {result.B1.notes && (
              <p className="text-[13px] text-muted leading-relaxed">{result.B1.notes}</p>
            )}
          </div>
        )}

        <div className="h-px bg-border" />

        {/* Coherence */}
        {result.B2 && (
          <div>
            <p className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint mb-2">Coherence</p>
            <div className="rounded-sm border border-border overflow-hidden" style={{ background: 'var(--color-surface)' }}>
              <Row label="Mean coherence" value={result.B2.coherence_mean} />
              <Row label="Min coherence" value={result.B2.coherence_min} />
            </div>
            {result.B2.per_cluster && result.B2.per_cluster.length > 0 && (
              <div className="mt-2 flex flex-col gap-1.5">
                {result.B2.per_cluster.map((c, i) => (
                  <div key={i} className="px-3 py-2 rounded-sm border border-border text-[12px] leading-snug"
                    style={{ background: 'var(--color-surface)' }}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="font-mono text-faint">cluster {i + 1}</span>
                      <span className={`font-mono font-bold ${tierCls(c.coherence)}`}>{c.coherence.toFixed(3)}</span>
                    </div>
                    {c.reasoning && <p className="text-muted">{c.reasoning}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Compliance & Contradiction */}
        {(result.B3 || result.B4) && (
          <div>
            <p className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint mb-2">User alignment</p>
            <div className="rounded-sm border border-border overflow-hidden" style={{ background: 'var(--color-surface)' }}>
              <Row label="Compliance with instructions" value={result.B3?.compliance_score} />
              <Row label="Contradiction score" value={result.B4?.contradiction_score} invert />
            </div>
            {result.B3?.notes && (
              <p className="mt-2 text-[12px] text-muted leading-relaxed px-1">{result.B3.notes}</p>
            )}
            {result.B4?.notes && (
              <p className="mt-1 text-[12px] text-muted leading-relaxed px-1">{result.B4.notes}</p>
            )}
            {result.B4?.examples && result.B4.examples.length > 0 && (
              <div className="mt-2 flex flex-col gap-1">
                {result.B4.examples.map((ex, i) => (
                  <div key={i} className="px-3 py-2 rounded-sm border border-red-100 bg-red-50 text-[12px] text-red-700 leading-snug">{ex}</div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="h-px bg-border" />

        {/* Process metrics */}
        <div>
          <p className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint mb-2">Process metrics</p>
          <div className="rounded-sm border border-border overflow-hidden" style={{ background: 'var(--color-surface)' }}>
            {result.A1 && (
              <div className="flex items-center justify-between py-2.5 px-3 border-b border-border">
                <span className="text-[13px] text-muted">Silhouette</span>
                <div className="flex items-center gap-2 font-mono text-[13px]">
                  <span className="text-faint">{result.A1.silhouette_initial.toFixed(3)}</span>
                  <span className="text-faint">→</span>
                  <span className={`font-bold ${tierCls(result.A1.silhouette_final)}`}>{result.A1.silhouette_final.toFixed(3)}</span>
                  <span className={`text-[11px] ${result.A1.silhouette_final > result.A1.silhouette_initial ? 'text-green-700' : 'text-red-700'}`}>
                    {result.A1.silhouette_final > result.A1.silhouette_initial ? '↑' : '↓'}
                    {Math.abs(result.A1.silhouette_final - result.A1.silhouette_initial).toFixed(3)}
                  </span>
                </div>
              </div>
            )}
            {result.A2 && (
              <div className="flex items-center justify-between py-2.5 px-3 border-b border-border">
                <span className="text-[13px] text-muted">Turns</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[13px] font-bold text-ink">{result.A2.turns}</span>
                  {result.A2.weighted_turns !== undefined && (
                    <span className="font-mono text-[11px] text-faint">({result.A2.weighted_turns.toFixed(1)} weighted)</span>
                  )}
                  <span className="font-mono text-[10px] px-2 py-0.5 rounded-sm border border-border text-faint capitalize">
                    {result.A2.termination}
                  </span>
                </div>
              </div>
            )}
            {result.A3 && (
              <div className="flex items-center justify-between py-2.5 px-3 border-b border-border last:border-0">
                <span className="text-[13px] text-muted">Mean cognitive load</span>
                <span className={`font-mono text-[13px] font-bold ${tierCls(result.A3.mean_cognitive_load != null ? 1 - (result.A3.mean_cognitive_load - 1) / 4 : null)}`}>
                  {result.A3.mean_cognitive_load?.toFixed(1) ?? '–'}
                </span>
              </div>
            )}
          </div>

          {result.A3?.cognitive_load_driver_by_turn && result.A3.cognitive_load_driver_by_turn.length > 0 && (
            <div className="mt-2 flex flex-col gap-1">
              <p className="font-mono text-[10px] text-faint tracking-wider uppercase px-1 mb-0.5">Cognitive load by turn</p>
              {result.A3.cognitive_load_driver_by_turn.map((driver, i) => (
                <div key={i} className="flex gap-2 px-3 py-2 rounded-sm border border-border text-[12px]"
                  style={{ background: 'var(--color-surface)' }}>
                  <span className="font-mono text-faint shrink-0">T{i + 1}</span>
                  <span className="text-muted leading-snug">{driver}</span>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </Modal>
  )
}
