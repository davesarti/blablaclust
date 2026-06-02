import React, { useEffect, useState } from 'react'
import { evalSession } from '../../api/client'
import type { EvalResult } from '../../types'
import Modal from './Modal'

function tier(v: number | undefined, invert = false): string {
  if (v === undefined) return 'text-faint'
  const good = invert ? v <= 0.30 : v >= 0.70
  const mid  = invert ? v <= 0.60 : v >= 0.50
  if (good) return 'text-green-700'
  if (mid)  return 'text-amber-600'
  return 'text-red-700'
}

function Score({ label, value, invert }: { label: string; value?: number; invert?: boolean }) {
  if (value === undefined) return null
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <span className="text-[13px] text-muted">{label}</span>
      <span className={`font-mono text-[13px] font-bold ${tier(value, invert)}`}>{value.toFixed(2)}</span>
    </div>
  )
}

interface Props { sessionId: string; onClose: () => void }

export default function EvalModal({ sessionId, onClose }: Props) {
  const [result, setResult] = useState<EvalResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    evalSession(sessionId)
      .then(setResult)
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [sessionId])

  return (
    <Modal title="Evaluation" onClose={onClose} width="max-w-lg">
      <div className="p-5">
        {loading && <div className="py-8 text-center text-faint text-sm">Running evaluation…</div>}
        {error && <div className="px-3 py-2.5 rounded-sm border border-red-200 bg-red-50 text-red-700 text-[13px]">{error}</div>}

        {result && (
          <div className="flex flex-col gap-5">
            {/* Overall */}
            {result.B1 && (
              <div>
                <div className="flex items-baseline gap-3 mb-1">
                  <span className={`font-serif text-4xl font-normal ${tier(result.B1.overall_score)}`}>
                    {result.B1.overall_score.toFixed(2)}
                  </span>
                  <span className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint">Overall</span>
                </div>
                {result.B1.notes && <p className="text-[13px] text-muted leading-relaxed">{result.B1.notes}</p>}
              </div>
            )}

            <div className="h-px bg-border" />

            {/* Quality judges */}
            <div>
              <p className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint mb-3">Quality</p>
              <div className="rounded-sm border border-border overflow-hidden" style={{ background: 'var(--color-surface)' }}>
                <Score label="Coherence (mean)" value={result.B2?.coherence_mean} />
                <Score label="Coherence (min)" value={result.B2?.coherence_min} />
                <Score label="Compliance" value={result.B3?.compliance_score} />
                <Score label="Contradiction" value={result.B4?.contradiction_score} invert />
              </div>
              {result.B4?.examples && result.B4.examples.length > 0 && (
                <div className="mt-2 flex flex-col gap-1">
                  {result.B4.examples.slice(0, 3).map((ex, i) => (
                    <div key={i} className="px-3 py-2 rounded-sm border border-red-100 bg-red-50 text-[12px] text-red-700">{ex}</div>
                  ))}
                </div>
              )}
            </div>

            {/* Process metrics */}
            <div>
              <p className="font-mono text-[10px] font-bold tracking-widest uppercase text-faint mb-3">Process</p>
              <div className="rounded-sm border border-border overflow-hidden" style={{ background: 'var(--color-surface)' }}>
                {result.A2 && (
                  <div className="flex items-center justify-between py-2 px-3 border-b border-border">
                    <span className="text-[13px] text-muted">Turns</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[13px] font-bold text-ink">{result.A2.turns}</span>
                      <span className="font-mono text-[10px] px-2 py-0.5 rounded-sm border border-border text-faint capitalize">
                        {result.A2.termination}
                      </span>
                    </div>
                  </div>
                )}
                {result.A3 && (
                  <div className="flex items-center justify-between py-2 px-3 border-b border-border">
                    <span className="text-[13px] text-muted">Mean cognitive load</span>
                    <span className={`font-mono text-[13px] font-bold ${tier(1 - (result.A3.mean_cognitive_load - 1) / 4)}`}>
                      {result.A3.mean_cognitive_load.toFixed(1)}
                    </span>
                  </div>
                )}
                {result.A1 && (
                  <div className="flex items-center justify-between py-2 px-3">
                    <span className="text-[13px] text-muted">Silhouette</span>
                    <div className="flex items-center gap-2 font-mono text-[13px]">
                      <span className="text-faint">{result.A1.silhouette_initial.toFixed(3)}</span>
                      <span className="text-faint">→</span>
                      <span className={`font-bold ${tier(result.A1.silhouette_final)}`}>{result.A1.silhouette_final.toFixed(3)}</span>
                      <span className={`text-[11px] ${result.A1.silhouette_final > result.A1.silhouette_initial ? 'text-green-700' : 'text-red-700'}`}>
                        {result.A1.silhouette_final > result.A1.silhouette_initial ? '↑' : '↓'}
                        {Math.abs(result.A1.silhouette_final - result.A1.silhouette_initial).toFixed(3)}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </Modal>
  )
}
