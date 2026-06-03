import React, { useEffect, useState } from 'react'
import { useApp } from '../../store/AppContext'
import { getDatasets, createSession, initClustering, deleteSession } from '../../api/client'
import type { Dataset, Session } from '../../types'
import Modal from './Modal'

interface Props { onCreated: (s: Session) => void }

export default function NewSessionModal({ onCreated }: Props) {
  const { state, dispatch } = useApp()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [name, setName] = useState('')
  const [k, setK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (state.modal !== 'new-session') return
    getDatasets().then(d => { setDatasets(d); if (d.length) setDatasetId(d[0].dataset_id) }).catch(() => {})
  }, [state.modal])

  if (state.modal !== 'new-session') return null

  const selectedDataset = datasets.find(d => d.dataset_id === datasetId)

  // k must be a whole number in [2, 20]. The <input type="number"> min/max only
  // constrain the spinner arrows — typing or pasting a value (e.g. 0) bypasses
  // them — so we validate explicitly here and gate both the button and submit.
  const K_MIN = 2
  const K_MAX = 20
  const kValid = Number.isInteger(k) && k >= K_MIN && k <= K_MAX

  async function handleCreate() {
    if (!datasetId) return
    if (!kValid) {
      setError(`Initial clusters (k) must be a whole number between ${K_MIN} and ${K_MAX}.`)
      return
    }
    setLoading(true)
    setError('')
    // Track the session id separately: createSession persists a row immediately,
    // but the session is only meaningful once initClustering succeeds. If
    // clustering fails (bad k, server error, …) we roll the session back so
    // failed attempts never leave empty "zombie" sessions behind.
    let createdId: string | null = null
    try {
      const dsName = selectedDataset?.dataset_name ?? datasetId
      const { id } = await createSession({ dataset_id: datasetId, name: name.trim() || dsName })
      createdId = id
      await initClustering(id, k)
      const session: Session = { id, name: name.trim() || dsName, dataset_name: dsName, status: 'active' }
      dispatch({ type: 'CLOSE_MODAL' })
      onCreated(session)
    } catch (e: unknown) {
      if (createdId) {
        // Best-effort cleanup of the orphaned session; ignore cleanup errors.
        try { await deleteSession(createdId) } catch { /* noop */ }
      }
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal title="New session" onClose={() => dispatch({ type: 'CLOSE_MODAL' })}>
      <div className="p-5 flex flex-col gap-4">
        {error && (
          <div className="px-3 py-2.5 rounded-sm border border-red-200 bg-red-50 text-red-700 text-[15px]">{error}</div>
        )}

        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[12px] font-bold tracking-widest uppercase text-faint">Session name</span>
          <input value={name} onChange={e => setName(e.target.value)}
            placeholder="Optional name…"
            className="px-3 py-2 rounded-sm border border-border text-[14px] text-ink outline-none focus:border-borders transition-colors"
            style={{ background: 'var(--color-surface)' }} />
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[12px] font-bold tracking-widest uppercase text-faint">Dataset</span>
          <select value={datasetId} onChange={e => setDatasetId(e.target.value)}
            className="px-3 py-2 rounded-sm border border-border text-[14px] text-ink outline-none focus:border-borders transition-colors cursor-pointer"
            style={{ background: 'var(--color-surface)' }}>
            {datasets.map(d => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_name} — {d.n_points} pts
              </option>
            ))}
          </select>
          {selectedDataset?.description && (
            <p className="text-[14px] text-faint leading-snug px-1">{selectedDataset.description}</p>
          )}
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[12px] font-bold tracking-widest uppercase text-faint">Initial clusters (k)</span>
          <input type="number" value={Number.isNaN(k) ? '' : k} min={K_MIN} max={K_MAX} step={1}
            onChange={e => setK(e.target.value === '' ? NaN : Math.trunc(+e.target.value))}
            aria-invalid={!kValid}
            className="px-3 py-2 rounded-sm border text-[14px] text-ink outline-none transition-colors"
            style={{ background: 'var(--color-surface)', borderColor: kValid ? 'var(--color-border)' : '#fca5a5' }} />
          {!kValid && (
            <span className="text-[13px] text-red-600 px-1">Enter a whole number between {K_MIN} and {K_MAX}.</span>
          )}
        </label>

        <div className="flex gap-2 pt-1">
          <button onClick={() => dispatch({ type: 'CLOSE_MODAL' })}
            className="flex-1 py-2 rounded-sm border border-border text-[15px] font-medium text-muted hover:bg-surface2 transition-colors">
            Cancel
          </button>
          <button onClick={handleCreate} disabled={loading || !datasetId || !kValid}
            className="flex-1 py-2 rounded-sm text-[15px] font-medium text-white transition-all hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: 'var(--color-accent)' }}>
            {loading ? 'Starting…' : 'Start session'}
          </button>
        </div>
      </div>
    </Modal>
  )
}
