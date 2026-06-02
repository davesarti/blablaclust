import React, { useEffect, useState } from 'react'
import { useApp } from '../../store/AppContext'
import { getDatasets, createSession, initClustering } from '../../api/client'
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

  async function handleCreate() {
    if (!datasetId) return
    setLoading(true)
    setError('')
    try {
      const dsName = selectedDataset?.dataset_name ?? datasetId
      const { id } = await createSession({ dataset_id: datasetId, name: name.trim() || dsName })
      await initClustering(id, k)
      const session: Session = { id, name: name.trim() || dsName, dataset_name: dsName, status: 'active' }
      dispatch({ type: 'CLOSE_MODAL' })
      onCreated(session)
    } catch (e: unknown) {
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
          <span className="font-mono text-[15px] font-bold tracking-widest uppercase text-faint">Session name</span>
          <input value={name} onChange={e => setName(e.target.value)}
            placeholder="Optional name…"
            className="px-3 py-2 rounded-sm border border-border text-[14px] text-ink outline-none focus:border-borders transition-colors"
            style={{ background: 'var(--color-surface)' }} />
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[15px] font-bold tracking-widest uppercase text-faint">Dataset</span>
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
          <span className="font-mono text-[15px] font-bold tracking-widest uppercase text-faint">Initial clusters (k)</span>
          <input type="number" value={k} min={2} max={20} onChange={e => setK(+e.target.value)}
            className="px-3 py-2 rounded-sm border border-border text-[14px] text-ink outline-none focus:border-borders transition-colors"
            style={{ background: 'var(--color-surface)' }} />
        </label>

        <div className="flex gap-2 pt-1">
          <button onClick={() => dispatch({ type: 'CLOSE_MODAL' })}
            className="flex-1 py-2 rounded-sm border border-border text-[15px] font-medium text-muted hover:bg-surface2 transition-colors">
            Cancel
          </button>
          <button onClick={handleCreate} disabled={loading || !datasetId}
            className="flex-1 py-2 rounded-sm text-[15px] font-medium text-white transition-all hover:opacity-90 disabled:opacity-40"
            style={{ background: 'var(--color-accent)' }}>
            {loading ? 'Starting…' : 'Start session'}
          </button>
        </div>
      </div>
    </Modal>
  )
}
