import React, { useCallback, useEffect, useRef, useState } from 'react'
import Modal from './Modal'
import { deleteDataset, getDatasets, previewDataset, uploadDataset } from '../../api/client'
import type { Dataset } from '../../types'
import type { DatasetPreview, DatasetUploadResponse } from '../../api/client'

// ── Dataset row ──────────────────────────────────────────────────────────────

interface DatasetRowProps {
  dataset: Dataset
  onPreview: (id: string) => void
  onDelete: (id: string, name: string) => void
  previewing: boolean
  deleting: boolean
}

function DatasetRow({ dataset, onPreview, onDelete, previewing, deleting }: DatasetRowProps) {
  const [confirmDel, setConfirmDel] = useState(false)
  return (
    <div className="flex items-center gap-4 px-5 py-4 border-b border-border last:border-0 hover:bg-surface2/40 transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[15px] font-bold text-ink truncate">{dataset.dataset_name}</span>
          <span className="font-mono text-[12px] text-faint shrink-0">{dataset.n_points.toLocaleString()} pts</span>
        </div>
        {dataset.description && (
          <div className="text-[13px] text-faint mt-0.5 leading-snug">{dataset.description}</div>
        )}
      </div>

<div className="flex items-center gap-2 shrink-0">
        <button
          onClick={() => onPreview(dataset.dataset_id)}
          disabled={previewing}
          className="font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted hover:border-borders transition-colors disabled:opacity-40">
          {previewing ? '…' : 'preview'}
        </button>

        {confirmDel ? (
          <>
            <button
              onClick={() => { setConfirmDel(false); onDelete(dataset.dataset_id, dataset.dataset_name) }}
              disabled={deleting}
              className="font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 transition-colors disabled:opacity-40">
              {deleting ? '…' : 'confirm'}
            </button>
            <button
              onClick={() => setConfirmDel(false)}
              className="font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted transition-colors">
              cancel
            </button>
          </>
        ) : (
          <button
            onClick={() => setConfirmDel(true)}
            className="font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-red-700 hover:border-red-200 transition-colors">
            del
          </button>
        )}
      </div>
    </div>
  )
}

// ── Preview panel ─────────────────────────────────────────────────────────────

function PreviewPanel({ preview, onClose }: { preview: DatasetPreview; onClose: () => void }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="font-mono text-[13px] font-bold tracking-widest uppercase text-faint">
            {preview.dataset_name}
          </span>
          <span className="font-mono text-[13px] text-faint ml-3">
            {preview.points.length} sample rows
          </span>
        </div>
        <button onClick={onClose}
          className="font-mono text-[12px] text-faint hover:text-muted transition-colors">
          ✕ close preview
        </button>
      </div>

      {preview.description && (
        <p className="text-[14px] text-muted italic leading-relaxed px-1">{preview.description}</p>
      )}

      <div className="rounded-sm border border-border" style={{ background: 'var(--color-surface)' }}>
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border" style={{ background: 'var(--color-surface2)' }}>
          <span className="shrink-0 w-7 font-mono text-[11px] font-bold tracking-widest uppercase text-faint">#</span>
          <span className="flex-1 font-mono text-[11px] font-bold tracking-widest uppercase text-faint">text</span>
        </div>
        {preview.points.map((p, i) => (
          <div key={p.id} className="flex items-start gap-3 px-4 py-2.5"
            style={{ borderBottom: i < preview.points.length - 1 ? '1px solid var(--color-border)' : undefined }}>
            <span className="shrink-0 w-7 font-mono text-[11px] text-faint mt-0.5">{i + 1}</span>
            <span className="flex-1 text-[13px] text-ink leading-relaxed">
              {p.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Upload panel ──────────────────────────────────────────────────────────────

type UploadState = 'idle' | 'uploading' | 'done' | 'error'

function UploadPanel({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [state, setState] = useState<UploadState>('idle')
  const [result, setResult] = useState<DatasetUploadResponse | null>(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) { setFile(f); if (!name) setName(f.name.replace(/\.csv$/i, '')) }
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) { setFile(f); if (!name) setName(f.name.replace(/\.csv$/i, '')) }
  }

  async function handleUpload() {
    if (!file || !name.trim()) return
    setState('uploading'); setError('')
    try {
      const r = await uploadDataset(file, name.trim(), true)
      setResult(r); setState('done'); onUploaded()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e)); setState('error')
    }
  }

  if (state === 'done' && result) {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-[18px]" style={{ color: 'var(--color-accent)' }}>✓</span>
          <span className="font-mono text-[13px] font-bold text-ink">{result.dataset_name}</span>
          <span className="font-mono text-[12px] text-faint">uploaded successfully</span>
        </div>
        <div className="flex gap-6">
          {[
            { label: 'inserted',   value: result.inserted },
            { label: 'skipped',    value: result.skipped },
            { label: 'embeddings', value: result.embeddings_generated },
          ].map(({ label, value }) => (
            <div key={label} className="flex flex-col items-center gap-0.5">
              <span className="font-mono text-[20px] font-bold text-ink">{value.toLocaleString()}</span>
              <span className="font-mono text-[11px] text-faint uppercase tracking-wider">{label}</span>
            </div>
          ))}
        </div>
        <button
          onClick={() => { setFile(null); setName(''); setResult(null); setState('idle') }}
          className="self-start font-mono text-[12px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint hover:text-muted hover:border-borders transition-colors">
          upload another
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className="relative flex flex-col items-center justify-center gap-2 rounded-sm border-2 border-dashed cursor-pointer transition-all duration-150 py-8"
        style={{
          borderColor: dragging ? 'var(--color-accent)' : file ? 'var(--color-borders)' : 'var(--color-border)',
          background: dragging ? 'rgba(74,124,89,.06)' : 'var(--color-surface)',
        }}>
        <input ref={inputRef} type="file" accept=".csv" className="hidden" onChange={onFileChange} />
        {file ? (
          <>
            <span className="font-mono text-[15px] font-bold text-ink">{file.name}</span>
            <span className="font-mono text-[12px] text-faint">{(file.size / 1024).toFixed(1)} KB — click to change</span>
          </>
        ) : (
          <>
            <span className="font-mono text-[13px] text-faint">Drop a CSV file here</span>
            <span className="font-mono text-[11px] text-faint opacity-60">or click to browse · expects: text column</span>
          </>
        )}
      </div>

      <div className="flex gap-3">
        <label className="flex flex-col gap-1.5 flex-1">
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-faint">Dataset name</span>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="e.g. my_dataset"
            className="px-3 py-2.5 rounded-sm border border-border text-[15px] text-ink outline-none focus:border-borders transition-colors"
            style={{ background: 'var(--color-bg)' }}
          />
        </label>

      </div>

      {error && (
        <div className="px-4 py-3 rounded-sm border border-red-200 bg-red-50 text-[13px] text-red-700">{error}</div>
      )}

      <button
        onClick={handleUpload}
        disabled={!file || !name.trim() || state === 'uploading'}
        className="flex items-center justify-center gap-2.5 py-3 rounded-sm text-white font-medium text-[15px] transition-all hover:opacity-90 disabled:opacity-40"
        style={{ background: 'var(--color-accent)' }}>
        {state === 'uploading' ? (
          <>
            <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin-slow" />
            Uploading & generating embeddings…
          </>
        ) : (
          '↑ Upload dataset'
        )}
      </button>
    </div>
  )
}

// ── Main modal ────────────────────────────────────────────────────────────────

interface Props { onClose: () => void }

export default function DatasetsModal({ onClose }: Props) {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [tab, setTab] = useState<'list' | 'upload'>('list')

  const loadDatasets = useCallback(async () => {
    try { setDatasets(await getDatasets()) }
    catch { /* silent */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { loadDatasets() }, [loadDatasets])

  async function handlePreview(id: string) {
    if (previewId === id) { setPreviewId(null); setPreview(null); setPreviewError(null); return }
    setPreviewId(id); setPreviewLoading(true); setPreviewError(null)
    try { setPreview(await previewDataset(id)) }
    catch (e: unknown) {
      setPreview(null)
      setPreviewId(null)
      setPreviewError(e instanceof Error ? e.message : 'Preview failed')
    }
    finally { setPreviewLoading(false) }
  }

  async function handleDelete(id: string, name: string) {
    setDeletingId(id)
    try {
      await deleteDataset(id)
      setDatasets(prev => prev.filter(d => d.dataset_id !== id))
      if (previewId === id) { setPreviewId(null); setPreview(null) }
    } catch { /* silent */ }
    finally { setDeletingId(null) }
  }

  return (
    <Modal title="Datasets" onClose={onClose} width="max-w-5xl">
      <div className="flex flex-col" style={{ minHeight: 400 }}>
        {/* Tab bar */}
        <div className="flex gap-1 px-6 pt-5 pb-0 shrink-0">
          {(['list', 'upload'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className="font-mono text-[12px] font-bold tracking-widest uppercase px-4 py-2 rounded-t-sm transition-colors"
              style={{
                color: tab === t ? 'var(--color-ink)' : 'var(--color-faint)',
                background: tab === t ? 'var(--color-surface)' : 'transparent',
                borderBottom: tab === t ? '2px solid var(--color-accent)' : '2px solid transparent',
              }}>
              {t === 'list' ? `datasets (${datasets.length})` : 'upload new'}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {tab === 'list' && (
            <div className="flex flex-col">
              {loading ? (
                <div className="py-16 text-center font-mono text-[13px] text-faint">loading…</div>
              ) : datasets.length === 0 ? (
                <div className="py-16 text-center">
                  <p className="font-mono text-[13px] text-faint mb-4">No datasets yet.</p>
                  <button onClick={() => setTab('upload')}
                    className="font-mono text-[12px] font-bold tracking-wider uppercase px-4 py-2 rounded-sm border border-border text-faint hover:text-muted hover:border-borders transition-colors">
                    Upload your first dataset →
                  </button>
                </div>
              ) : (
                <>
                  <div className="rounded-sm border-b border-border mx-6 mt-5" style={{ background: 'var(--color-surface)' }}>
                    {/* Table header */}
                    <div className="flex items-center gap-4 px-5 py-2.5 border-b border-border"
                      style={{ background: 'var(--color-surface2)' }}>
                      <span className="flex-1 font-mono text-[11px] font-bold tracking-widest uppercase text-faint">Name</span>
                      <span className="shrink-0 w-48" />
                    </div>
                    {datasets.map(d => (
                      <DatasetRow
                        key={d.dataset_id}
                        dataset={d}
                        onPreview={handlePreview}
                        onDelete={handleDelete}
                        previewing={previewLoading && previewId === d.dataset_id}
                        deleting={deletingId === d.dataset_id}
                      />
                    ))}
                  </div>

                  {/* Preview error */}
                  {previewError && (
                    <div className="mx-6 mt-4 px-4 py-3 rounded-sm border border-red-200 bg-red-50 text-[13px] text-red-700 font-mono">
                      preview error: {previewError}
                    </div>
                  )}

                  {/* Preview panel */}
                  {preview && !previewLoading && (
                    <div className="mx-6 mt-5 mb-2 rounded-sm border border-border p-5"
                      style={{ background: 'var(--color-surface)' }}>
                      <PreviewPanel preview={preview} onClose={() => { setPreview(null); setPreviewId(null) }} />
                    </div>
                  )}
                  {previewLoading && (
                    <div className="mx-6 mt-5 py-6 text-center font-mono text-[13px] text-faint">
                      loading preview…
                    </div>
                  )}
                  <div className="h-6" />
                </>
              )}
            </div>
          )}

          {tab === 'upload' && (
            <div className="p-6">
              <UploadPanel onUploaded={() => { loadDatasets(); }} />
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}
