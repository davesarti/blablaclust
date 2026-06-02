import type {
  Dataset, Session, Cluster, OracleTurn, TurnRead,
  SessionState, ClusterPointsResponse, EvalResult, UmapData,
} from '../types'

const BASE = ''  // proxied by Vite dev server

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

export interface DatasetUploadResponse {
  dataset_id: string; dataset_name: string;
  inserted: number; skipped: number; embeddings_generated: number;
}
export interface DatasetPreview {
  dataset_id: string; dataset_name: string; description: string;
  points: { id: string; data: Record<string, unknown>; has_embedding: boolean }[];
}

// Datasets
export const getDatasets = () => req<Dataset[]>('/datasets')
export const previewDataset = (id: string) => req<DatasetPreview>(`/datasets/${id}/preview`)
export const deleteDataset  = (id: string) => req<{ dataset_id: string; dataset_name: string }>(`/datasets/${id}`, { method: 'DELETE' })
export const uploadDataset  = (file: File, datasetName: string, generateEmbeddings = true): Promise<DatasetUploadResponse> => {
  const form = new FormData()
  form.append('file', file)
  form.append('dataset_name', datasetName)
  form.append('generate_embeddings', String(generateEmbeddings))
  return fetch('/datasets/upload', { method: 'POST', body: form }).then(async res => {
    if (!res.ok) {
      const d = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(d.detail ?? res.statusText)
    }
    return res.json()
  })
}

// Sessions
export const getSessions   = () => req<Session[]>('/sessions')
export const createSession = (body: { dataset_id: string; name: string }) =>
  req<{ id: string }>('/sessions', { method: 'POST', body: JSON.stringify(body) })
export const getSessionState = (id: string) => req<SessionState>(`/sessions/${id}/state`)
export const patchSessionState = (id: string, status: string) =>
  req<void>(`/sessions/${id}/state`, { method: 'PATCH', body: JSON.stringify({ status }) })
export const deleteSession = (id: string) =>
  req<void>(`/sessions/${id}/delete`, { method: 'DELETE' })
export const evalSession = (id: string) => req<EvalResult>(`/sessions/${id}/eval`, { method: 'POST' })

// Clusters
export const getActiveClusters = (sessionId: string) =>
  req<Cluster[]>(`/clusters/active?session_id=${sessionId}`)
export const initClustering = (sessionId: string, k: number) =>
  req<void>(`/clusters/${sessionId}`, { method: 'POST', body: JSON.stringify({ k, generate_names: true }) })
export const getClusterPoints = (clusterId: string) =>
  req<ClusterPointsResponse>(`/clusters/${clusterId}/points`)

// Turns
export const sendTurn = (turn: OracleTurn) =>
  req<TurnRead>('/turns', { method: 'POST', body: JSON.stringify(turn) })
export const getTurns = (sessionId: string) => req<TurnRead[]>(`/turns/${sessionId}`)

// UMAP
export const getUmap = (sessionId: string, geometryAware = false) =>
  req<UmapData>(`/sessions/${sessionId}/umap${geometryAware ? '?geometry_aware=true' : ''}`)
