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

// Datasets
export const getDatasets = () => req<Dataset[]>('/datasets')

// Sessions
export const getSessions   = () => req<Session[]>('/sessions')
export const createSession = (body: { dataset_name: string; name: string }) =>
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
  req<UmapData>(`/umap/${sessionId}${geometryAware ? '?geometry_aware=true' : ''}`)
