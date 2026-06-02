import React, { createContext, useContext, useReducer, useCallback } from 'react'
import type { AppSessionState, Cluster, ChatMessage, Session } from '../types'

type View = 'welcome' | 'workspace'

interface AppState {
  view: View
  session: AppSessionState | null
  // modal flags
  modal: 'new-session' | 'stop' | 'expand' | 'eval' | 'umap' | null
  expandClusterId: string | null
  evalSessionId: string | null
}

type Action =
  | { type: 'GO_WORKSPACE'; session: AppSessionState }
  | { type: 'GO_WELCOME' }
  | { type: 'SET_CLUSTERS'; clusters: Cluster[] }
  | { type: 'SET_BUSY'; busy: boolean }
  | { type: 'APPEND_CHAT'; message: ChatMessage }
  | { type: 'TOGGLE_CLUSTER_SELECT'; clusterId: string }
  | { type: 'CLEAR_CLUSTER_SELECT' }
  | { type: 'UPDATE_METRICS'; turnNumber: number; tokenInput: number; tokenOutput: number; cost: number; cogLoad: number }
  | { type: 'UPDATE_STATUS'; status: 'active' | 'converged' | 'closed' }
  | { type: 'OPEN_MODAL'; modal: AppState['modal']; clusterId?: string; sessionId?: string }
  | { type: 'CLOSE_MODAL' }

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'GO_WORKSPACE':
      return { ...state, view: 'workspace', session: action.session }
    case 'GO_WELCOME':
      return { ...state, view: 'welcome', session: null, modal: null }
    case 'SET_CLUSTERS':
      if (!state.session) return state
      return { ...state, session: { ...state.session, clusters: action.clusters } }
    case 'SET_BUSY':
      if (!state.session) return state
      return { ...state, session: { ...state.session, isBusy: action.busy } }
    case 'APPEND_CHAT':
      if (!state.session) return state
      return { ...state, session: { ...state.session, chat: [...state.session.chat, action.message] } }
    case 'TOGGLE_CLUSTER_SELECT': {
      if (!state.session) return state
      const next = new Set(state.session.selectedClusterIds)
      next.has(action.clusterId) ? next.delete(action.clusterId) : next.add(action.clusterId)
      return { ...state, session: { ...state.session, selectedClusterIds: next } }
    }
    case 'CLEAR_CLUSTER_SELECT':
      if (!state.session) return state
      return { ...state, session: { ...state.session, selectedClusterIds: new Set() } }
    case 'UPDATE_METRICS':
      if (!state.session) return state
      return {
        ...state, session: {
          ...state.session,
          turnNumber: action.turnNumber,
          tokenUsage: {
            input:  state.session.tokenUsage.input  + action.tokenInput,
            output: state.session.tokenUsage.output + action.tokenOutput,
          },
          costUsd: state.session.costUsd + action.cost,
          cognitiveLoad: action.cogLoad,
        },
      }
    case 'UPDATE_STATUS':
      if (!state.session) return state
      return { ...state, session: { ...state.session, session: { ...state.session.session, status: action.status } } }
    case 'OPEN_MODAL':
      return { ...state, modal: action.modal, expandClusterId: action.clusterId ?? null, evalSessionId: action.sessionId ?? null }
    case 'CLOSE_MODAL':
      return { ...state, modal: null, expandClusterId: null, evalSessionId: null }
    default:
      return state
  }
}

const initial: AppState = { view: 'welcome', session: null, modal: null, expandClusterId: null, evalSessionId: null }

interface Ctx {
  state: AppState
  dispatch: React.Dispatch<Action>
  goWorkspace: (session: AppSessionState) => void
  goWelcome: () => void
}

const AppCtx = createContext<Ctx | null>(null)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial)
  const goWorkspace = useCallback((s: AppSessionState) => dispatch({ type: 'GO_WORKSPACE', session: s }), [])
  const goWelcome   = useCallback(() => dispatch({ type: 'GO_WELCOME' }), [])
  return <AppCtx.Provider value={{ state, dispatch, goWorkspace, goWelcome }}>{children}</AppCtx.Provider>
}

export function useApp() {
  const ctx = useContext(AppCtx)
  if (!ctx) throw new Error('useApp must be inside AppProvider')
  return ctx
}

export type { AppState, Action, View }
export function makeSession(session: Session, clusters: Cluster[], turns: import('../types').TurnRead[]): AppSessionState {
  let tokenIn = 0, tokenOut = 0, cost = 0, cogLoad = 1
  const chat: ChatMessage[] = []

  for (const t of turns) {
    chat.push({ role: 'user', text: t.oracle_input.raw_text, turnNumber: t.turn_number })
    const so = t.system_output
    if (so?.display?.content) {
      chat.push({ role: 'system', text: so.display.content, turnNumber: t.turn_number })
    }
    if (so?.token_usage) {
      tokenIn  += so.token_usage.input_tokens ?? 0
      tokenOut += so.token_usage.output_tokens ?? 0
    }
    if (so?.cost_usd) cost += so.cost_usd
    if (so?.cognitive_load_score) cogLoad = so.cognitive_load_score
  }

  return {
    sessionId: session.id,
    session,
    clusters,
    chat,
    turnNumber: turns.length > 0 ? turns[turns.length - 1].turn_number : 0,
    tokenUsage: { input: tokenIn, output: tokenOut },
    costUsd: cost,
    cognitiveLoad: cogLoad,
    selectedClusterIds: new Set(),
    isBusy: false,
  }
}
