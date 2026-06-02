export interface Dataset {
  dataset_id: string;
  dataset_name: string;
  n_points: number;
  has_embeddings: boolean;
  description?: string;
}

export interface Session {
  id: string;
  name: string;
  dataset_name: string;
  status: 'active' | 'converged' | 'closed';
}

export interface Cluster {
  id: string;
  name: string;
  description: string;
  size: number;
  representative_points?: string[];
}

export interface OracleTurn {
  session_id: string;
  raw_text: string;
  feedback_type: 'global' | 'cluster' | 'point' | 'instructional';
  target_cluster_ids: string[];
  target_point_ids: string[];
  metadata: Record<string, unknown>;
}

export interface DisplayItem {
  name: string;
  size?: number;
  text_preview?: string;
  uncertainty_score?: number;
}

export interface SystemOutput {
  action: string;
  clusters_updated: boolean;
  display: {
    type: string;
    content: string;
    items: DisplayItem[];
  };
  cognitive_load_score: number;
  token_usage?: { input_tokens: number; output_tokens: number };
  cost_usd?: number;
  state_snapshot?: Record<string, unknown>;
}

export interface TurnRead {
  session_id: string;
  turn_number: number;
  oracle_input: OracleTurn;
  system_output: SystemOutput;
}

export interface SessionState {
  session_id: string;
  turn_number: number;
  dataset_name: string;
  status: 'active' | 'converged' | 'closed';
}

export interface ClusterPoint {
  id: string;
  data: Record<string, unknown>;
  probability: number;
}

export interface ClusterPointsResponse {
  cluster_id: string;
  session_id: string;
  turn_number?: number;
  points: ClusterPoint[];
}

// Evaluation
export interface EvalResult {
  B1?: { overall_score: number; notes: string };
  B2?: { coherence_mean: number; coherence_min: number; per_cluster: { coherence: number; reasoning: string }[] };
  B3?: { compliance_score: number; notes: string };
  B4?: { contradiction_score: number; notes: string; examples: string[] };
  A2?: { turns: number; weighted_turns: number; termination: string };
  A3?: { mean_cognitive_load: number; cognitive_load_driver_by_turn: string[] };
  A1?: { silhouette_initial: number; silhouette_final: number };
}

// UMAP
export interface UmapPoint { x: number; y: number; text: string }
export interface UmapArrow { from: [number, number]; to: [number, number]; label: string }
export interface UmapGeomTurn { points: UmapPoint[]; axis_label: string }

export interface UmapData {
  session_id: string;
  dataset_name: string;
  reducer: string;
  n_points: number;
  turns: number[];
  points: UmapPoint[];
  assignments: Record<string, (string | null)[]>;
  clusters: Record<string, { name: string; created_at_turn: number; dissolved_at_turn?: number }>;
  silhouette_by_turn: Record<string, number | null>;
  centroids_by_turn: Record<string, Record<string, [number, number]>>;
  reembed_turns: number[];
  axis_arrows: Record<string, UmapArrow>;
  geometry_aware?: Record<string, UmapGeomTurn>;
}

// App state
export interface ChatMessage {
  role: 'user' | 'system';
  text: string;
  turnNumber?: number;
}

export interface AppSessionState {
  sessionId: string;
  session: Session;
  clusters: Cluster[];
  chat: ChatMessage[];
  turnNumber: number;
  tokenUsage: { input: number; output: number };
  costUsd: number;
  cognitiveLoad: number;
  selectedClusterIds: Set<string>;
  isBusy: boolean;
}
