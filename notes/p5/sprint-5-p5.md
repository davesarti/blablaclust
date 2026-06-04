# Sprint 5 — P5 (Arianna)

## Consegnato questo sprint

### React frontend — riscrittura completa (`frontend/`) ✅

Sostituzione del prototipo HTML (`ui/index.html`) con una SPA React + TypeScript
+ Tailwind v4 (commit `81ea060`, 2026-06-02).

**Stack:** Vite + React 18 + TypeScript + Tailwind v4 + Plotly.js

**Componenti principali:**
- `WelcomePage.tsx` — landing animata con lista sessioni e creazione nuova sessione
- `WorkspacePage.tsx` — layout a due pannelli: griglia cluster + chat sidebar fissa
- `ChatPanel.tsx` — interfaccia conversazionale con history, typing indicator, invio
- `ClusterCard.tsx` — card cluster con chip selezionabili, expand, pin punti
- `AnalyticsPanel.tsx` — UMAP + visualizzazione evoluzione per turno
- `store/AppContext.tsx` — stato globale applicazione
- `api/client.ts` — client API tipizzato

**Modali:**
- `DatasetsModal.tsx` — upload CSV, anteprima, delete dataset
- `NewSessionModal.tsx` — configurazione nuova sessione
- `EvalModal.tsx` — metriche A1–B4 per sessione
- `UmapModal.tsx` — proiezione UMAP geometria-aware per turno
- `ExpandClusterModal.tsx` — ricerca e pin punti per cluster
- `PersonaModal.tsx` — sessioni persona nel UI
- `StopSessionModal.tsx` — stop + export

Fix successivi: TypeScript build errors, null metrics handling, WSL2 browser
access, dataset management, pinning, eval modal on resume, UMAP geometry toggle.

---

### Eval harness: LLM-as-oracle ✅

Implementato il sistema di valutazione automatica che sostituisce l'oracolo
umano con un LLM per eseguire sessioni complete in modo riproducibile.

- **`src/eval/llm_oracle.py`** — oracle LLM che legge la vista oracolo e
  produce feedback testuale per turno; gestione retry su parse failure,
  cost tracking provider-aware, resilienza a errori 4xx
- **`src/eval/persona.py`** — definizione e caricamento delle personas da file
  JSON (`personas/`); ogni persona definisce stile di interazione, obiettivi,
  tolleranza alla frustrazione
- **`src/eval/oracle_view.py`** — costruisce la vista che l'oracle LLM vede
  a ogni turno (cluster attuali, point IDs, history); aggiornato per includere
  i point ID e insegnare all'oracle il concetto di pinning
- **`src/eval/eval_report.py`** — aggregazione metriche e report di sessione
- **`scripts/run_persona_eval.py`** — esegue una batteria di sessioni con
  personas diverse su un dataset, raccoglie metriche A1–B4
- **`scripts/run_scenario_eval.py`** — esegue scenari predefiniti (`scenarios/`)
- **`scripts/run_baseline_eval.py`** — baseline senza dialogo per confronto;
  bootstrap confidence intervals per ogni metrica

#### Personas implementate (`personas/`)

23 personas che coprono stili di interazione diversi: da `satisfied_minimalist`
(converge velocemente) a `never_satisfied_max_turns` (stress test), passando per
`bilingual_drifter`, `contradictory_oracle`, `prompt_injector_redteam` e altri.

#### Scenari predefiniti (`scenarios/`)

7 scenari con dataset e sequenze di feedback prefissate per test riproducibili:
`topic_merge`, `topic_split`, `sentiment_split`, `contradictory_oracle`,
`stable_oracle`, `high_load_oracle`.

---

### Ricerca bibliografica e contestualizzazione ✅

**Nota:** il lavoro bibliografico è stato condotto in parallelo durante tutti gli
sprint, a partire dall'analisi iniziale dei paper di riferimento negli sprint 2–3
che ha motivato le scelte architetturali (in particolare il semantic reembedding
e le metriche di valutazione). Questo sprint raccoglie e formalizza quel lavoro
in documenti condivisi nella repo.

**Paper analizzati:**
- Schild et al. (2021) — Interactive clustering con MUST-LINK/CANNOT-LINK;
  precursore pre-LLM diretto del paradigma conversazionale di BlaBlaClust
- Zhang et al., EMNLP 2023 — ClusterLLM; embedding fine-tuning via triplet LLM
- Hong et al., EMNLP 2025 — Dial-In LLM; LLM-ITL per customer service
- Fischer & Biemann, arXiv 2026 — Perspectives; HITL con SetFit+LoRA fine-tuning

**Documenti prodotti:**
- **`docs/related-work.md`** — contestualizzazione accademica di BlaBlaClust nei
  confronti dei 5 paper; tabelle comparative su paradigma oracle, embedding,
  valutazione; contributi originali e future directions
- **`docs/related-work-radar.ipynb`** — radar chart comparativo su 6 dimensioni
  di design (Automation, Interactivity, Embedding adaptability, Domain generality,
  Oracle accessibility, Evaluation rigor) per tutti e 5 i sistemi

---

### Esperimento: confronto modelli embedding ✅

**`docs/embedding-model-comparison.md`** — confronto quantitativo MiniLM vs
`BAAI/bge-base-en-v1.5` su 20newsgroups (300 doc, 6 classi):

| Metrica | MiniLM | BGE-base | Δ |
|---|---|---|---|
| Silhouette | 0.0463 | 0.0702 | +52% |
| NMI | 0.644 | 0.771 | +20% |
| ARI | 0.607 | 0.752 | +24% |
| Tempo embedding (CPU) | 28s | 242s | ×8.5 |

**Decisione:** miglioramento significativo ma costo computazionale 8× su CPU
non accettabile per un demo interattivo. Upgrade implementato sul branch
`feature/upgrade-embedding-bge` e non mergato. Script di benchmark:
`scripts/compare_embeddings.py`.

---

### Piano test: instruct-tuned reembedding ✅

**`docs/instruct-reembed-comparison.md`** — test plan per confrontare la
strategia attuale (MiniLM cosine hybrid `(N, D+1)`) con
`intfloat/multilingual-e5-large-instruct` su asse sentiment (IMDB, 300 doc).

Test non eseguito: il modello (560M parametri) richiede GPU — su CPU supera i
15 minuti per 300 documenti. Il piano è documentato e pronto per l'esecuzione
in ambiente GPU.

---

## Problemi aperti

- **MiniLM varianza coseno sempre sotto soglia su assi tonali**: il fallback LLM
  è sempre attivo per "angry tone" e assi simili; risolto parzialmente con
  N=600 ma il limite architetturale di MiniLM rimane — vedi
  `docs/embedding-model-comparison.md` e `docs/instruct-reembed-comparison.md`
  per le direzioni future
- **Eval harness non ancora eseguito su dataset completo**: le personas e gli
  scenari sono pronti ma una batteria sistematica richiede budget API non
  disponibile in fase di sviluppo
