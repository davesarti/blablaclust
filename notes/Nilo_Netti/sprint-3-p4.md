# Sprint 3 — P4 (LLM & Prompting)

## File status

| File | Status |
|---|---|
| `prompts/f_output.txt` | ✅ updated — 3 fixes (issue #26, #28, #35) |
| `src/schemas.py` | ✅ updated — `token_usage` and `cost_usd` on `SystemTurn` (issue #12) |
| `src/engine/f_uncertainty.py` | ✅ rewritten — cluster-level uncertainty (issue #45) |
| `src/engine/f_next_best_step.py` | ✅ updated — cluster-level Rule 2, stop threshold fix (issue #44, #45) |
| `src/harness.py` | ✅ updated — cognitive load formula fix (issue #44); lazy anthropic imports |
| `backend/routers/turns.py` | ✅ updated — LLM reply always used, session.status closed on stop (issue #44); JSON display guard |
| `tests/test_f_next_best_step.py` | ✅ updated — tests for new ClusterUncertainty API |

---

## What was done

### Prompt fixes — `f_output.txt`

- **Issue #26** (`64fd820`): instructed the model to use cluster names and descriptions as signals when proposing merge/split. Merge only when two clusters are redundant across ALL meaningful dimensions (topic + sentiment/tone), not just shared topic.

- **Issue #35** (`9e9b907`): added explicit constraint that a merge operation always requires ≥2 `cluster_ids`. When the oracle asks to "eliminate" a cluster, there is no delete operation — the model must merge it with the most similar remaining cluster.

### Schema fix — `src/schemas.py`

- **Issue #12** (`f5b117d`): added `token_usage: Optional[Dict[str, int]]` and `cost_usd: Optional[float]` to `SystemTurn`, needed by P5 to display real token counts and cost in the UI sidebar.

### Cluster-level uncertainty (`issue #45`, `18b9eed`)

The old Rule 2 asked the oracle to classify individual data points with opaque UUIDs — unusable at 1200 points. Removed and redesigned at the structural level:

**New data structures in `f_uncertainty.py`:**
- `ClusterOverlap` — cluster pair with `overlap_fraction` = fraction of points ambiguous between both clusters
- `ClusterCohesion` — cluster with low `mean_max_prob` (diffuse, split candidate)
- `ClusterUncertainty` — container for both signals
- `f_cluster_uncertainty()` — computes both from soft-assignment DB rows

**Rule 2 reinstated in `f_next_best_step.py`:**
- Rule 2a: significant overlap → ask about merge, using cluster names (not UUIDs)
- Rule 2b: low cohesion → ask about split, using cluster name
- One structural question at a time, most urgent first

**Before:** *"Review X (uncertainty 0.51). How should this be classified?"*
**After:** *"Clusters 'Battery' and 'Battery Life' overlap: 18% of data points are ambiguous between them. Are they meaningfully distinct, or should they be merged?"*

---

## Architectural decisions

- `BoundaryPoint` / `f_uncertainty()` kept as legacy API for backwards compatibility with existing tests — new code uses `f_cluster_uncertainty()`.

---

### Lazy anthropic imports (`491376d`)

`harness.py` importava `anthropic` al top-level, il che causava `ImportError` se il
pacchetto non era installato anche solo importando il modulo in un contesto senza
API key. Spostato a import lazy dentro `call_claude`/`call_claude_async` — il
modulo è ora importabile anche senza `anthropic` nell'environment.

### Rimozione ask per-punto su uncertainty (`d042219`)

Rimosso il comportamento precedente di Rule 2 che chiedeva all'oracle di classificare
singoli data point via UUID. Con 1200 punti, presentare UUID è inutilizzabile. Questo
commit è il precursore necessario alla riscrittura cluster-level (`18b9eed`).

### JSON display guard in turns.py (`50a0335`, `2026-05-28`)

Modelli small/free (es. `openai/gpt-oss-20b`) mettono a volte JSON strutturato
nel campo `display` invece di prosa leggibile — il JSON grezzo appariva nella chat
dell'oracle. Fix: se `raw_display` inizia con `{` o `[` oppure fa il parse come JSON
valido, si usa il messaggio di `f_next_best_step` (sempre in inglese leggibile) come
fallback. Aggiunge 17 righe a `backend/routers/turns.py`.

---

## Open issues

- **Issue #46** (filed for P3): `test_rename_defaults_new_description_to_empty_string` fails after P3's sprint-3 changes to `f_apply_operations.py` — the DB mock needs to be updated to reflect the new behavior (preserving existing cluster description).

## Interface with other roles

- **P5** implemented `scripts/run_eval.py` (automated evaluation harness) and 5 JSON scenarios in `scenarios/`. Sprint 4 P4 plan coordinates with this harness for the `oracle_sim` prompt.
- **P5** added `src/engine/f_validate_point.py` + `prompts/f_validate_point.txt` — per-point cluster validation.
- **P3** introduced a test regression in `test_rename_defaults_new_description_to_empty_string` (issue #46 filed).

## Sprint 4 plan (P4)

- [ ] `prompts/oracle_sim.txt` — LLM-as-oracle prompt to use with `scripts/run_eval.py` (coordinated with P5)
- [ ] Evaluate whether `detect_contradiction` needs strengthening (currently keyword matching on split↔merge)
- [ ] Add configurable timeout to OpenRouter calls (currently no timeout → hangs if the provider is slow)
