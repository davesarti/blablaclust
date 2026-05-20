# Sprint 2 — P4 (LLM & Prompting)

## Stato file

| File | Stato |
|---|---|
| `src/harness.py` | ✅ aggiornato — `detect_contradiction` reale, provider openrouter |
| `prompts/f_output.txt` | ✅ aggiornato — operazioni strutturate (issue #13) |
| `prompts/f_next_best_step.txt` | ✅ creato (poi funzione riscritta rule-based da P3) |
| `prompts/f_eval.txt` | ✅ creato |
| `src/engine/f_output.py` | ✅ ritorna `(raw, usage)` tuple, logging, fix `add_system_turn` |
| `src/engine/f_eval.py` | ✅ implementato con LLM call |
| `src/schemas.py` | ✅ `SystemTurn` espone `token_usage` e `cost_usd` |

---

## Cosa è stato fatto

### Fix e refactoring
- **`add_system_turn` inconsistency**: ora passa sempre `raw` dict — coerente tra live e replay dal DB
- **`f_output` ritorna tuple**: `(raw, msg.usage)` invece di solo `raw` (issue #12)
- **`_DRY_RUN_OUTPUT`**: aggiornato al nuovo schema con `operations` invece di `clusters_updated`

### Nuove funzionalità
- **Logging LLM calls**: `log_llm_call` chiamato in `f_output` e `f_eval` dopo ogni risposta
- **`detect_contradiction`**: keyword matching split↔merge su stesso `target_cluster_id` — provider-agnostic
- **`f_eval`**: valutazione qualità clustering (coherence_score, coverage_score, suggested_merges, notes)
- **Issue #13**: `f_output.txt` riscritto — Claude ritorna operazioni (merge/split/rename), non cluster list fittizia

### Decisioni architetturali
- Provider attivo: OpenRouter con DeepSeek free (`LLM_PROVIDER=openrouter`)
- `prompts/f_uncertainty.txt` non necessario — `f_uncertainty` è puro Python/DB
- `prompts/f_next_state.txt` non necessario — `f_next_state` è puro Python

---

## Cosa faccio nel prossimo sprint
- (da compilare)
