# Sprint 4 — P4 (LLM & Prompting)

## File status

| File | Status |
|---|---|
| `src/engine/f_update_preferences.py` | ✅ creato — distillazione LLM delle preferenze oracle (issue #64) |
| `prompts/f_update_preferences.txt` | ✅ creato — prompt per preference extraction |
| `prompts/f_output.txt` | ✅ aggiornato — blocco `{oracle_preference_summary}` aggiunto |
| `src/harness.py` | ✅ refactored — solo interfaccia condivisa, zero import anthropic |
| `src/harness_claude.py` | ✅ creato — modulo Claude dedicato, simmetrico a openai/openrouter |
| `scenarios/topic_merge_20ng.json` | ✅ creato — scenario eval per 20 Newsgroups |
| `scenarios/topic_split_20ng.json` | ✅ creato |
| `scenarios/contradictory_topic_20ng.json` | ✅ creato |
| `scripts/serve_ui.py` | ✅ fix FK UUID Dataset |

---

## Cosa è stato fatto

### 1. Oracle preference summary (`e186fae`, issue #64)

**Contesto.** `f_output` vedeva la storia grezza dei turni oracle (testo libero
concatenato) ma non aveva una sintesi strutturata di *cosa l'oracle vuole* — ogni
turno era trattato in isolamento. Su sessioni lunghe (>5 turni) il modello perdeva
il filo delle preferenze espresse nei turni iniziali.

**Soluzione.** Dopo ogni turno, `f_update_preferences()` chiama l'LLM per distillare
dalla storia completa un profilo oracle in 3–5 bullet. Il profilo viene salvato su
`ChatSession.preference_summary` (nuova colonna, migrата automaticamente in
`_run_migrations()`) e iniettato nel prompt di `f_output` sotto un blocco
`ORACLE PREFERENCE PROFILE`.

```
ORACLE PREFERENCE PROFILE (derived from full feedback history):
{oracle_preference_summary}
```

La chiamata è **best-effort**: un fallimento non blocca il turno, lascia il profilo
invariato.

**File toccati** (9): `f_update_preferences.py` (nuovo, 76 righe),
`f_update_preferences.txt` (nuovo, 23 righe), `f_output.txt`, `f_output.py`,
`schemas.py`, `models.py`, `main.py`, `session_state.py`, `turns.py`.

**Effetto misurabile.** In sessioni con oracle coerente, `f_output` cita esplicitamente
le preferenze già espresse invece di re-interpretare ogni turno da zero. In scenari
con oracle contraddittorio, il profilo accumula le contraddizioni — input diretto per
il giudice B4.

---

### 2. Refactor harness — 3-provider split pulito (`c7a30b7`, issue #54)

**Problema.** `harness.py` conteneva codice Claude-specifico (import anthropic,
`_PRICING` Claude, `count_tokens` via `client.messages.count_tokens`) mescolato
all'interfaccia condivisa. Aggiungere un provider richiedeva modifiche al file
condiviso — fragile.

**Soluzione.** `harness.py` ora contiene solo:
- prompt loading, `hash_prompt`, `render_prompt`
- `ConversationContext`
- `_retry_sync` / `_retry_async` (generico, accetta `is_transient` callable)
- `call_llm` dispatcher (lazy import per provider)
- `estimate_cost_usd` (delega al modulo provider)

`harness_claude.py` (nuovo, 210 righe) è il terzo modulo simmetrico a
`harness_openai.py` e `harness_openrouter.py`:
- `DEFAULT_MODEL`, `_PRICING`, `_is_transient_error` (Anthropic-specifici)
- `call_claude`, `call_claude_async`, `call_claude_batch`
- `count_tokens` (endpoint `messages.count_tokens` di Anthropic)
- `extract_usage`, `estimate_cost_usd_claude`

Pricing Gemini spostato da `harness.py` a `harness_openrouter.py`.
`f_cognitive_load.py` e `eval_cache.py` puntati a `harness_claude`.

**Acceptance criteria verificati:**
- `harness.py` non contiene `import anthropic`
- `LLM_PROVIDER` unset usa ancora Claude (nessun cambio di comportamento)
- 295 test passano dopo il refactor

---

### 3. Eval scenarios per 20 Newsgroups (`99658ca`, issue #50)

3 nuovi scenari in `scenarios/` che specchiano quelli Amazon per confronto
cross-dataset:

| Scenario | k | Operazione | Terminazione attesa |
|---|---|---|---|
| `topic_merge_20ng` | 7→5 | oracle merge cluster science + recreation | `converged` |
| `topic_split_20ng` | 3→5 | oracle split cluster broad-topic | `converged` |
| `contradictory_topic_20ng` | 4→5 | oracle reversa uno split due volte | `converged` |

**Risultati verificati** (google/gemini-2.5-flash-lite via OpenRouter):

| Scenario | B1 | B4 | Note |
|---|---|---|---|
| `topic_merge_20ng` | 0.85 | 0.0 | oracle chiaro, alta qualità |
| `topic_split_20ng` | 0.85 | 0.0 | idem |
| `contradictory_topic_20ng` | 0.75 | 0.6 | contraddizione rilevata correttamente |

B1 comparabile agli scenari Amazon (0.75–0.85) — il pipeline non è tuned su Amazon.

---

### 4. Fix FK UUID Dataset al seeding (`88e53cd`)

`serve_ui.py` passava `DATASET_NAME` (stringa `'amazon_reviews'`) direttamente
come `dataset_id` a `ingest_csv_path` dopo che `models.py` aveva introdotto una
tabella `Dataset` con UUID come PK. Fix: chiama `get_or_create_dataset` prima
dell'ingestione per ottenere l'UUID reale, poi lo passa a `ingest_csv_path` e
`generate_embeddings_for_dataset`. Server ora si avvia senza crash su DB freschi.

---

## Decisioni architetturali

- `f_update_preferences` è **best-effort by design**: se l'LLM fallisce, il turno
  va avanti con il profilo precedente. Non è corretto bloccare il clustering per un
  fallimento di una chiamata ausiliaria.
- Il refactor harness non cambia l'API pubblica — tutti i call-site esistenti
  (`from src.harness import call_llm, render_prompt, ...`) funzionano invariati.
- Gli scenari 20NG usano lo stesso formato JSON degli scenari Amazon —
  `run_scenario_eval.py` non richiede modifiche per supportarli.

---

## Sprint 4 plan (pianificato vs consegnato)

| Task (da sprint-3-p4 plan) | Stato |
|---|---|
| `prompts/oracle_sim.txt` — oracle LLM per persona eval | ➡️ fatto da P5 (`llm_oracle.py`) |
| Valutare `detect_contradiction` (keyword matching) | ✅ lasciato com'è — B4 copre il caso |
| Timeout configurabile su chiamate OpenRouter | ⬜ non fatto |
| Oracle preference summary (issue #64) | ✅ consegnato |
| 3-provider split pulito (issue #54) | ✅ consegnato |
| Scenari eval 20NG (issue #50) | ✅ consegnato |

---

## Cosa mi ha bloccato

- Nessun blocco critico questo sprint.
- La migrazione automatica della colonna `preference_summary` in `_run_migrations()`
  ha richiesto un allineamento con P1 (che gestisce `backend/main.py`) — risolto
  con commit diretto previa comunicazione.

## Cosa faccio nel prossimo sprint

- Nessuno sprint successivo pianificato (fine progetto).
- Pending: timeout configurabile su OpenRouter (issue aperto, bassa priorità).
- Report e presentazione finale.
