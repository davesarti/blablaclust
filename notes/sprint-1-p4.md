# Sprint 1 — P4 (LLM & Prompting)

## Ruolo e responsabilità

P4 è il layer LLM del sistema. Nessun altro ruolo importa `anthropic` direttamente — tutto passa da `src/harness.py`.

Componenti del grading di competenza P4:
- **Component 2** — Agentic Harness (`src/harness.py`)
- **Component 7** — Prompts versionati (`prompts/`)
- **Component 8** — Resilience & Error Handling
- **Component 9** — Cost & Rate-Limit (con P5)

---

## File di competenza P4

| File | Stato |
|---|---|
| `src/harness.py` | ✅ creato (Week 1) |
| `prompts/f_output.txt` | ✅ creato (Week 1) |
| `prompts/f_next_state.txt` | ⬜ Week 2 |
| `prompts/f_next_best_step.txt` | ⬜ Week 3 |
| `prompts/f_uncertainty.txt` | ⬜ Week 3 |
| `prompts/f_eval.txt` | ⬜ Week 4 |

---

## Struttura `src/harness.py`

### Prompt loading (Component 7)

```python
load_prompt(name: str) -> str
    # legge prompts/{name}.txt come stringa grezza

hash_prompt(name: str) -> str
    # SHA-256 del file (prime 16 cifre) — va loggato in ogni run JSONL

render_prompt(name: str, **kwargs) -> str
    # chiama str.format(**kwargs) sul template — MAI usare f-string nel codice
```

### Chiamate API (Component 2)

```python
call_claude(messages, system, model, max_tokens) -> Message
    # sincrona, retry automatico su errori transitori, dry-run mode

call_claude_async(messages, system, model, max_tokens) -> Message
    # async, stessa logica

call_claude_batch(requests: list[dict], model) -> list[Message | Exception]
    # N chiamate in parallelo via asyncio.gather
```

### Retry (Component 8)

```python
_is_transient_error(exc) -> bool
    # True per RateLimitError, APIConnectionError, status 429/5xx
    # False per 4xx permanenti (400, 401, 403, 404) — non si riprova

_retry_sync(fn, max_retries, base_delay, max_delay) -> Any
    # exponential backoff: delay = min(base * 2^attempt + jitter, max_delay)

_retry_async(fn, ...) -> Any
    # identica ma con await asyncio.sleep
```

### Token counting & costi (Component 9)

```python
count_tokens(messages, system, model) -> int
    # usa client.messages.count_tokens() — zero costo, non chiama messages.create

extract_usage(message) -> dict
    # {input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}

estimate_cost_usd(usage, model) -> float
    # pricing hardcoded per modello, restituisce stima in USD
```

### Stato conversazionale

```python
class ConversationContext:
    session_id: str
    turns: list[dict]          # {role, content} per l'API

    add_oracle_turn(oracle_turn: dict)   # appende turno utente
    add_system_turn(system_turn: dict)   # appende risposta sistema
    build_messages(model) -> list[dict]  # windowing automatico se > MAX_INPUT_TOKENS
    get_cognitive_load_score() -> float  # n_turni / 20
    detect_contradiction(new_turn, clusters) -> ContradictionRecord | None
        # Week 1: stub → None
        # Week 3: logica reale (split↔merge, approve↔reject su stesso cluster)

class ContradictionRecord:
    turn_number, description, earlier_turn,
    feedback_type_a, feedback_type_b, target_cluster_id
```

---

## Variabili d'ambiente (`.env`)

```bash
ANTHROPIC_API_KEY=...          # obbligatoria
ANTHROPIC_MODEL=claude-sonnet-4-6
HARNESS_DRY_RUN=false          # true = nessuna chiamata API reale
HARNESS_MAX_RETRIES=4
HARNESS_BASE_DELAY=1.0
HARNESS_MAX_DELAY=60.0
MAX_INPUT_TOKENS_PER_TURN=8000
```

---

## Piano settimanale

### Week 1 — Harness base + f_output ✅

1. `requirements.txt`: aggiunto `anthropic>=0.40.0`, `python-dotenv>=1.0.0`
2. `.env.example`: aggiornato con variabili Anthropic
3. `prompts/f_output.txt`: prompt per generare/aggiornare cluster con nomi e descrizioni
4. `src/harness.py`: tutte le funzioni sopra, dry-run mode, ConversationContext stub

**Smoke test (DRY_RUN):**
```bash
HARNESS_DRY_RUN=true python3 -c "
from src.harness import render_prompt, call_claude, hash_prompt
prompt = render_prompt('f_output', session_id='s1', turn_number=1,
    total_points=50, clusters_json='[]', feedback_type='global',
    oracle_raw_text='3 clusters please', target_cluster_id='', history_summary='')
msg = call_claude([{'role': 'user', 'content': prompt}], system='')
print(hash_prompt('f_output'), msg.content[0].text)
"
```

### Week 2 — f_next_state + feedback history

- Scrivere `prompts/f_next_state.txt`
- Popolare `ConversationContext` con storia reale dal DB (richiede `crud.get_session_turns` da P1)
- Test su 5 turni reali con chiamata vera (non dry-run)

`f_next_state.txt` variabili: `{current_state_json}`, `{oracle_raw_text}`, `{feedback_type}`, `{target_cluster_id}`, `{feedback_history_json}`

### Week 3 — Contradiction tracking + f_next_best_step

- Implementare `detect_contradiction` in `ConversationContext`
  - Cerca pattern opposti sullo stesso `target_cluster_id`: split↔merge, approve↔reject
  - Popola `ContradictionRecord` e aggiunge a `self.contradictions`
- Scrivere `prompts/f_next_best_step.txt`
- Aggiungere logging LLM calls (coordinare interfaccia con P5)

`f_next_best_step.txt` variabili: `{current_state_json}`, `{last_oracle_input}`, `{uncertainty_scores_json}`, `{turn_number}`, `{cognitive_load}`

### Week 4 — Ottimizzazioni + f_eval

- Ottimizzare token (prompt caching dove possibile)
- `prompts/f_uncertainty.txt` e `prompts/f_eval.txt`
- Gestione errori API robusta (log strutturato dei fallimenti)

---

## Interfacce con gli altri ruoli

### P3 usa harness così:
```python
from src.harness import render_prompt, call_claude, hash_prompt

prompt = render_prompt("f_output", **kwargs)
msg = call_claude([{"role": "user", "content": prompt}], system="")
result = json.loads(msg.content[0].text)   # parse JSON in P3, non in harness
```
**P3 non importa `anthropic` direttamente.**

### P5 (logger):
P5 esporrà `log_llm_call(session_id, prompt_name, prompt_hash, usage, cost_usd)`.
In attesa, usare uno stub locale. Coordinare l'interfaccia entro fine Week 1.

### P1 (crud):
In Week 2 servirà `crud.get_session_turns(session_id, db) -> list[Turn]` per ricostruire
la storia conversazionale da DB in `ConversationContext`.

---

## Regola architetturale critica

`harness.py` **non fa mai il parse del JSON** nella risposta del modello.
Restituisce il `Message` grezzo. Il parse è responsabilità di P3 (`engine/`).
Se il modello risponde con JSON malformato, l'errore emerge nel layer corretto.

---

## Cosa mi ha bloccato
- (da compilare a fine sprint)

## Cosa faccio nel prossimo sprint
- Scrivere `prompts/f_next_state.txt`
- Integrare `ConversationContext` con DB (aspetta `crud.get_session_turns` da P1)
- Test su turni reali con API vera
