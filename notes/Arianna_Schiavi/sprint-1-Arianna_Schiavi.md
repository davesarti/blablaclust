# Sprint 1 — P5 (Arianna)

## Ruolo e responsabilità

P5 copre tre aree: **evaluation & metrics** (misurare la qualità del clustering
nel tempo), **logging strutturato** (ogni chiamata LLM tracciata su file JSONL),
e **UI / CLI** (interfaccia per l'oracle umano). Componenti di grading P5:
- Component 9 — Cost & Rate-Limit (condiviso con P4)
- Component 10 — Evaluation harness
- Component 11 — UI / oracle interface

---

## Consegnato questo sprint

### Logger strutturato — `src/logger.py` ✅
- `log_llm_call(session_id, prompt_name, prompt_hash, usage, cost_usd)` — scrive
  una riga JSONL su `logs/llm_calls.jsonl` con timestamp ISO, tutti i campi
  usage (input/output/cache tokens) e costo stimato in USD; swallows `OSError`
  per non bloccare il flusso principale se il file non è scrivibile
- `deviation(a, b)` — metrica di distanza normalizzata tra due distribuzioni
  di probabilità (usata in evaluation per misurare quanto il clustering è cambiato
  tra turni successivi)
- Directory `logs/` creata automaticamente al primo import

### CLI interattiva — `scripts/cli.py` ✅
Interfaccia a riga di comando per l'oracle umano:
- Crea o riprende una sessione via `POST /sessions` / `GET /sessions`
- Invia turni tramite `POST /sessions/{id}/turns` con feedback testuale
- Gestisce `action="stop"` restituito dal sistema per terminare la sessione
- Nota: versione temporanea — al momento del delivery il backend aveva un bug
  nel routing degli endpoint turns che causava 404; risolto nel sprint 2

## Interfaccia con gli altri ruoli

P4 usa `log_llm_call` da `src/logger.py` dopo ogni chiamata LLM in `f_output.py`.
Interfaccia concordata con P4 entro fine sprint 1:
```python
log_llm_call(
    session_id: str,
    prompt_name: str,   # es. "f_output"
    prompt_hash: str,   # SHA-256 del file prompt (16 cifre)
    usage: dict,        # {input_tokens, output_tokens, ...}
    cost_usd: float,
)
```

## Problemi aperti

- **CLI → 404 su turns**: `POST /sessions/{id}/turns` non era ancora wired nel
  router al momento del delivery; la CLI è funzionante ma il backend non risponde
  correttamente — priorità sprint 2
- **Evaluation non ancora collegata**: `deviation()` è implementata ma non c'è
  ancora un harness che la chiami a ogni turno; serve aspettare che P3 stabilizzi
  il loop principale

## Piano sprint 2 (P5)

- [ ] UI web: `ui/index.html` con chiamate API reali — sostituisce la CLI come
      interfaccia principale per l'oracle umano
- [ ] Integrare il logger con il server una volta che il routing turns è fixato
- [ ] `scripts/serve_ui.py`: script per servire la UI con uvicorn
