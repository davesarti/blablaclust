# Sprint 2 — P5 (Arianna)

## Consegnato questo sprint

- `src/logger.py`: logging JSONL + `deviation()` ✅
- `scripts/cli.py`: CLI interattiva (create/resume sessione, invia turni via
  `/sessions/{id}/turns`, gestisce action="stop") ✅
- `ui/index.html`: interfaccia web base + `ui/TODO.md` con gap da colmare ✅

## Problemi aperti / osservazioni

### Il loop non è ancora chiuso
Al momento il feedback dell'oracle non ri-clusterizza il DB e `f_next_best_step`
non è sul path API. La CLI restituisce 404 sull'endpoint di turn (non ancora
wired). Priorità sprint 3: chiudere questo loop.

### Operazioni strutturali vs. intent semantici
Merge / split / move / rename operano sulla topologia dei cluster, non sul
contenuto semantico. L'oracle può dire "fai un cluster con le recensioni più
arrabbiate" e il sistema non ha strumenti per eseguirlo.

Ho scritto una proposta architetturale dettagliata in
`docs/semantic-reembed-proposal.md`. Riassunto: al Turn 1 estrarre l'asse
semantico dell'oracle → re-embed tutti i punti in uno spazio ibrido
(originale + asse) → ricalcolare k-means. I turni successivi operano su una
geometria già orientata semanticamente.

## Piano sprint 3 (P5)

- [ ] `scripts/smoke_test.sh` — obbligatorio dal corso, ancora da fare
- [ ] Harness LLM-as-oracle: script che fa girare una sessione completa con un
      oracle LLM (DeepSeek via OpenRouter) invece di un umano — produce log JSONL
      per l'evaluation
- [ ] Metriche: turns-to-convergence, oracle_satisfaction_score,
      target_alignment_score, costo in token per sessione
- [ ] Notebook Jupyter con grafici finali (settimana 3)
- [ ] UI: alta priorità dal TODO.md — distinguere ask/show/stop, mostrare
      contradiction_detected

## Decisioni da portare in riunione di gruppo

1. Chi implementa `f_semantic_reembed`? (P2 per gli embedding, P3 per la logica
   engine, P4 per il prompt design — vedi `docs/semantic-reembed-proposal.md`)
2. `axis_hint` estratto automaticamente dall'LLM al Turn 1 o input manuale dalla UI?
3. `oracle_satisfaction_score` (1-5) da aggiungere all'`InputOracle` — chi fa
   lo schema Pydantic (P1) e chi aggiunge il campo alla UI (P5)?
