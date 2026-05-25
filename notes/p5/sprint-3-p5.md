# Sprint 3 — P5 (Arianna)

## Consegnato questo sprint

### Issue #29 — `log_clustering_run` spostato in `src/logger.py` ✅
- Eliminato `src/engine/clustering_log.py` (modulo separato, P2-scope)
- Aggiunta `log_clustering_run()` direttamente in `src/logger.py`, con pattern
  identico a `log_llm_call`: scrive su `_clustering_log_path` (variabile di
  modulo patchabile nei test), swallows `OSError`
- Aggiornato import in `src/engine/initial_clustering.py`
- Aggiornati `tests/conftest.py` e `tests/test_clustering_log.py` per
  monkeypatchare `logger._clustering_log_path` invece del vecchio modulo
- Tutti e 7 i test passano (commit `ff820ff`, 2026-05-24)

### Issue #19 — UI enhancements ✅
Cinque miglioramenti su `ui/index.html` (commit `1045e86`, 2026-05-25):

1. **Responsive layout** — `height: 100dvh` su tutti i container (fix per
   mobile browser che sottraggono spazio con la barra indirizzi); breakpoint
   `@media (max-width: 580px)` che porta sidebar in orizzontale e cluster
   grid a colonna singola

2. **Session resume** — la welcome screen carica `GET /sessions` al page load,
   filtra le sessioni non chiuse e mostra una lista "Resume a session" con
   dataset, ID abbreviato, badge di stato e pulsante Resume →; `resumeSession()`
   recupera cluster attivi e turn number in parallelo senza ricominciare

3. **Multi-cluster selection** — `state.selectedClusterIds` cambiato da valore
   singolo a `Set`; cliccando più card si aggiungono/rimuovono; `sendMessage()`
   passa tutti gli ID come `target_cluster_ids`

4. **Cluster chips** — sopra il campo di testo ogni cluster selezionato appare
   come chip `📌 Nome ×` con deselect individuale; chips nascosti automaticamente
   quando nessun cluster è selezionato

5. **Expand modal: ricerca + paginazione** — campo search con filtro live,
   anteprima a 5 elementi, bottone "Show all N items"; titolo modale aggiornato
   con conteggio items

### Redesign palette ✅
- Rimosso il tema crema/marrone monomatico (scarso contrasto, feedback prof)
- Nuova palette editoriale "carta + salvia" (commit `9269ecf`, 2026-05-25):
  - background `#f5f4ef` (carta), testo `#1e1e1a` (inchiostro), accento
    `#4a7c59` (verde salvia profondo)
  - Cluster card con accenti botanici distinti: salvia, oliva, teal, ambra, viola
  - Zero colori hardcoded marroni rimasti nel CSS

### Issue #35 — Bug report: merge con cluster_id singolo ✅ (aperto per P3)
- Tracciato il bug `"Engine produced an invalid operation: merge_clusters needs
  at least 2 distinct clusters"`: il LLM restituisce un merge con 1 solo
  cluster_id quando l'oracle dice "elimina questo cluster"
- Causa: `prompts/f_output.txt` non specifica che merge richiede ≥2 id né
  cosa fare per operazioni di "delete" (che non esistono nel sistema)
- Fix proposto: aggiungere constraint esplicito in `CONSTRAINTS` di
  `f_output.txt` (issue #35 assegnata a chi gestisce i prompt)

## Debug investigation

Investigato il `KeyError: 'target_point_ids'` riportato dalla UI in produzione.
Il server live (avviato senza `--reload`) girava con codice stale; il debug
server sulla porta 8001 con il codice corrente non riproduceva l'errore e
mostrava invece il bug del merge (issue #35). Conclusione: l'errore `target_point_ids`
era un artefatto del codice vecchio in memoria, non un bug del codice attuale.

## Problemi aperti

- **Bug prompt LLM (issue #35)**: il fix richiede modifica a `prompts/f_output.txt`,
  file di competenza di un altro membro del team
- **Nomi cluster generici**: le sessioni nuove producono "Cluster 1, 2, 3, 4"
  perché la chiamata LLM di naming (`generate_names: true`) fallisce silenziosamente
  — da investigare sprint 4
- **Server restart obbligatorio**: il server va riavviato manualmente a ogni
  modifica (no `--reload`); considerare di abilitarlo in sviluppo

## Piano sprint 4 (P5)

- [ ] Harness LLM-as-oracle: script che fa girare una sessione completa con un
      oracle LLM (DeepSeek via OpenRouter) invece di un umano
- [ ] Metriche: turns-to-convergence, costo token/sessione, oracle satisfaction
- [ ] Notebook Jupyter con grafici finali
- [ ] Investigare il fallback silenzioso di `generate_names: true`
