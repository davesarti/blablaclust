# Sprint 2 — P5 (Arianna)

## Consegnato questo sprint

### UI web — `ui/index.html` ✅
Prima versione funzionante dell'interfaccia web BlaBlaClust:
- Layout a tre zone: header fisso, area cluster, drawer chat espandibile
- Chiamate API reali a tutti gli endpoint: `POST /sessions`, `POST /clusters/{id}`,
  `GET /clusters/active`, `POST /turns`, `GET /clusters/{id}/points`
- Rendering delle cluster card con nome, descrizione, contatore, barra proporzionale
- Drawer chat con messaggi utente/sistema, typing indicator, invio con Enter
- Export JSON/CSV dei cluster finali
- Servita via `scripts/serve_ui.py` su `http://localhost:8000/ui`
- Palette editoriale crema/marrone; `ui/TODO.md` con gap da colmare (successivo sprint)

### Logger aggiornato — `src/logger.py` ✅
Aggiornamenti minori al logger per allinearlo all'integrazione con il server:
- `serve_ui.py` aggiornato per servire la UI dalla root corretta
- Database demo aggiornato con dati seed per test manuali

## Problemi aperti / osservazioni

### Il loop non era ancora chiuso (risolto in sprint 3)
Al momento del delivery il feedback dell'oracle non ri-clusterizzava il DB
e `f_next_best_step` non era sul path API. La CLI restituiva 404. Chiuso in
sprint 3 con il fix dei routers da parte del team.

### Operazioni strutturali vs. intent semantici
Merge / split / move / rename operano sulla topologia dei cluster, non sul
contenuto semantico. L'oracle può dire "fai un cluster con le recensioni più
arrabbiate" e il sistema non ha strumenti per eseguirlo.

Proposta architetturale documentata in `docs/semantic-reembed-proposal.md`:
al Turn 1 estrarre l'asse semantico dell'oracle → re-embed tutti i punti in
uno spazio ibrido (originale + asse) → ricalcolare k-means. I turni successivi
operano su una geometria già orientata semanticamente.

## Decisioni da portare in riunione di gruppo

1. Chi implementa `f_semantic_reembed`? (P2 per gli embedding, P3 per la logica
   engine, P4 per il prompt design — vedi `docs/semantic-reembed-proposal.md`)
2. `axis_hint` estratto automaticamente dall'LLM al Turn 1 o input manuale dalla UI?
3. `oracle_satisfaction_score` (1-5) da aggiungere all'`InputOracle` — chi fa
   lo schema Pydantic (P1) e chi aggiunge il campo alla UI (P5)?
