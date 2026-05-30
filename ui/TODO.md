# UI TODO — BlaBlaClust

Gap tra quello che il backend produce e quello che la UI mostra attualmente.

---

## Alta priorità

### 1. Distinguere `action: ask / show / stop`
`f_next_best_step` restituisce tre azioni diverse ma la UI le tratta tutte come un messaggio chat normale.

- `show` → comportamento attuale (nessuna modifica)
- `ask` → header del messaggio diverso, colore/stile distinto (es. bordo blu), label "The system is asking:"
- `stop` → messaggio di chiusura prominente + disabilita input + segna sessione come converged automaticamente

Il campo è `turnData.system_output.action`.

---

## Media priorità

### 2. Boundary point dell'"ask" interattivi
Quando `action === "ask"`, `display.items` contiene fino a 5 punti ambigui:
```json
{ "point_id": "...", "text_preview": "...", "uncertainty_score": 0.47, "cluster_scores": {...} }
```
Ora vengono mostrati come bullet plain text nel `msg-diff`. Miglioramento: renderli come card cliccabili che pre-compilano l'input con il `point_id` nel campo `target_point_ids` (feedback type `point`).

Richiede anche di esporre il feedback type `point` nell'`InputOracle` inviato al backend (ora la UI manda solo `global` o `cluster`).

---

### 3. Diff visivo dei cluster tra turni
Quando `clusters_updated === true` la UI chiama `refreshClusters()` e aggiorna le card, ma non mostra *cosa* è cambiato (cluster rinominati, fusi, splittati, conteggi modificati).

Proposta: confrontare `state.clusters` prima e dopo il refresh e mostrare nel messaggio sistema un mini-diff testuale (es. "Cluster A: 120→95 items", "Cluster B merged into C").

---

### 4. Feedback type `instructional` nell'input
Il tipo `instructional` è già nel backend (`InputOracle.feedback_type`) ma non è selezionabile dalla UI. Potrebbe essere un toggle/select vicino alla textarea per gli utenti che vogliono dare istruzioni generali al sistema ("from now on, always keep electronics and accessories separate").

---

## Bassa priorità

### 5. Persistenza / resumption di sessione
Refresh della pagina → stato perso. Il DB mantiene sessioni e turni, quindi tecnicamente si potrebbe recuperare.

Opzione minima: al caricamento della pagina, chiamare `GET /sessions` e mostrare sessioni attive recenti con un bottone "Resume".

---

## Note tecniche

- Tutto in `ui/index.html` — nessuna build step, nessuna dipendenza esterna
- Prima di aggiungere funzionalità verificare che l'endpoint backend corrispondente esista e sia wired (vedi `notes/progress_report.md` — il loop non è ancora chiuso al 2026-05-20)
- I feedback type `point` e `instructional` esistono negli schema Pydantic ma non sono testati end-to-end
