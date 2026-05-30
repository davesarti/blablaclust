# Report: branch `feature/semantic-reembed`

**Autore:** P5 (Arianna Schiavi)
**Ultimo aggiornamento:** 2026-05-30 (rev 10)
**Stato:** implementazione completa, integrata su main

---

## Obiettivo

Permettere all'oracle di orientare il clustering lungo un asse semantico specifico
(es. "angry tone", "battery life") in qualsiasi momento della sessione, invece di
partire sempre da un k-means generico sul topic.

---

## Architettura della feature (stato attuale)

### Punto di ingresso: operazione `semantic_reembed` emessa dall'LLM

Nella versione attuale (dopo l'integrazione del commit `d0480c7` di Davide),
`axis_hint` **non è più un campo dello schema** `InputOracle`. Il re-embedding
è ora un'operazione come qualsiasi altra: l'oracle scrive in linguaggio libero e
`f_output` decide di emettere `{"type": "semantic_reembed", "axis_label": "..."}`.

Può avvenire **a qualsiasi turno**, non solo al primo.

### Flusso completo

```
POST /turns  (oracle input testo libero)
    │
    ├── clarify-confirmation short-circuit (deterministico)
    │   Se il turno precedente aveva un clarify_pending E l'oracle conferma
    │   con "yes"/"sì"/"ok"/ecc. → bypassa f_output, emette semantic_reembed
    │   direttamente con l'asse salvato
    │
    └── f_output (LLM) → classifica l'intent
            │
            ├── action:"clarify"  (frase nuda ambigua: "angry tone", "battery life")
            │   → salva {"type":"clarify_pending","axis_label":"..."} in state_snapshot
            │   → mostra domanda di conferma all'oracle
            │
            ├── action:"semantic_reembed"  (intent esplicito: "cluster by X")
            │   → _run_semantic_reembed(axis_label)
            │       → semantic_clustering → nuovi cluster + SoftAssignment
            │
            └── merge / split / rename / move / no_change / explain
                    → f_apply_operations → cluster_operations
```

### Rilevamento dell'asse semantico nei turni successivi

Una volta che una sessione ha fatto un semantic_reembed, l'asse viene recuperato
dalla storia dei turni invece di essere rieseguito:

```python
# turns.py
session_axis_hint = _active_axis_from_history(prior_turns)
# scorre i turni al contrario cercando l'ultimo op semantic_reembed
# accetta sia "axis_label" (nuovo) che "axis_hint" (retrocompatibilità sessioni vecchie)
```

Viene passato a `f_apply_operations → merge/split_cluster → name_clusters` così
i nomi dei cluster generati da operazioni strutturali restano coerenti con l'asse.

---

## File creati / modificati

### Nuovi file

| File | Scopo |
|------|-------|
| `src/engine/f_semantic_reembed.py` | Calcola la matrice ibrida `(N, D+1)` |
| `src/engine/semantic_clustering.py` | Orchestra re-embed + k-means + naming |
| `prompts/semantic_axis_poles.txt` | Prompt LLM per la generazione dei poli dell'asse |
| `tests/test_f_semantic_reembed.py` | 37 unit test per `f_semantic_reembed` |
| `tests/test_semantic_clustering.py` | 25 unit test per `semantic_clustering` |
| `tests/test_harness_json.py` | 7 unit test per `loads_llm_json` / `_escape_unescaped_quotes` |
| `tests/test_turns_semantic_reembed_op.py` | Test per il nuovo path semantic_reembed via f_output |

### File modificati

| File | Modifica |
|------|----------|
| `src/schemas.py` | `axis_hint` **rimosso** da `InputOracle` (rev 10) |
| `src/engine/cluster_naming.py` | Parametro `axis_hint`; blocco AXIS CONTEXT biforcato; usa `loads_llm_json` |
| `src/engine/cluster_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `src/engine/f_apply_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `src/engine/f_output.py` | Usa `loads_llm_json` invece di `json.loads(extract_json_text(...))` |
| `src/harness.py` | `_escape_unescaped_quotes` + `loads_llm_json`; fallback `{...}` su `extract_json_text` |
| `src/harness_openai.py` | Guard `content is None` in `call_gpt` e `call_gpt_async` |
| `src/harness_openrouter.py` | Guard `content is None` in `call_openrouter` e `call_openrouter_async` |
| `backend/routers/turns.py` | Refactor completo del path semantico (rev 10, vedi sotto) |
| `prompts/cluster_naming.txt` | Slot `{axis_context}` con regola biforcata; escape virgolette interne |
| `prompts/f_output.txt` | `semantic_reembed` come operazione LLM; azione `clarify` per frasi ambigue |
| `ui/index.html` | Placeholder adattivo al turno; `axis_hint` **rimosso** dal payload; `buildDiff` estratto |
| `tests/conftest.py` | `HARNESS_DRY_RUN=true` impostato prima di qualsiasi import (fix race condition) |
| `tests/test_f_apply_operations.py` | Fixture aggiornate con `axis_hint=None` |

---

## Dettaglio implementazione

### `f_semantic_reembed.py`

Due strategie + selettore ibrido:

```
_generate_axis_poles(axis_label)
  → (pole_pos_text, pole_neg_text)   # 1 chiamata LLM, sempre

_cosine_axis_scores(points, pole_pos_text, pole_neg_text)
  → cosine_var > COSINE_VARIANCE_THRESHOLD (0.01)?
      sì → usa i punteggi coseno direttamente
      no → _llm_axis_scores(points, axis_label)
              → llm_std < LLM_STD_THRESHOLD (1.0)?
                  sì → raise AxisNotDiscriminativeError
                  no → usa i punteggi LLM
```

**Generazione poli LLM (`_generate_axis_poles`)**

Chiede all'LLM due testi concreti come poli (non frasi astratte "very angry").
I testi concreti si embedderanno meglio perché vivono nella stessa distribuzione
dei dati. Fallback automatico alle frasi astratte in caso di errore.

**Strategia coseno (gratuita)**

```python
score(p) = dot(emb_p_norm, pole_pos) - dot(emb_p_norm, pole_neg)
```

Funziona quando `cosine_variance > COSINE_VARIANCE_THRESHOLD`.

**Strategia LLM batch scoring (fallback)**

LLM valuta ogni testo da 0 a 10 lungo l'asse (batch di 25).
Per dataset grandi (N > `LLM_SAMPLE_SIZE = 200`): campiona 200 punti e li fa
scorare, poi propaga i punteggi agli altri N-200 punti via **Ridge regression**
(alpha=1.0) invece di nearest-neighbour coseno. Riduce da ~48 a ~8 chiamate LLM.

**Perché Ridge invece di nearest-neighbour**

Il nearest-neighbour usa la similarità coseno di MiniLM per trovare il punto
"più simile" al quale ereditare il punteggio. Il problema: MiniLM cattura la
similarità di *topic*, non di *tono*. Per assi tonali (angry tone, formality)
una recensione calma di elettronica eredita il punteggio dalla recensione arrabbiata
di elettronica più vicina per topic — propagazione sbagliata.

La Ridge regression cerca invece la migliore combinazione lineare di tutte le 384
dimensioni dell'embedding che predice i punteggi LLM dei 200 punti campionati.
Anche se MiniLM non codifica esplicitamente il tono in una singola dimensione,
esiste una combinazione lineare debole che correla con esso (struttura della frase,
scelta lessicale, lunghezza). Con 200 labeled points Ridge la trova.

I log di terminale riportano entrambe le metriche per confronto:

```
[semantic-reembed] NN propagation   std=X.XXX  min=X.XX  max=X.XX
[semantic-reembed] Ridge propagation  train_R²=X.XXX  std=X.XXX  min=X.XX  max=X.XX
```

**`train_R²`**: quanto bene la direzione lineare fittata predice i punteggi LLM
sui 200 punti di training. Valori attesi: 0.1–0.4 per assi tonali (segnale debole),
0.4–0.8 per assi tematici (MiniLM cattura bene il topic).

**`std` Ridge vs NN**: se `std_ridge > std_nn`, Ridge ha trovato più varianza
lungo l'asse → separazione k-means potenzialmente migliore.

---

### Esperimento: Ridge vs NN — risultati empirici

I log di terminale riportano entrambe le metriche per ogni sessione che usa il fallback LLM.

| Sessione | Asse | std NN | std Ridge (α) | R² Ridge | Silhouette | Note |
|----------|------|--------|---------------|----------|------------|------|
| 1 | angry tone | 2.815 | 1.507 (α=1.0) | 0.525 | 0.340 | Ridge peggio di NN (0.551 prev.) |
| 2 | angry tone | — | — (α=0.01) | — | — | in corso |

**Analisi sessione 1 (α=1.0):**
- R²=0.525 è sorprendentemente alto per un asse tonale in MiniLM — esiste una direzione lineare reale
- Ma std Ridge (1.507) < std NN (2.815): la regolarizzazione forte schiaccia le predizioni verso la media
- Distribuzione Ridge più gaussiana/continua → k-means separa meno nettamente → silhouette cala (0.551→0.340)
- NN copia valori estremi (0 e 10) creando distribuzione bimodale artificiale ma efficace per k-means

**Decisione dopo i test:**
Se α=0.01 non recupera la silhouette almeno al livello NN (≥0.5), si torna a NN.
NN ha basi teoriche più deboli ma risultati empirici migliori su questo dataset/asse.

---

**`AxisNotDiscriminativeError`**

Sollevata quando `llm_std < LLM_STD_THRESHOLD`. `turns.py` la intercetta e
restituisce un ask-turn senza avanzare il turn_number — il DB non viene toccato.

**Matrice ibrida risultante**

```
X = [ orig_norm × sqrt(1 - axis_weight) | axis_norm × sqrt(axis_weight) ]
shape: (N, D+1)   dtype: float32
```

`axis_weight=0.7` → 70% asse, 30% topic. Lo scaling `sqrt` garantisce che
`axis_weight` sia la frazione esatta di segnale k-means proveniente dall'asse
(ragionamento geometrico: distanza quadratica attesa ≈ 2 per entrambe le componenti).

---

### `semantic_clustering.py`

```python
K_AUTO_MIN = 2
K_AUTO_MAX = 5
```

k viene scelto automaticamente via silhouette score (`_auto_select_k`) quando non
è passato esplicitamente. Testa k=2..5 sul nuovo spazio ibrido già calcolato e
restituisce il k con silhouette media più alta. Sostituisce il vecchio cap fisso
`AXIS_K_CAP = 3`.

Flusso:
```
reembed_for_axis(points, axis_hint, axis_weight)
  → X (matrice ibrida)
  → _auto_select_k(X, K_AUTO_MIN, K_AUTO_MAX)
  → KMeans(k) su X
  → dissolvi vecchi cluster (dissolved_at_turn = turn_number)
  → crea nuovi DbCluster
  → name_clusters(axis_hint=axis_hint)
  → scrivi SoftAssignment a turn_number
```

---

### `turns.py` — refactor del path semantico (rev 10)

**Differenze rispetto a rev 9:**

| Aspetto | Rev 9 | Rev 10 |
|---|---|---|
| Trigger | `axis_hint` nel payload (Turn 1 only) | Operazione `semantic_reembed` emessa da f_output (qualsiasi turno) |
| Schema | `InputOracle.axis_hint: Optional[str]` | Campo rimosso |
| Rilevamento asse successivi | `prior_turns[0].oracle_input.get("axis_hint")` | `_active_axis_from_history(prior_turns)` scorre ops |
| Intent ambiguo ("angry tone") | `no_change` silenzioso | `clarify` con domanda + `clarify_pending` nel state_snapshot |
| Conferma ("yes") | L'LLM doveva inferire l'asse dalla history | Deterministico: `_pending_clarify_axis` + `_is_affirmation` |

**Nuovo helper `_active_axis_from_history`:**

Cerca nei turni precedenti (al contrario) l'ultimo `op.type == "semantic_reembed"`
e ne restituisce l'`axis_label`. Accetta anche `axis_hint` per retrocompatibilità.

**Nuovo helper `_run_semantic_reembed`:**

Estrae il caricamento dei `DataPoint` dal DB e la chiamata a `semantic_clustering`
in una funzione separata riutilizzabile.

**Flusso deterministico per la conferma:**

```python
# 1. All'inizio di create_turn:
pending_axis = _pending_clarify_axis(prior_turns)
if pending_axis and _is_affirmation(payload.raw_text):
    # bypass f_output, costruisce raw direttamente
    raw = {"action": "semantic_reembed",
           "operations": [{"type": "semantic_reembed", "axis_label": pending_axis}], ...}

# 2. Dopo f_output, se action=="clarify":
pending_label = raw.get("axis_label")
snapshot_operations = [{"type": "clarify_pending", "axis_label": pending_label}]
operations = []  # nessuna modifica al clustering questo turno

# 3. Fine turno — state_snapshot usa snapshot_operations se presente:
ops_to_store = snapshot_operations if snapshot_operations else operations
```

---

### `f_output.txt` — nuove istruzioni per intent semantico

Aggiunto `clarify` come nuova action. Regola a tre casi:

**(1) Intent esplicito → `semantic_reembed` direttamente**
Frasi con verbo operativo: "cluster by X", "raggruppa per X", "look at this through X",
"split everything by X", "re-cluster by X", ecc.

**(2) Intent ambiguo → `clarify`**
Frase nuda senza verbo operativo e senza riferimento a cluster specifici:
"angry tone", "battery life", "sentiment", "tono arrabbiato".
Il `display` chiede: "Did you mean to reorganise all clusters along the '[X]' axis?
This will dissolve the current clusters and create new ones. Reply 'yes' or
'cluster by [X]' to confirm."
Il JSON include anche `axis_label: "[X]"` a livello top per permettere a `turns.py`
di salvarlo senza dover fare parsing del testo del display.

**(3) Conferma dopo clarify → `semantic_reembed` direttamente**
(Regola di prompt, ma di fatto bypassa l'LLM tramite il meccanismo deterministico
in `turns.py` descritto sopra.)

**Operazione `semantic_reembed` nel JSON:**
```json
{"type": "semantic_reembed", "axis_label": "angry tone"}
```
Esclusiva: non può essere combinata con merge/split/rename nello stesso turno.

---

### Naming con contesto asse

Quando `axis_hint` è presente, `name_clusters` inietta nel prompt il blocco
AXIS CONTEXT con regola **biforcata** (fix rev 9):

```
- Se i testi riguardano '{axis_hint}' → nome per posizione sull'asse
  (es. 'Poor Battery Life', 'Excellent Battery Life')
- Se i testi NON riguardano '{axis_hint}' → nome tematico descrittivo
  (es. 'Shipping and Returns', 'Product Durability')
```

Prima del fix, il prompt vietava esplicitamente i nomi tematici — causando
nomi errati ("High Battery Life") su cluster di contenuto non correlato all'asse.

---

### Vincoli aritmetici in `f_output.txt`

Il prompt copre esplicitamente i due casi di modifica del numero di cluster:

```
CLUSTER COUNT ARITHMETIC:
- K > N (aumentare): NEVER merge; emetti (K-N) split su cluster diversi
- K < N (ridurre): NEVER split; emetti UN SOLO merge di (N-K+1) cluster IDs
- CRITICAL: non fare "consolida tutto poi espandi" — va direttamente agli split
```

---

## Costo LLM stimato per sessione

| Operazione | Chiamate LLM | Costo stimato |
|------------|-------------|---------------|
| Re-embed — generazione poli (`_generate_axis_poles`) | 1 (sempre) | ~$0.001 |
| Re-embed — scoring coseno (variance > 0.01) | 0 | — |
| Re-embed — scoring LLM fallback (1200 punti, sample 200) | 8 × batch-25 | ~$0.01–0.02 |
| Re-embed — naming k cluster | 1 | ~$0.003 |
| Turno normale — `f_output` | 1 | ~$0.006–0.010 |
| Turno clarify — `f_output` | 1 | ~$0.006–0.010 |
| Turno conferma ("yes") | 0 (bypass deterministico) | — |
| Naming su merge/split | 1 per op | ~$0.003 |

**Stima sessione con re-embedding + 5 turni:** ~$0.07–0.12

---

## Miglioramenti da main integrati (2026-05-30)

### Commit P2 — `d0480c7` (Davide Sarti)

Refactor del path semantico in `turns.py`:

| File | Modifica |
|------|----------|
| `backend/routers/turns.py` | Rimosso path "Turn 1 only"; aggiunto `_active_axis_from_history`, `_run_semantic_reembed`; f_output sempre chiamato primo |
| `prompts/f_output.txt` | Istruzioni complete per `semantic_reembed` come operazione LLM con `axis_label` |
| `src/schemas.py` | `axis_hint` rimosso da `InputOracle` |
| `tests/test_turns_semantic_reembed_op.py` | Nuova suite per il path semantico via f_output |
| `ui/index.html` | `axis_hint` rimosso dal payload UI |

### Commit P2 — `8b20a29` (Thomas Ottonello)

| File | Modifica |
|------|----------|
| `tests/conftest.py` | Aggiunto `os.environ.setdefault("HARNESS_DRY_RUN", "true")` prima di qualsiasi import — risolve la race condition che causava vere chiamate LLM quando `test_semantic_clustering` importava harness prima di `test_turns_endpoint` |
| `tests/test_semantic_clustering.py` | Fix `clusters, assignments, _ = initial_clustering(...)` (3-tuple da commit `2e16880`) |
| `notes/p2/sprint-4-p2.md` | Sprint notes P2 |
| `notes/progress_report.md` | Progress report aggiornato |

### Miglioramenti integrati da main (2026-05-27)

| File | Cosa è arrivato |
|------|-----------------|
| `src/engine/f_uncertainty.py` | Riscrittura: `ClusterUncertainty`, `ClusterOverlap`, `ClusterCohesion`, `f_cluster_uncertainty()` |
| `src/engine/f_next_best_step.py` | Regole ask usano nomi cluster non UUID; soglia stop alzata a 5 |
| `src/engine/initial_clustering.py` | Softmax con temperatura scalata; silhouette wrappata in try/except |
| `src/engine/cluster_operations.py` | Fix merge: massa droppata (non ripiegata) → no dataset collapse |
| `src/engine/f_apply_operations.py` | Inline `new_name` su merge; `new_names`+`k` su split; rename preserva description |
| `backend/routers/turns.py` | `f_cluster_uncertainty`; display LLM visibile su qualsiasi action |
| `src/harness.py` | Cognitive load con floor division |
| `prompts/f_output.txt` | Merge richiede >= 2 id; nomi oracle VERBATIM |

---

## Sequenza di commit (branch + main)

| Hash | Descrizione |
|------|-------------|
| `0971e97` | feat: implement semantic re-embedding at Turn 1 |
| `669c992` | feat: expose semantic re-embed in UI and add terminal debug logging |
| `9ac923e` | fix: replace alpha/beta with axis_weight for correct geometric influence |
| `1e542c5` | fix: cap default k at 3 for semantic re-embedding |
| `202176e` | fix: propagate axis_hint to cluster naming throughout the call chain |
| `f45aafc` | fix: cap k at 3 in turns.py; speed up LLM scoring via sampling |
| `27d79d3` | fix: add cluster count arithmetic constraint to f_output |
| `37e2773` | feat: integrate main improvements into semantic-reembed branch |
| `df0888d` | docs: update report with integrated main improvements |
| `396698f` | docs: log open issue — free-form language causes multi-step merge+split |
| `a210804` | fix: forbid merge when K>N and split when K<N |
| `f8f00a1` | fix: populate token_usage and cost_usd in SystemTurn |
| `336bf08` | feat: dynamic input placeholder shows cluster-aware examples |
| `21d184b` | fix: placeholder shows all operations separated by dots |
| `00bb25c` | fix: tolerate unescaped quotes in LLM JSON |
| `0ae00f7` | fix: remove hardcoded 'angry tone' example from axis_context |
| `3b325c6` | feat: return friendly ask-turn when axis doesn't discriminate |
| `c5695f0` | fix: make axis-not-discriminative message dataset-agnostic |
| `e2aa841` | feat: generate LLM axis poles for better semantic re-embedding |
| `b82700c` | feat: auto-select k via silhouette; fix off-axis cluster naming |
| `8dd02bc` | Merge branch 'feature/semantic-reembed' into main |
| `d6d8aee` | fix: unpack 3-tuple from initial_clustering in test fixture |
| `50f2dee` | chore: integrate remote commits d0480c7 + 8b20a29 from main |
| `ad462aa` | feat(prompt): add clarify action for ambiguous semantic-axis inputs |
| `bc7b10c` | fix(prompt): handle confirmation reply after clarify |
| `a063783` | fix: deterministic clarify-confirm flow for semantic reembed |
| `783e6c0` | fix(naming): axis_context uses degree-based criterion instead of topic-relatedness |
| `6b79c4d` | fix(naming): remove hardcoded axis examples from axis_context prompt |
| `(current)` | feat: Ridge regression propagation replaces nearest-neighbour in LLM fallback |

---

## Test coverage

| Suite | Test | Stato |
|-------|------|-------|
| `test_f_semantic_reembed.py` | 37 | ✅ tutti passano |
| `test_semantic_clustering.py` | 25 | ✅ tutti passano |
| `test_turns_semantic_reembed_op.py` | (P2, vedi suite) | ✅ |
| `test_f_apply_operations.py` | 13 | ✅ tutti passano |
| `test_harness_json.py` | 7 | ✅ tutti passano |
| **Suite completa** | **236** | ✅ **236 pass, 0 fail** |

---

## Problemi aperti

### 1. Linguaggio libero non interpretato correttamente

L'esecutore `f_output` è ottimizzato per istruzioni operative dirette. Frasi
implicite ("these look the same", "make 5 clusters") restano inconsistenti.

**Categoria testata (2026-05-27):**
- Input: `"make 5 clusters"` da 3 cluster
- Risultato errato: LLM ha mergiato tutti e 3 invece di splittare
- Fix applicato: vincolo CLUSTER COUNT ARITHMETIC con proibizioni esplicite (`NEVER merge se K>N`)
- Da ritestarsi con test manuale

---

## Problemi risolti nel corso dello sviluppo

| Problema | Causa | Fix |
|----------|-------|-----|
| 8 test non patchabili | Import lazy di `SentenceTransformer` dentro le funzioni | Spostati a livello di modulo |
| k=5 non cappato | `turns.py` passava `k=len(clusters)` esplicitamente | Rimosso l'argomento `k` |
| ~48 chiamate LLM (troppo lento) | Ogni punto all'LLM | Sampling 200/N + propagazione NN |
| Nomi cluster topic-based nonostante asse | `name_clusters` non riceveva `axis_hint` | Propagazione completa nella catena |
| Due merge invece di uno (5→3) | Prompt non spiegava la regola merge N-way | Vincolo CLUSTER COUNT ARITHMETIC |
| Merge invece di split (3→5) | Vincolo copriva solo riduzione | Esteso con caso INCREASE |
| "make 5 clusters" da 3 → merge di tutti | LLM ignorava la direzione | Proibizioni esplicite; da ritestarsi |
| 3 test rotti in `test_f_apply_operations` | Kwarg `axis_hint=None` non previsto | Fixture aggiornate |
| Formula peso asse sbagliata | Scaling lineare non considera le norme | Sostituito con `axis_weight` e scaling `sqrt` |
| Token counter UI sempre a zero | `SystemTurn` mancava dei campi `token_usage`/`cost_usd` | Aggiunti campi, wire in `turns.py` |
| Placeholder input generico | Testo fisso non suggeriva azioni | Placeholder dinamico con nomi cluster reali |
| Cluster sempre nominati con nomi "angry tone" | Esempi hardcoded nel prompt naming | Esempi dinamici `High/Medium/Low {axis_hint}` |
| Asse non discriminativo → HTTP 422 | Eccezione propagata a HTTP | Catch → ask-turn, DB invariato |
| Messaggio errore Amazon-specific | Testo hardcoded | Riscritto dataset-agnostico |
| Poli coseno astratti mal embeddati | `"very {axis}"` fuori distribuzione | `_generate_axis_poles` con testi concreti |
| Parse JSON crash su inch mark | `json.loads` strict su LLM output | `loads_llm_json` + `_escape_unescaped_quotes` |
| Crash `content is None` | `completion.choices[0].message.content` può essere None | Guard esplicita |
| k sempre 3 indipendentemente dall'asse | Cap fisso `AXIS_K_CAP = 3` | `_auto_select_k` via silhouette su k=2..5 |
| Cluster off-axis nominati con nomi asse | Prompt vietava nomi tematici | Regola biforcata: asse se pertinente, tematico altrimenti |
| Re-embedding solo al Turn 1 | `axis_hint` in payload, check `turn_number == 1` | Refactor: operazione LLM a qualsiasi turno (Davide, `d0480c7`) |
| `axis_hint` rimosso da schema | Commit `d0480c7` — campo non più nel payload | Test suite aggiornata, UI aggiornata |
| 22 errori in `test_semantic_clustering` | `initial_clustering` ritorna 3-tuple da commit `2e16880` | `clusters, assignments, _ = initial_clustering(...)` |
| Race condition HARNESS_DRY_RUN | Env var impostata troppo tardi, dopo il primo import di `harness` | `conftest.py` imposta l'env prima di qualsiasi import |
| "angry tone" → `no_change` silenzioso | Nessuna regola per frasi ambigue senza verbo | Azione `clarify` + salvataggio `clarify_pending` in state_snapshot |
| "yes" dopo clarify → display OK ma nessun reclustering | LLM non trovava l'asse nella storia (SystemTurn ≠ raw f_output) | Bypass deterministico: `_pending_clarify_axis` + `_is_affirmation` in `turns.py` |
