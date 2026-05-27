# Report: branch `feature/semantic-reembed`

**Autore:** P5 (Arianna Schiavi)
**Ultimo aggiornamento:** 2026-05-27
**Stato:** implementazione completa, in test manuale

---

## Obiettivo

Permettere all'oracle di orientare il clustering lungo un asse semantico
specifico (es. "angry tone", "battery life") già al primo turno, invece di
partire sempre da un k-means generico sul topic.

---

## Architettura della feature

### Punto di ingresso: `axis_hint` su Turn 1

`InputOracle` (schema Pydantic in `src/schemas.py`) accetta un campo opzionale:

```python
axis_hint: Optional[str] = Field(default=None)
```

Quando `axis_hint` è presente e `turn_number == 1`, `backend/routers/turns.py`
bypassa il path normale (`f_output → f_apply_operations`) e chiama invece
`semantic_clustering`.

### Path normale vs. path semantico

```
POST /turns

  turn_number == 1 AND axis_hint presente
    └─► semantic_clustering(axis_hint=...) → nuovi cluster + nuovi SoftAssignment

  altrimenti
    └─► f_output (LLM) → operazioni → f_apply_operations → cluster_operations
```

---

## File creati / modificati

### Nuovi file

| File | Scopo |
|------|-------|
| `src/engine/f_semantic_reembed.py` | Calcola la matrice ibrida `(N, D+1)` |
| `src/engine/semantic_clustering.py` | Orchestra re-embed + k-means + naming |
| `tests/test_f_semantic_reembed.py` | 15 unit test per `f_semantic_reembed` |
| `tests/test_semantic_clustering.py` | 22 unit test per `semantic_clustering` |

### File modificati

| File | Modifica |
|------|----------|
| `src/schemas.py` | `axis_hint` aggiunto a `InputOracle` |
| `src/engine/cluster_naming.py` | Parametro `axis_hint`; blocco AXIS CONTEXT nel prompt quando presente |
| `src/engine/cluster_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `src/engine/f_apply_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `backend/routers/turns.py` | Semantic path al Turn 1; recupero `session_axis_hint` per i turni successivi |
| `prompts/cluster_naming.txt` | Slot `{axis_context}` inserito |
| `prompts/f_output.txt` | Vincolo aritmetico N→K merge aggiunto ai CONSTRAINTS |
| `ui/index.html` | Placeholder adattivo al turno; `axis_hint` incluso nel payload al Turn 0 |
| `tests/test_f_apply_operations.py` | Fixture aggiornate con `axis_hint=None` |

---

## Dettaglio implementazione

### `f_semantic_reembed.py`

Due strategie + selettore ibrido.

**Strategia coseno (gratuita)**

```python
pole_pos = model.encode(f"very {axis_label}")
pole_neg = model.encode(f"not {axis_label} at all")
score(p) = dot(emb_p, pole_pos) - dot(emb_p, pole_neg)
```

Fallisce silenziosamente quando il modello MiniLM non separa l'asse:
`cosine_variance <= COSINE_VARIANCE_THRESHOLD (= 0.01)`.

**Strategia LLM batch scoring (fallback)**

LLM valuta ogni testo da 0 a 10 lungo l'asse. Batch di 25 per chiamata.

Per dataset grandi (N > `LLM_SAMPLE_SIZE = 200`): campiona 200 punti casuali,
li fa valutare, propaga i punteggi al resto via nearest-neighbour coseno.
Riduce da ~48 a ~8 chiamate LLM su 1200 punti.

**Matrice ibrida risultante**

```
X = [ orig_norm × sqrt(1 - axis_weight) | axis_norm × sqrt(axis_weight) ]
shape: (N, D+1)   dtype: float32
```

Ragionamento geometrico: con embedding row-normalizzati (‖riga‖ = 1) e punteggi
asse standardizzati (var = 1), la distanza quadratica attesa tra due punti casuali
è ≈ 2 per entrambi i componenti. Lo scaling `sqrt` garantisce che `axis_weight`
sia la **frazione esatta** di segnale k-means proveniente dall'asse.
`axis_weight=0.7` → 70% asse, 30% topic.

### `semantic_clustering.py`

```
AXIS_K_CAP = 3
```

Un asse semantico è unidimensionale (alto/medio/basso). Se k non è passato
esplicitamente, viene cappato a `min(k_corrente, 3)` per evitare cluster che
ridivengono topic-based per mancanza di varianza.

Flusso:

```
reembed_for_axis(points, axis_hint, axis_weight)
  → KMeans(k) sul nuovo spazio
  → dissolvi vecchi cluster (dissolved_at_turn = turn_number)
  → crea nuovi DbCluster
  → name_clusters(axis_hint=axis_hint)
  → scrivi SoftAssignment a turn_number
```

### Propagazione `axis_hint` per i turni successivi

Al Turn 1 l'`axis_hint` viene salvato nel DB come parte di `Turn.oracle_input`.
Dai Turn 2 in poi `turns.py` lo legge:

```python
session_axis_hint = prior_turns[0].oracle_input.get("axis_hint")
```

e lo passa a `f_apply_operations → merge/split_cluster → name_clusters`.
Così i nomi dei cluster generati da operazioni strutturali ai turni successivi
restano coerenti con l'asse semantico della sessione.

### Naming con contesto asse

Quando `axis_hint` è presente, `name_clusters` inietta nel prompt:

```
AXIS CONTEXT
These clusters were produced by re-embedding along the semantic axis "{axis_hint}".
Name each cluster to reflect where it falls along this axis — use degree/tone labels
(e.g. for 'angry tone': 'Very Angry', 'Mildly Frustrated', 'Neutral/Satisfied')
rather than topic labels like 'Book Reviews' or 'Electronics'.
```

### Vincolo aritmetico in `f_output.txt`

Il prompt dell'esecutore LLM includeva un bug latente: se l'oracle chiedeva
"passa da 5 a 3 cluster", il modello emetteva due merge da 2 ID ciascuno
invece di uno da 3 ID, riducendo il conteggio in modo errato.

Aggiunto ai CONSTRAINTS:

```
CLUSTER COUNT ARITHMETIC: if the oracle asks to go from N to K clusters, count
carefully. To reduce from N to K in one step, emit a SINGLE merge of exactly
(N - K + 1) cluster IDs. Verify: N - (len(cluster_ids) - 1) = K before responding.
```

### UI (`ui/index.html`)

La textarea al Turn 0 mostra il placeholder:

> Describe a semantic axis for this session (e.g. "angry tone", "battery life").
> Your first message will re-orient the entire embedding space along this axis.

Il payload al Turn 0 include `axis_hint: <testo inserito dall'utente>`.
Dai turni successivi il payload è normale (nessun `axis_hint`).

---

## Sequenza di commit sul branch

| Hash | Descrizione |
|------|-------------|
| `0971e97` | feat: implement semantic re-embedding at Turn 1 |
| `669c992` | feat: expose semantic re-embed in UI and add terminal debug logging |
| `9ac923e` | fix: replace alpha/beta with axis_weight for correct geometric influence |
| `1e542c5` | fix: cap default k at 3 for semantic re-embedding |
| `202176e` | fix: propagate axis_hint to cluster naming throughout the call chain |
| `f45aafc` | fix: cap k at 3 in turns.py; speed up LLM scoring via sampling |
| `27d79d3` | fix: add cluster count arithmetic constraint to f_output; update axis_hint test fixtures |

---

## Test coverage

| Suite | Test | Stato |
|-------|------|-------|
| `test_f_semantic_reembed.py` | 15 | ✅ tutti passano |
| `test_semantic_clustering.py` | 22 | ✅ tutti passano |
| `test_f_apply_operations.py` | 13 | ✅ tutti passano |
| Suite completa | 144 | ✅ 141 pass, 3 fail pre-esistenti* |

*I 3 fail in `test_turns_endpoint.py` sono un problema di ordinamento tra test
pre-esistente (non introdotto da questo branch); passano quando eseguiti in
isolamento.

---

## Problemi risolti nel corso dello sviluppo

| Problema | Causa | Fix |
|----------|-------|-----|
| 8 test non patchabili | Import lazy di `SentenceTransformer` ecc. dentro le funzioni | Spostati a livello di modulo |
| k=5 non cappato | `turns.py` passava `k=len(clusters)` esplicitamente, bypassando `if k is None` | Rimosso l'argomento `k` dalla chiamata |
| ~48 chiamate LLM (troppo lento) | Ogni punto veniva mandato all'LLM | Sampling 200/N + propagazione NN |
| Nomi cluster topic-based nonostante asse | `name_clusters` non riceveva `axis_hint` | Propagazione completa in tutta la catena |
| Due merge invece di uno (N→K sbagliato) | Prompt non spiegava l'aritmetica delle operazioni | Aggiunto vincolo CLUSTER COUNT ARITHMETIC |
| 3 test rotti in `test_f_apply_operations` | Kwarg `axis_hint=None` non previsto nei mock | Fixture aggiornate |
| Formula peso asse sbagliata (α=0.7, β=0.3 → 15%) | Scaling lineare non considera le norme dei vettori | Sostituito con `axis_weight` e scaling `sqrt` |
