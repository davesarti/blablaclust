# Report: branch `feature/semantic-reembed`

**Autore:** P5 (Arianna Schiavi)
**Ultimo aggiornamento:** 2026-05-30 (rev 9)
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
| `prompts/semantic_axis_poles.txt` | Prompt per la generazione LLM dei poli dell'asse |
| `tests/test_f_semantic_reembed.py` | 37 unit test per `f_semantic_reembed` |
| `tests/test_semantic_clustering.py` | 28 unit test per `semantic_clustering` |
| `tests/test_harness_json.py` | 7 unit test per `loads_llm_json` / `_escape_unescaped_quotes` |

### File modificati

| File | Modifica |
|------|----------|
| `src/schemas.py` | `axis_hint` aggiunto a `InputOracle` |
| `src/engine/cluster_naming.py` | Parametro `axis_hint`; blocco AXIS CONTEXT dinamico; usa `loads_llm_json` |
| `src/engine/cluster_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `src/engine/f_apply_operations.py` | `axis_hint` propagato a `merge_clusters` e `split_cluster` |
| `src/engine/f_output.py` | Usa `loads_llm_json` invece di `json.loads(extract_json_text(...))` |
| `src/harness.py` | `_escape_unescaped_quotes` + `loads_llm_json`; fallback `{...}` su `extract_json_text` |
| `src/harness_openai.py` | Guard `content is None` in `call_gpt` e `call_gpt_async` |
| `src/harness_openrouter.py` | Guard `content is None` in `call_openrouter` e `call_openrouter_async` |
| `backend/routers/turns.py` | Catch `AxisNotDiscriminativeError` → ask-turn con `turn_number=0` |
| `prompts/cluster_naming.txt` | Slot `{axis_context}` inserito; regola escape delle virgolette interne al JSON |
| `prompts/f_output.txt` | Vincolo aritmetico N→K merge aggiunto ai CONSTRAINTS |
| `ui/index.html` | Placeholder adattivo al turno; `axis_hint` incluso nel payload; `buildDiff` helper estratto |
| `tests/test_f_apply_operations.py` | Fixture aggiornate con `axis_hint=None` |

---

## Dettaglio implementazione

### `f_semantic_reembed.py`

Due strategie + selettore ibrido. Flusso completo al Turn 1:

```
_generate_axis_poles(axis_label)
  → (pole_pos_text, pole_neg_text)   # 1 chiamata LLM, sempre

_cosine_axis_scores(points, pole_pos_text, pole_neg_text)
  → cosine_var > COSINE_VARIANCE_THRESHOLD?
      sì → usa i punteggi coseno direttamente
      no → _llm_axis_scores(points, axis_label)
              → llm_std < LLM_STD_THRESHOLD?
                  sì → raise AxisNotDiscriminativeError
                  no → usa i punteggi LLM
```

**Generazione poli LLM (`_generate_axis_poles`)**

Prima di calcolare i punteggi coseno, il sistema chiede all'LLM di generare due
testi concreti come poli dell'asse (prompt `prompts/semantic_axis_poles.txt`).
I testi concreti (es. una recensione furente vs. una neutra) si embedderanno
meglio delle frasi astratte ("very angry" / "not angry at all") perché vivono
nella stessa distribuzione dei dati.

In caso di errore (LLM non disponibile, risposta malformata, campi vuoti) fa
fallback automatico alle frasi astratte senza interrompere il pipeline.

**Strategia coseno (gratuita, dopo la generazione poli)**

```python
pole_pos = model.encode(pole_pos_text)   # testo concreto generato dall'LLM
pole_neg = model.encode(pole_neg_text)
# entrambi normalizzati a norma unitaria
score(p) = dot(emb_p_norm, pole_pos) - dot(emb_p_norm, pole_neg)
```

Funziona quando il modello MiniLM separa l'asse:
`cosine_variance > COSINE_VARIANCE_THRESHOLD (= 0.01)`.

**Strategia LLM batch scoring (fallback)**

LLM valuta ogni testo da 0 a 10 lungo l'asse. Batch di 25 per chiamata.

Per dataset grandi (N > `LLM_SAMPLE_SIZE = 200`): campiona 200 punti casuali,
li fa valutare, propaga i punteggi al resto via nearest-neighbour coseno.
Riduce da ~48 a ~8 chiamate LLM su 1200 punti.

**`AxisNotDiscriminativeError`**

Se anche il fallback LLM non riesce a discriminare l'asse
(std dei punteggi < `LLM_STD_THRESHOLD = 1.0`), viene sollevata
`AxisNotDiscriminativeError`. `turns.py` la intercetta e restituisce un
ask-turn con `turn_number=0` — il turn non viene scritto nel DB, quindi
lo stato UI rimane al Turn 1 e l'oracle può inserire un asse diverso.

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
K_AUTO_MIN = 2
K_AUTO_MAX = 5
```

k viene scelto automaticamente via silhouette score quando non è passato
esplicitamente. `_auto_select_k(X, k_min, k_max)` testa k=2..5 sul nuovo
spazio ibrido già calcolato e restituisce il k con silhouette media più alta.
Questo permette ai dati di determinare la granularità naturale dell'asse
(es. asse binario → k=2; asse a tre livelli → k=3) invece di cap fisso a 3.

Se k è passato esplicitamente (es. `k=3` dalla chiamata diretta) l'auto-select
viene bypassato.

Flusso:

```
reembed_for_axis(points, axis_hint, axis_weight)
  → X (matrice ibrida, già calcolata prima di scegliere k)
  → _auto_select_k(X, K_AUTO_MIN, K_AUTO_MAX)   # k=None → silhouette
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

Quando `axis_hint` è presente, `name_clusters` inietta nel prompt un blocco
AXIS CONTEXT con esempi di nomi **dinamici** basati sull'asse richiesto:

```
AXIS CONTEXT
These clusters were produced by re-embedding along the semantic axis "{axis_hint}".
Name each cluster to reflect its position on this axis.
Choose names that describe degree or intensity along the '{axis_hint}' spectrum
— for example 'High {axis_hint.title()}', 'Medium {axis_hint.title()}',
'Low {axis_hint.title()}' — or use natural synonyms that make the position
immediately clear. Do NOT use topic labels like 'Electronics' or 'Book Reviews'.
```

Nella rev 7 il blocco conteneva esempi hardcoded `'Very Angry'` /
`'Mildly Frustrated'` / `'Neutral/Satisfied'`: l'LLM li seguiva letteralmente
per qualsiasi asse. Il fix sostituisce quegli esempi con `High/Medium/Low
{axis_hint.title()}` generati dinamicamente dal nome dell'asse richiesto.

**Fix rev 9 — naming biforcato per cluster non-asse**

Un secondo bug: quando un cluster "non correlato all'asse" (es. "Unrelated
Customer Experiences") veniva splittato, i sotto-cluster ricevevano comunque
nomi asse (es. "High Battery Life", "Low Battery Life") perché il prompt
vietava esplicitamente i nomi tematici (`Do NOT use topic labels`).

Il blocco AXIS CONTEXT è stato riscritto con una regola biforcata:

```
- Se i testi del cluster riguardano '{axis_hint}' → nome per posizione
  sull'asse (es. 'Poor Battery Life', 'Excellent Battery Life').
- Se i testi NON riguardano '{axis_hint}' → nome tematico descrittivo
  che riflette il contenuto reale (es. 'Shipping and Returns').
```

Il prompt `prompts/cluster_naming.txt` include anche la regola di escaping per
evitare virgolette interne non escaped (es. misure `15.6"` scritte come
`15.6 inch`) che rompevano il parse JSON.

### Vincolo aritmetico in `f_output.txt`

Il prompt dell'esecutore LLM non spiegava l'aritmetica delle operazioni di
conteggio. Due bug osservati:

1. **Riduzione (5→3):** il modello emetteva due merge da 2 ID invece di uno da 3.
2. **Aumento (3→5):** il modello emetteva un merge (riducendo a 2) invece di due split.

Il CONSTRAINTS ora copre entrambi i casi:

```
CLUSTER COUNT ARITHMETIC:
- To REDUCE (K < N): emit a SINGLE merge of exactly (N - K + 1) cluster IDs.
  Verify: N - (len(cluster_ids) - 1) = K before responding.
- To INCREASE (K > N): emit exactly (K - N) separate split operations, each on
  a DIFFERENT existing cluster. Verify: N + (number of split operations) = K.
- Never mix merges and splits to hit a target count in one turn.
```

### UI (`ui/index.html`)

La textarea al Turn 0 mostra il placeholder:

> Describe a semantic axis for this session (e.g. "angry tone", "battery life").
> Your first message will re-orient the entire embedding space along this axis.

Il payload al Turn 0 include `axis_hint: <testo inserito dall'utente>`.
Dai turni successivi il payload è normale (nessun `axis_hint`).

---

## Costo LLM stimato per sessione

I costi sono visibili nella UI (contatore token + stima $) ora che il fix è applicato.
Stime teoriche per una sessione tipo con axis_hint:

| Operazione | Chiamate LLM | Costo stimato |
|------------|-------------|---------------|
| Turn 1 — generazione poli asse (`_generate_axis_poles`) | 1 (sempre) | ~$0.001 |
| Turn 1 — scoring asse coseno (se variance > 0.01) | 0 | — |
| Turn 1 — scoring asse LLM fallback (1200 punti, sample 200) | 8 × batch-25 | ~$0.01–0.02 |
| Turn 1 — naming 3 cluster | 1 | ~$0.003 |
| Turn 2+ — f_output per turno | 1 per turno | ~$0.006–0.010 |
| Turn 2+ — naming su merge/split | 1 per op | ~$0.003 |

**Stima sessione completa** (Turn 1 + 5 turni operativi): ~$0.07–0.12

Quando la strategia coseno funziona (cosine_variance > 0.01) il Turn 1 costa
solo 2 chiamate LLM: 1 per i poli + 1 per il naming.

I log del terminale mostrano già `cost_usd=$X.XXXX` per ogni `f_output` call. Da ora anche la UI accumula il totale per sessione.

---

## Miglioramenti da main e fix aggiuntivi (2026-05-30)

| File | Modifica |
|------|----------|
| `src/harness.py` | `_escape_unescaped_quotes` + `loads_llm_json`: parse JSON tollerante per virgolette non escaped (inch mark `15.6"`, frasi citate); fallback `{...}` su `extract_json_text` |
| `src/harness_openai.py` | Guard `content is None` in `call_gpt` e `call_gpt_async`: ValueError esplicita invece di crash su `NoneType` |
| `src/harness_openrouter.py` | Stessa guard in `call_openrouter` e `call_openrouter_async` |
| `src/engine/cluster_naming.py` | Usa `loads_llm_json` al posto di `json.loads(extract_json_text(...))` |
| `src/engine/f_output.py` | Usa `loads_llm_json` |
| `ui/index.html` | Helper `buildDiff` estratto inline → funzione separata; ritorna `null` per cluster invisibili (nome ma size=0) |

## Miglioramenti integrati da main (2026-05-27)

| File | Cosa è arrivato da main |
|------|-------------------------|
| `src/engine/f_uncertainty.py` | Riscrittura: `ClusterUncertainty`, `ClusterOverlap`, `ClusterCohesion`, `f_cluster_uncertainty()` — uncertainty a livello cluster invece di per-punto |
| `src/engine/f_next_best_step.py` | Regole ask usano nomi cluster non UUID; soglia stop alzata a 5 (load=4 non ferma più la sessione prematuramente) |
| `src/engine/initial_clustering.py` | Softmax con temperatura scalata sui dati → assegnamenti meno uniformi |
| `src/engine/cluster_operations.py` | Fix merge: la massa dei cluster fusi viene droppata (non ripiegata sul nuovo) → impedisce che l'intero dataset collassi sul cluster fuso |
| `src/engine/f_apply_operations.py` | Inline `new_name` su merge; `new_names`+`k` su split; rename preserva description esistente se l'oracle non ne specifica una |
| `backend/routers/turns.py` | Usa `f_cluster_uncertainty`; display LLM visibile su qualsiasi action (non solo `show`); `session.status="closed"` quando action è `stop` |
| `src/harness.py` | Cognitive load con floor division: score=5 solo vicino a MAX_TURNS |
| `prompts/f_output.txt` | Merge richiede >= 2 id; `k` e `new_names` su split; nomi oracle usati VERBATIM |

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
| `37e2773` | feat: integrate main improvements into semantic-reembed branch |
| `df0888d` | docs: update report with integrated main improvements |
| `396698f` | docs: log open issue — free-form language causes multi-step merge+split |
| `a210804` | fix: forbid merge when K>N and split when K<N in cluster count constraint |
| `f8f00a1` | fix: populate token_usage and cost_usd in SystemTurn so UI counters work |
| `336bf08` | feat: dynamic input placeholder shows cluster-aware examples for turns 2+ |
| `21d184b` | fix: placeholder shows all operations at once separated by dots |
| `00bb25c` | fix: tolerate unescaped quotes in LLM JSON (`loads_llm_json`, None content guards, `buildDiff`) |
| `0ae00f7` | fix: remove hardcoded 'angry tone' example from axis_context naming prompt |
| `3b325c6` | feat: return friendly ask-turn when axis doesn't discriminate the data |
| `c5695f0` | fix: make axis-not-discriminative message dataset-agnostic |
| `e2aa841` | feat: generate LLM axis poles for better semantic re-embedding |
| *(pending)* | feat: auto-select k via silhouette score; fix off-axis cluster naming |

---

## Test coverage

| Suite | Test | Stato |
|-------|------|-------|
| `test_f_semantic_reembed.py` | 37 | ✅ tutti passano |
| `test_semantic_clustering.py` | 25 | ✅ tutti passano |
| `test_f_apply_operations.py` | 13 | ✅ tutti passano |
| `test_harness_json.py` | 7 | ✅ tutti passano (nuova suite) |
| Suite completa | 163 | ✅ 163 pass, 3 fail pre-esistenti* |

*I 3 fail in `test_turns_endpoint.py` sono un problema di ordinamento tra test
pre-esistente (non introdotto da questo branch); passano quando eseguiti in
isolamento.

---

## Problemi aperti / limitazioni osservate in test manuale

### 1. Linguaggio libero non interpretato correttamente (APERTO)

L'esecutore `f_output` è ottimizzato per istruzioni operative dirette
("merge these clusters", "split into 3"). Frasi di alto livello che implicano
un'operazione senza nominarla esplicitamente vengono interpretate in modo
inconsistente.

**Esempio osservato (2026-05-27):**
- Stato: 3 cluster (`Very Angry`, `Neutral/Satisfied`, `Mildly Frustrated`)
- Input oracle: `"make 5 clusters"`
- Risultato atteso: 2 split su due cluster diversi → 5 cluster totali
- Risultato effettivo: il LLM ha prima **mergiato tutti e 3 in uno** (`Mixed Reviews`), 
  poi l'oracle ha dovuto dire `"split into 5"` → 2 turn invece di 1

**Analisi precisa:** il LLM ha mergiato tutti e 3 i cluster in 1 (`Mixed Reviews`), riducendo a 1 invece di aumentare a 5. Il vincolo INCREASE diceva cosa fare ma non diceva esplicitamente "NEVER merge se K > N". Il modello ha scelto una strategia "consolida tutto poi l'oracle chiede lo split" invece di andare diretto ai 2 split.

**Fix applicato (rev 5):** il vincolo è stato riscritto con:
- `If K > N: NEVER merge` — proibizione esplicita
- `If K < N: NEVER split` — proibizione speculare
- `CRITICAL: do NOT "consolidate then expand"` — proibizione esplicita della strategia sbagliata osservata

**Impatto sulla valutazione:** il numero di turni necessari per convergere viene artificialmente gonfiato quando l'oracle usa linguaggio naturale non-operativo. Da tenere in conto nell'Esperimento 3 (Free NL vs Structured NL). Da ritestarsi con "make 5 clusters" da 3 cluster.

---

## Problemi risolti nel corso dello sviluppo

| Problema | Causa | Fix |
|----------|-------|-----|
| 8 test non patchabili | Import lazy di `SentenceTransformer` ecc. dentro le funzioni | Spostati a livello di modulo |
| k=5 non cappato | `turns.py` passava `k=len(clusters)` esplicitamente, bypassando `if k is None` | Rimosso l'argomento `k` dalla chiamata |
| ~48 chiamate LLM (troppo lento) | Ogni punto veniva mandato all'LLM | Sampling 200/N + propagazione NN |
| Nomi cluster topic-based nonostante asse | `name_clusters` non riceveva `axis_hint` | Propagazione completa in tutta la catena |
| Due merge invece di uno (5→3) | Prompt non spiegava la regola merge N-way | Aggiunto vincolo CLUSTER COUNT ARITHMETIC per la riduzione |
| Merge invece di split (3→5 → risultato 2) | Vincolo copriva solo riduzione, il LLM applicava merge anche per aumentare | Esteso vincolo con caso INCREASE: K-N split su cluster distinti |
| "make 5 clusters" da 3 → merge di tutti e 3 in 1 | LLM ignorava la direzione; vincolo INCREASE mancava di "NEVER merge se K > N" | Riscritto vincolo con proibizioni esplicite per entrambe le direzioni; da ritestarsi |
| 3 test rotti in `test_f_apply_operations` | Kwarg `axis_hint=None` non previsto nei mock | Fixture aggiornate |
| Formula peso asse sbagliata (α=0.7, β=0.3 → 15%) | Scaling lineare non considera le norme dei vettori | Sostituito con `axis_weight` e scaling `sqrt` |
| Token counter UI sempre a zero | `SystemTurn` mancava dei campi `token_usage`/`cost_usd`; `turns.py` non li popolava | Aggiunti campi a schema, wire in `turns.py` |
| Placeholder input generico nei turni successivi | Testo fisso "Share your feedback" non suggeriva azioni | Placeholder dinamico con nomi cluster reali e tutte le operazioni disponibili |
| Cluster sempre nominati con nomi "angry tone" | Esempio hardcoded `'Very Angry' / 'Mildly Frustrated' / 'Neutral/Satisfied'` nel prompt di naming | Sostituito con esempi dinamici `High/Medium/Low {axis_hint.title()}` |
| Asse non discriminativo → HTTP 422 (toast errore) | `reembed_for_axis` propagava l'eccezione fino a HTTP 422 | Catch `AxisNotDiscriminativeError` in `turns.py` → ask-turn con `turn_number=0`; il DB non viene toccato |
| Messaggio errore "Amazon-specific" | Testo conteneva "Amazon reviews" come dataset di riferimento | Riscritto con linguaggio dataset-agnostico |
| Poli coseno astratti mal embeddati | `"very {axis}"` non vive nella distribuzione dei dati reali | `_generate_axis_poles` chiede all'LLM due testi concreti come poli; fallback alle frasi astratte in caso di errore |
| Parse JSON crash su inch mark / virgolette interne | `json.loads` strict su output LLM non sanitizzato | `loads_llm_json` + `_escape_unescaped_quotes`: tenta il parse strict, fallback con escape selettivo |
| Crash su `content is None` nei harness OpenAI/OpenRouter | `completion.choices[0].message.content` può essere `None` su refusal o errori API | Guard esplicita con `ValueError` descrittiva prima di usare il valore |
| k sempre 3 indipendentemente dall'asse | Cap fisso `AXIS_K_CAP = 3` ignorava la struttura reale dei dati | `_auto_select_k` testa k=2..5 via silhouette score sulla matrice ibrida già calcolata; es. asse binario → k=2 |
| Cluster off-axis nominati con nomi asse | `axis_context` vietava esplicitamente i nomi tematici (`Do NOT use topic labels`) | Regola biforcata: se i testi riguardano l'asse → nome asse; altrimenti → nome tematico descrittivo |

---

## Prospettive future

### 1. Svincolare il re-embedding dal Turn 1

Attualmente `axis_hint` è accettato **solo al Turn 1** (`if new_turn_number == 1 and payload.axis_hint`). Questo crea due limitazioni:

- L'oracle deve sapere l'asse semantico prima ancora di vedere i cluster iniziali.
- Se l'oracle cambia idea sull'asse a metà sessione (es. inizia con "angry tone" e poi vuole passare a "price sensitivity") non può farlo senza aprire una nuova sessione.

**Step proposto:** rendere il re-embedding invocabile a qualsiasi turno, non solo al primo. Il trigger potrebbe essere:
- Un campo `axis_hint` presente nel payload a qualsiasi turno (non solo turno 1)
- Oppure un'istruzione testuale riconosciuta da `f_output` che emette un'operazione di tipo `semantic_reembed` (da aggiungere al protocollo operazioni)

La difficoltà tecnica principale è la **coerenza storica delle SoftAssignment**: i turni precedenti sono stati calcolati in uno spazio diverso. Bisogna decidere se invalidare la history o tenerla come contesto narrativo senza usarla per il calcolo dell'uncertainty.

### 2. Test sistematici sui limiti di comprensione del LLM

I test manuali finora hanno rivelato comportamenti inconsistenti su istruzioni implicite ("make 5 clusters" invece di "split into 2"). Serve una batteria di test più sistematica per mappare i confini di comprensione dell'esecutore.

**Categorie da testare:**

| Categoria | Esempi da testare | Comportamento atteso |
|-----------|-------------------|----------------------|
| Conteggio implicito | "I want more clusters", "too many groups, reduce" | Split/merge corretto senza numero esplicito |
| Linguaggio valutativo | "these two look the same", "this cluster is too broad" | Merge / split senza nominare l'operazione |
| Riferimenti nominali | "merge the angry ones", "split the neutral cluster" | Identificare cluster per nome non per ID |
| Operazioni concatenate | "merge A e B e poi splitta il risultato in 3" | Rifiutare (forward-reference) e chiedere conferma step-by-step |
| Istruzioni contraddittorie | "merge A e B" dopo aver appena splittato A | `contradiction_detected: true` + spiegazione |
| Nomi non standard | `rename to "blah"`, `call it "???"` | Nome usato VERBATIM senza "miglioramenti" |

**Metrica:** per ciascuna categoria, contare quante sessioni producono l'operazione corretta al primo turno vs. quante richiedono correzione. Soglia di accettabilità proposta: ≥ 80% corretto al primo turno per le categorie "conteggio implicito" e "riferimenti nominali", ≥ 95% per "nomi verbatim".
