# Sprint 4 — P5 (Arianna)

## Consegnato questo sprint

### Feature: Semantic Re-Embedding (branch `feature/semantic-reembed`) ✅

Implementata la feature principale del piano: al Turn 1 l'oracle può fornire
un asse semantico (`axis_hint`) che riorienta l'intero spazio di embedding
prima del k-means, invece di partire sempre da un clustering topic-based.

#### Architettura

- **`src/engine/f_semantic_reembed.py`** — due strategie + selettore ibrido:
  - Coseno su ancore (`"very {axis}"` / `"not {axis} at all"`) — gratuito
  - LLM batch scoring (fallback se varianza coseno ≤ 0.01) — campiona 200/N
    punti e propaga via nearest-neighbour; riduce da ~48 a ~8 chiamate LLM
  - Matrice ibrida `(N, D+1)` con scaling `sqrt` per geometria esatta:
    `axis_weight=0.7` → 70% asse, 30% topic
- **`src/engine/semantic_clustering.py`** — orchestra re-embed + k-means +
  naming; `AXIS_K_CAP = 3` cappa il default k per assi 1-D
- **`backend/routers/turns.py`** — path semantico al Turn 1 se `axis_hint`
  presente; recupero `session_axis_hint` dai turni successivi per naming
  coerente
- **`src/schemas.py`** — `axis_hint: Optional[str]` aggiunto a `InputOracle`
- **`src/engine/cluster_naming.py`** — blocco `AXIS CONTEXT` nel prompt
  quando `axis_hint` è presente; nomi per grado/tono invece che per topic
- **Propagazione completa**: `turns.py → f_apply_operations → merge/split_cluster
  → name_clusters` — tutti i cluster creati successivamente al Turn 1 restano
  coerenti con l'asse semantico

#### Prompt (`prompts/f_output.txt`)

Aggiunti vincoli all'esecutore LLM:
- **CLUSTER COUNT ARITHMETIC**: istruzioni esplicite per riduzione (merge N-way)
  e aumento (K-N split su cluster distinti); proibizione esplicita di
  "consolidate then expand"
- Merge richiede ≥ 2 id; no operazione `delete` — merge con cluster più simile
- `k` e `new_names` su split; nomi oracle usati VERBATIM

#### Test

- 15 test per `f_semantic_reembed` — tutti passano
- 22 test per `semantic_clustering` — tutti passano
- Test esistenti aggiornati per le nuove signature (`axis_hint`, `k`)

---

### Integrazione miglioramenti da main ✅

Letti i file su `main` con `git show main:<path>` e applicati manualmente
nel branch senza merge/rebase:

| Area | Miglioramento |
|------|---------------|
| Uncertainty | Cluster-level (`ClusterOverlap`, `ClusterCohesion`, `f_cluster_uncertainty`) — ask usa nomi non UUID |
| Stop threshold | Alzato a 5; load=4 non ferma più prematuramente |
| Softmax | Temperatura scalata → assegnamenti meno uniformi, silhouette migliore |
| Merge fix | Massa dei cluster fusi droppata (non ripiegata) — impedisce collasso dataset |
| Inline names | Oracle può nominare cluster in merge/split nello stesso turno |
| Rename | Preserva description esistente se oracle non la specifica |
| Cognitive load | Floor division invece di round() |

---

### Fix: token counter UI sempre a zero ✅

`SystemTurn` mancava dei campi `token_usage` e `cost_usd` (presenti nello
schema su main ma non nel branch); `turns.py` calcolava `usage` e `cost` ma
li scartava. Fix: aggiunti campi a schema, popolati in `turns.py` e passati al
`SystemTurn` prima del salvataggio. La UI mostra ora i valori reali per sessione.

---

### UI: placeholder dinamico nel campo di testo ✅

Il placeholder del campo input si adatta al turno corrente:
- **Turn 0**: guida per l'asse semantico
- **Turn 1+**: esempi con i nomi reali dei cluster correnti separati da `·`
  (merge, split, rename, make N clusters)

---

### Documentazione ✅

- `docs/semantic-reembed-report.md` — report tecnico completo della feature,
  aggiornato ad ogni modifica (pushato su `main` per condivisione con il gruppo)
- Copre: architettura, file modificati, fix applicati, problemi aperti,
  costi LLM stimati, prospettive future

---

## Risultati test manuali

| Sessione | axis_hint | silhouette | Cluster prodotti | Note |
|----------|-----------|-----------|-----------------|------|
| 1 | "angry tone" | 0.055 | 5 topic-based | prima del fix axis_weight |
| 2 | "angry tone" | 0.055 | 5 tone-aware | prima del fix k cap |
| 3 | "angry tone" | 0.151 | 3 tone-aware | k=3 ma lento (48 LLM calls) |
| 4 | "angry tone" | 0.475 | `Very Angry` / `Neutral/Satisfied` / `Mildly Frustrated` | dopo tutti i fix |

Silhouette passata da 0.055 a 0.475 dopo i fix.

## Problemi aperti

- **"make 5 clusters" da 3**: fix applicato (proibizione esplicita merge se
  K>N) ma non ancora riverificato con test manuale
- **Linguaggio libero non-operativo**: istruzioni implicite ("these look the
  same") ancora inconsistenti — documentato nel report, da testare
  sistematicamente

## Piano sprint 5 (P5)

- [ ] Harness LLM-as-oracle: script che fa girare una sessione completa con
      oracle LLM (DeepSeek via OpenRouter) — era in piano sprint 4, rimandato
- [ ] Batteria di test sistematici sui limiti di comprensione dell'LLM
      (6 categorie: conteggio implicito, linguaggio valutativo, riferimenti
      per nome, operazioni concatenate, contraddizioni, nomi verbatim)
- [ ] Notebook Jupyter con metriche finali (turns-to-convergence, costo/sessione)
- [ ] Valutare svincolo del re-embedding dal Turn 1 (cambio asse mid-session)
