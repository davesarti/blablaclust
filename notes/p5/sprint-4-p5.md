# Sprint 4 — P5 (Arianna)

## Consegnato questo sprint

### Feature: Semantic Re-Embedding ✅

Implementata la feature principale: l'oracle può fornire un asse semantico che
riorienta l'intero spazio di embedding prima del k-means, invece di partire
sempre da un clustering topic-based. Il re-embedding può avvenire a qualsiasi
turno (non solo al Turn 1).

#### Architettura

- **`src/engine/f_semantic_reembed.py`** — due strategie + selettore ibrido:
  - Coseno su ancore — gratuito, usa MiniLM con frasi polo generate dall'LLM
  - LLM batch scoring (fallback se varianza coseno ≤ 0.01) — campiona 600/N
    punti e propaga via nearest-neighbour; `LLM_SAMPLE_SIZE` alzato da 200 a 600
    dopo esperimento Ridge vs NN (vedi sotto)
  - Matrice ibrida `(N, D+1)` con scaling `sqrt` per geometria esatta:
    `axis_weight=0.7` → 70% asse, 30% topic
- **`src/engine/semantic_clustering.py`** — orchestra re-embed + k-means + naming
- **`backend/routers/turns.py`** — path semantico; flusso deterministico per
  conferme post-clarify (bypass LLM per risposta "yes")
- **`src/schemas.py`** — `axis_hint` aggiunto a `InputOracle`
- **`src/engine/cluster_naming.py`** — blocco AXIS CONTEXT per nomi per
  grado/tono invece che topic; criterio basato su posizione scalare sull'asse,
  non pertinenza tematica

#### Fix inclusi in questo sprint

**Pipeline non si attivava per input ambigui** — aggiunta action `clarify` con
logica a tre casi: intent esplicito → `semantic_reembed` diretto; frase ambigua
→ chiede conferma con spiegazione; conferma → `semantic_reembed` deterministico.

**Conferma "yes" non avviava il reclustering** — flusso deterministico in
`turns.py` che salva l'asse candidato nello `state_snapshot` al turno clarify e
lo rilancia direttamente al turno successivo senza passare dall'LLM.

**Nomi cluster non riflettevano l'asse** — criterio AXIS CONTEXT sostituito con
posizione scalare HIGH/LOW; rimossi esempi hardcoded per asse-agnosticità.

---

### Esperimento: Ridge regression vs NN per propagazione LLM scores ✅

**Ipotesi:** Ridge regression sul subspace MiniLM trova una direzione lineare
che correla con il tono meglio del nearest-neighbour coseno.

| N campioni | Metodo | std | R² | Silhouette | Bilanciamento |
|---|---|---|---|---|---|
| 200 | Ridge α=1.0 | 1.507 | 0.525 | 0.340 | — |
| 200 | Ridge α=0.01 | 4.087 | 0.990 | 0.357 | — |
| 200 | NN | ~2.8 | — | ~0.55 | — |
| 600 | Ridge α=1.0 | 2.37 | 0.330 | 0.489 | 993/207 |
| **600** | **NN** | **3.171** | **—** | **0.538** | **855/345** |

**Conclusione:** Ridge non supera NN. A N=200 il regime D=384>>N non lascia
spazio praticabile. A N=600 Ridge leviga la distribuzione perdendo la bimodalità
che k-means sfrutta. Configurazione finale: **NN con N=600**.

---

### Integrazione miglioramenti da main ✅

| Area | Miglioramento |
|---|---|
| Uncertainty | Cluster-level (`ClusterOverlap`, `ClusterCohesion`) |
| Stop threshold | Alzato a 5; load=4 non ferma prematuramente |
| Softmax | Temperatura scalata → silhouette migliore |
| Merge fix | Massa fusa droppata — impedisce collasso dataset |
| Inline names | Oracle può nominare in merge/split nello stesso turno |
| Cognitive load | Floor division invece di round() |

---

### Risultati test manuali

| Sessione | axis_hint | silhouette | Cluster prodotti | Note |
|---|---|---|---|---|
| 1 | "angry tone" | 0.055 | 5 topic-based | prima del fix axis_weight |
| 2 | "angry tone" | 0.055 | 5 tone-aware | prima del fix k cap |
| 3 | "angry tone" | 0.151 | 3 tone-aware | k=3 ma lento (48 LLM calls) |
| 4 | "angry tone" | 0.475 | Very Angry / Neutral / Mildly Frustrated | dopo tutti i fix |

### Test coverage

| Suite | Test | Stato |
|---|---|---|
| `test_f_semantic_reembed.py` | 21 | ✅ |
| `test_semantic_clustering.py` | 25 | ✅ |
| `test_f_apply_operations.py` | 13 | ✅ |
| **Suite completa** | **236** | ✅ 0 fail |

## Problemi aperti

- **MiniLM non discrimina assi tonali via coseno**: varianza coseno sempre sotto
  soglia (0.006–0.009 < 0.01) per "angry tone" → LLM fallback sempre attivo per
  assi tonali; il coseno funziona solo per assi tematici
- **"make 5 clusters" da 3**: fix del vincolo CLUSTER COUNT ARITHMETIC applicato
  ma non ancora riverificato con test manuale
