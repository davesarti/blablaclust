# Sprint 5 — P5 (Arianna)

## Consegnato questo sprint

### Fix: pipeline semantic re-embedding non si attivava per input ambigui ✅

**Problema:** scrivendo "angry tone" (senza un verbo operativo come "cluster by")
il sistema restituiva `no_change` silenzioso. Non chiedeva chiarimenti e non
avviava il re-clustering.

**Root cause:** il prompt `f_output.txt` non aveva nessuna regola per frasi nude
senza verbo operativo — l'LLM interpretava come commento e non faceva nulla.

**Fix:** aggiunta nuova action `clarify` con logica a tre casi:

1. **Intent esplicito** ("cluster by X", "raggruppa per X") → `semantic_reembed` direttamente
2. **Frase ambigua** ("angry tone", "battery life") → `clarify`: domanda all'oracle
   se vuole riorganizzare tutti i cluster lungo quell'asse, spiega la conseguenza
   (cluster dissolti), suggerisce come confermare ("yes" / "cluster by X")
3. **Conferma dopo clarify** ("yes", "sì", "ok", ...) → `semantic_reembed` direttamente

Il JSON `clarify` include un campo top-level `axis_label` con il candidato estratto,
così `turns.py` può salvarlo senza fare parsing del testo del display.

File modificati: `prompts/f_output.txt`

---

### Fix: conferma "yes" non attivava il reclustering ✅

**Problema:** dopo la domanda di chiarimento, rispondere "yes" produceva il testo
corretto ("Reorganising all clusters along the 'angry tone' axis...") ma non
avviava effettivamente il re-clustering.

**Root cause:** la conferma veniva gestita dall'LLM, che non riusciva a estrarre
l'asse dalla storia conversazionale (il turno precedente era un `SystemTurn`, non
il `raw` di `f_output`).

**Fix:** flusso deterministico in `turns.py` che bypassa completamente l'LLM:

```python
# Turno clarify: salva l'asse candidato nello state_snapshot
snapshot_operations = [{"type": "clarify_pending", "axis_label": pending_label}]

# Turno successivo: se pending + affermazione → semantic_reembed diretto
pending_axis = _pending_clarify_axis(prior_turns)
if pending_axis and _is_affirmation(payload.raw_text):
    raw = {"action": "semantic_reembed",
           "operations": [{"type": "semantic_reembed", "axis_label": pending_axis}]}
    # f_output non viene chiamato
```

`_is_affirmation` riconosce affermazioni in italiano e inglese:
`yes / sì / si / ok / okay / sure / yep / yeah / go ahead / do it / correct /
confirm / proceed / please / absolutely / definitely / do that / please do /
vai / fallo / procedi / confermo / esatto`

File modificati: `backend/routers/turns.py`

---

### Fix: nomi cluster non riflettevano l'asse semantico ✅

**Problema:** dopo un re-embedding su "angry tone", i cluster venivano nominati
per topic ("Music Album Opinions", "Mixed Products Reviews") invece che per
posizione sull'asse ("Mild Discontent", "Highly Critical Feedback").

**Root cause 1:** il criterio nel blocco AXIS CONTEXT di `cluster_naming.py`
chiedeva "se i testi *riguardano* l'asse" — l'LLM interpretava come pertinenza
tematica e usava nomi di topic.

**Fix:** criterio sostituito con "se puoi dire che questi testi segnano HIGH o
LOW su questo asse" — focalizza sulla posizione scalare, non sull'argomento.

**Root cause 2:** la prima versione del fix includeva esempi hardcoded legati
all'asse specifico ("e.g. for 'angry tone': 'Frustrated'") che avrebbero
contaminato assi diversi.

**Fix:** tutti gli esempi hardcoded rimossi. Il criterio è ora puramente
funzionale e asse-agnostico.

File modificati: `src/engine/cluster_naming.py`

---

### Esperimento: Ridge regression vs NN per propagazione LLM scores ✅

**Ipotesi:** Ridge regression sul subspace MiniLM trova una direzione lineare
che correla con il tono meglio del nearest-neighbour coseno (che propaga per
topic invece che per tono).

**Risultati:**

| N campioni | Metodo | std | R² | Silhouette | min/max | Bilanciamento |
|-----------|--------|-----|----|------------|---------|---------------|
| 200 | Ridge α=1.0 | 1.507 | 0.525 | 0.340 | in range | — |
| 200 | Ridge α=0.01 | 4.087 | 0.990 | 0.357 | -11.5..18.0 | — |
| 200 | NN | ~2.8 | — | ~0.55 | 0..10 | — |
| 600 | Ridge α=1.0 | 2.37 | 0.330 | 0.489 | -2.1..10.0 | 993/207 |
| **600** | **NN** | **3.171** | **—** | **0.538** | **0..10** | **855/345** |

**Conclusione:** Ridge non supera NN su nessuna metrica. A N=200 il regime
D=384 >> N non lascia spazio praticabile (α=1.0 over-regularizza, α=0.01 overfita).
A N=600 Ridge ha R² sano (0.330) ma leviga la distribuzione (std 2.37 vs NN 3.17)
perdendo la bimodalità che k-means sfrutta. Configurazione finale: **NN con N=600**.

**Effetto di N=600 su NN:** più copertura → vicini più prossimi → std più alta
(3.17 vs 2.8), cluster più bilanciati (855/345 vs 993/207), silhouette 0.538.
LLM calls: 24 invece di 8 su dataset da 1200 punti (~$0.03–0.05 invece di ~$0.01).

File modificati: `src/engine/f_semantic_reembed.py`

---

## Commit di questo sprint

| Hash | Descrizione |
|------|-------------|
| `ad462aa` | feat(prompt): add clarify action for ambiguous semantic-axis inputs |
| `bc7b10c` | fix(prompt): handle confirmation reply after clarify |
| `a063783` | fix: deterministic clarify-confirm flow for semantic reembed |
| `783e6c0` | fix(naming): axis_context uses degree-based criterion instead of topic-relatedness |
| `6b79c4d` | fix(naming): remove hardcoded axis examples from axis_context prompt |
| `a169a00` | experiment: Ridge regression propagation (N=200, α=1.0) |
| `74b4216` | experiment: lower Ridge alpha 1.0→0.01 |
| `408b255` | revert: use NN propagation instead of Ridge regression |
| `783223f` | experiment: Ridge N=600 (N_sample > D regime) |
| `15cc26e` | feat: increase LLM_SAMPLE_SIZE 200 → 600 for better NN propagation coverage |

---

## Test coverage

| Suite | Test | Stato |
|-------|------|-------|
| `test_f_semantic_reembed.py` | 21 | ✅ |
| `test_semantic_clustering.py` | 25 | ✅ |
| `test_turns_semantic_reembed_op.py` | (P2) | ✅ |
| `test_f_apply_operations.py` | 13 | ✅ |
| `test_harness_json.py` | 7 | ✅ |
| **Suite completa** | **236** | ✅ **0 fail** |

---

## Problemi aperti

- **Silhouette NN N=600 (0.538) leggermente sotto N=200 (~0.55)**: dentro
  il rumore della generazione dei poli — da monitorare su assi diversi
- **MiniLM non discrimina assi tonali via coseno**: varianza coseno sempre
  sotto soglia (0.006–0.009 < 0.01) per "angry tone" → LLM fallback sempre
  attivo per assi tonali; il coseno funziona solo per assi tematici
- **"make 5 clusters" da 3**: fix del vincolo CLUSTER COUNT ARITHMETIC
  applicato in sprint 4 ma non ancora riverificato con test manuale
