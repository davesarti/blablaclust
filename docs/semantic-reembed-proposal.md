# Proposta: Semantic Re-Embedding (Turn 1 Axis Extraction)

**Autore:** P5 (Arianna)  
**Data:** 2026-05-25  
**Stato:** proposta — non ancora implementata

---

## Problema

Le operazioni attuali del sistema (merge, split, move, rename) agiscono sulla
**topologia** dei cluster, non sul loro contenuto semantico.

Se l'oracle dice *"fai un cluster con le recensioni più arrabbiate"* oppure
*"separa per sentiment"*, il sistema non ha strumenti per eseguirlo: nessuna delle
operazioni correnti sa estrarre una dimensione semantica trasversale dai dati.

Radice tecnica: `all-MiniLM-L6-v2` è ottimizzato per sentence similarity
generica. Nello spazio di embedding prodotto da questo modello la dimensione
"arrabbiato vs. soddisfatto" non è una direzione geometrica isolata — è diffusa
in tutto lo spazio — quindi un k-means standard non la cattura.

---

## Soluzione proposta: due fasi di embedding

### Fase 0 — embedding generale + k-means iniziale (già esistente)
- Embedding con `all-MiniLM-L6-v2`, k-means su tutti i punti.
- Risultato: cluster di Turn 0 (pre-oracle) già scritti nel DB come
  `SoftAssignment.turn_number = 0`.

### Fase 1 — Turn 1: estrazione dell'asse semantico + re-embedding
Al primo turno dell'oracle, invece di processare il feedback come una semplice
operazione di clustering, il sistema:

1. **Estrae l'asse semantico** dall'intent dell'oracle (es. "sentiment", "batteria",
   "prezzo", "rabbia").
2. **Re-embeds** tutti i punti proiettando ciascuno lungo quell'asse.
3. **Ricalcola il k-means** nel nuovo spazio ibrido.
4. **Scrive un nuovo snapshot** nel DB a `turn_number = 1` con le nuove
   probabilità.

### Fase 2+ — turni successivi (operazioni strutturali)
I turni 2-N operano nello spazio già orientato semanticamente. Le richieste
dell'oracle possono essere più strutturali (merge, split, rename) oppure ulteriori
raffinamenti semantici lungo lo stesso asse.

---

## Implementazione: `f_semantic_reembed`

Due strategie, una ibrida che sceglie tra le due automaticamente.

### Strategia A — Cosine anchor poles (gratuita, usa infrastruttura esistente)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def f_semantic_reembed_cosine(points, axis_label: str):
    """
    Proietta ogni punto lungo l'asse semantico tramite coseno con ancore.

    axis_label: es. "angry", "battery life", "sentiment"
    Restituisce: array (N,) con score in [-1, 1] per ogni punto.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pole_pos = model.encode(f"very {axis_label}")
    pole_neg = model.encode(f"not {axis_label} at all")
    scores = []
    for p in points:
        emb = np.array(p.embedding)
        s = np.dot(emb, pole_pos) - np.dot(emb, pole_neg)
        scores.append(s)
    return np.array(scores)
```

**Pro:** nessuna chiamata LLM, nessun costo.  
**Contro:** funziona bene solo per assi semantici già impliciti nello spazio MiniLM
(es. topic similarity). Per assi tipo "arrabbiato" potrebbe avere varianza bassa
(il modello non li separa).

### Strategia B — LLM batch scoring

```python
def f_semantic_reembed_llm(points, axis_label: str, batch_size=25) -> np.ndarray:
    """
    Chiama l'LLM in batch per dare un punteggio 0-10 a ogni punto lungo l'asse.

    Costo: ceil(N / batch_size) chiamate LLM.
    Per 1200 punti con batch_size=25: ~48 chiamate.
    """
    scores = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        texts = "\n".join(f"{j}. {p.data['text'][:200]}" for j, p in enumerate(batch))
        prompt = render_prompt("semantic_reembed", axis=axis_label, texts=texts)
        response = call_llm(prompt)
        scores.extend(parse_scores(response))  # atteso: lista di int 0-10
    return np.array(scores, dtype=float)
```

**Pro:** funziona per qualsiasi asse semantico, anche non catturato da MiniLM.  
**Contro:** ~48 chiamate LLM per 1200 punti a ogni re-embedding (costo e latenza).

### Strategia ibrida (raccomandata)

```python
def reembed_for_axis(points, axis_label: str, alpha=0.7, beta=0.3):
    cosine_scores = f_semantic_reembed_cosine(points, axis_label)
    variance = np.var(cosine_scores)

    if variance > 0.01:
        # Il coseno distingue i punti: usa quello (gratis)
        axis_scores = cosine_scores
    else:
        # Il coseno non discrimina: chiama l'LLM
        axis_scores = f_semantic_reembed_llm(points, axis_label)

    axis_norm = (axis_scores - axis_scores.mean()) / (axis_scores.std() + 1e-8)
    original_embeddings = np.array([p.embedding for p in points])
    orig_norm = original_embeddings / (np.linalg.norm(original_embeddings, axis=1, keepdims=True) + 1e-8)

    return np.hstack([orig_norm * alpha, axis_norm.reshape(-1, 1) * beta])
```

Il vettore risultante combina l'embedding originale (peso α=0.7) con l'asse
semantico estratto (peso β=0.3). Il k-means successivo lavora su questo spazio
ibrido.

---

## Punto di inserimento nel codice

Il trigger è in `src/api/routers/turns.py`, quando `turn_number == 1`.

```python
# in turns.py, POST /sessions/{session_id}/turns
if turn.turn_number == 1 and oracle_input.axis_hint:
    new_embeddings = reembed_for_axis(all_points, oracle_input.axis_hint)
    run_kmeans(session, new_embeddings, turn_number=1)
    # write new SoftAssignments at turn_number=1
```

`axis_hint` è un campo opzionale da aggiungere a `InputOracle` (schema Pydantic).
Può essere estratto automaticamente dall'LLM se l'oracle usa linguaggio naturale,
oppure proposto dalla UI dopo il Turn 1.

---

## Contributo scientifico rispetto a Schild 2024

Schild usa MUST-LINK / CANNOT-LINK per aggiungere **vincoli** allo stesso spazio
di embedding. Il risultato è un k-means vincolato che rispetta i link ma opera
sempre nella stessa geometria.

Questo approccio invece **ri-orienta la geometria** stessa in base all'intent
semantico dell'oracle, il che è fondamentalmente diverso: non impone vincoli su
un clustering sbagliato, ma costruisce lo spazio in modo che le distinzioni
rilevanti per l'oracle siano già geometricamente visibili.

Potenziale risultato: convergenza più rapida (meno turni per raggiungere la
tassonomia target) e operazioni strutturali più efficaci nei turni successivi
(il k-means parte già da una geometria semanticamente sensata).

---

## Domande aperte / decisioni da prendere in gruppo

1. **Chi implementa `f_semantic_reembed`?** Potrebbe stare in P2 (embeddings) o
   P3 (engine functions). Il prompt `semantic_reembed.txt` va in `prompts/` (P4
   per il design).
2. **`axis_hint` è estratto automaticamente o inserito manualmente dall'oracle?**
   Opzione automatica: P4 aggiunge un piccolo step di estrazione asse prima del
   Turn 1. Opzione manuale: la UI (P5) mostra un campo dopo il Turn 1.
3. **Quando ri-embeddiamo?** Solo al Turn 1, o anche se l'oracle cambia asse
   radicalmente a un turno successivo? La seconda opzione complica il DB
   (le SoftAssignment dei turni precedenti diventano incomparabili).
4. **Soglia di varianza per la strategia ibrida:** 0.01 è una stima iniziale,
   da calibrare con qualche run sul dataset Amazon Electronics.
5. **Metrica di valutazione per il re-embedding:** confrontare turns-to-convergence
   e oracle_satisfaction_score con e senza re-embedding (esperimento controllato,
   stesso oracle prompt su sessioni diverse).

---

## Relazione con il piano sperimentale (Esperimento 3)

Questa proposta è la base per l'**Esperimento 3** del piano di evaluation:
*"Structured NL vs. Free NL vs. Semantic Re-Embedding"* — tre condizioni, stesso
oracle LLM, stessa sequenza di intent, metrica primaria: turns-to-convergence e
target_alignment_score al Turn finale.
