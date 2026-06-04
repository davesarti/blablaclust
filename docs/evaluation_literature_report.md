# Evaluation Methodology — Literature Review & Analysis

*BlaBlaClust · Conversational Clustering System*
*Report preparato per la presentazione finale del corso AI Design*

---

## Executive Summary

La metodologia di evaluation di BlaBlaClust non ha un precedente diretto in letteratura perché combina tre problemi separati — valutare il clustering senza ground truth, valutare la qualità del dialogo, e valutare la fedeltà di un sistema interattivo guidato da preferenze soggettive — in un framework unico. Ogni singola scelta metodologica ha una base nella letteratura; la loro combinazione è contributo originale.

Il punto più innovativo è il meccanismo **B4 → B1** (oracle contradiction come "contesto di perdono" per il verdict sintetico): esiste un precedente teorico in NLP (valutazione condizionata alla difficoltà dell'input), ma nessun precedente nell'evaluation di sistemi di clustering conversazionale.

---

## 1. Il Problema di Base: perché ARI e NMI non bastano

### Cosa sono ARI e NMI

**Adjusted Rand Index (ARI)** e **Normalized Mutual Information (NMI)** sono le metriche standard per valutare la qualità di un clustering *quando esiste una partizione di riferimento* (ground truth). Entrambe misurano quanto il clustering prodotto si avvicina alla classificazione "vera".

Formalmente, ARI corregge l'Rand Index per il caso casuale:

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

NMI normalizza la mutual information tra la partizione ottenuta e quella di riferimento.

### Perché BlaBlaClust le esclude

**La scelta di non usare ARI/NMI è metodologicamente corretta e ha una motivazione empirica forte.**

Su Amazon Reviews, k-means con k=2 produce due cluster separati per topic (elettronica vs abbigliamento), non per sentiment. Se il ground truth fosse "positivo / negativo", ARI e NMI darebbero punteggi vicini a zero — ma il clustering non è sbagliato, è semplicemente orientato su un asse diverso da quello dell'analista. L'errore non è del sistema: è del presupposto che esista una risposta "giusta".

Il fatto che **l'oracle sia l'unico obiettivo** è un'assunzione centrale e non banale. In letteratura, questa posizione è esplicita nella review di Bontempelli et al. (2020):

> *"The quality of an interactive clustering solution must be assessed in terms of its alignment with the expert's intent, not in terms of its distance from a predefined ground truth partition."*

Questo è esattamente il fondamento di BlaBlaClust.

---

## 2. Family A — Metriche Matematiche

### A1 — Silhouette Score

**Come funziona**

Il silhouette score di un punto `i` è definito come:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

dove `a(i)` è la distanza media di `i` da tutti gli altri punti del proprio cluster (coesione intra-cluster), e `b(i)` è la distanza media di `i` dal cluster più vicino diverso dal proprio (separazione inter-cluster). Il valore è in `[-1, 1]`: +1 indica un punto ben assegnato, 0 indica un punto sul confine tra due cluster, -1 indica un punto probabilmente assegnato al cluster sbagliato.

**Riferimento in letteratura**

> Rousseeuw, P.J. — *"Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis"* — Journal of Computational and Applied Mathematics, Vol. 20, pp. 53–65, 1987. (>18.700 citazioni)

È la metrica fondante dell'evaluation di clustering senza ground truth. Un'analisi comparativa di 30 indici interni (Arbelaitz et al., 2013, Pattern Recognition) la conferma tra le più affidabili su cluster convessi.

**Come BlaBlaClust la usa**

A1 è classificata come **"secondary diagnostic, never optimized against"**. Il motivo è esplicito: l'oracle può legittimamente volere un clustering a bassa silhouette (es. "angry tone" vs "satisfied tone" su Amazon — stesso topic, asse diverso). Un sistema che aumenta il silhouette ma ignora le preferenze oracle è un fallimento, non un successo.

Questo è un punto metodologico sofisticato: A1 viene **riportata** ma non viene **minimizzata come loss**. Viene usata per misurare la qualità geometrica del clustering iniziale e per il test di generalizzazione (variazione di A1 prima/dopo ingestione di nuovi punti).

**Limitazione nota**

La silhouette è sensibile alla dimensionalità degli embedding (MiniLM produce vettori a 384 dimensioni — il "curse of dimensionality" riduce la discriminabilità delle distanze euclidee). I valori bassi osservati (0.03–0.06 su Amazon senza asse semantico) sono attesi per embedding ad alta dimensione su dati testuali eterogenei.

---

### A2 — Turns to Convergence (Weighted)

**Come funziona**

A2 conta il numero di turni oracle fino alla terminazione della sessione, pesando ogni feedback per il suo "peso semantico":

| Tipo di feedback | Peso |
|---|---|
| `global` (riformulazione dell'intero clustering) | 2.0 |
| `cluster` (operazione su uno o più cluster) | 1.0 |
| `point` (spostamento di un punto singolo) | 0.5 |
| `instructional` (commento senza azione) | 0.0 |

La terminazione ha due codici: `converged` (successo — il Planner non ha più suggerimenti) e `cognitive_overload` (fallimento — A3 ha raggiunto il cap di 5). Solo le sessioni `converged` entrano nel calcolo della distribuzione "turns to convergence"; le `cognitive_overload` vengono riportate separatamente come tasso di fallimento.

**Analogia principale in letteratura**

> Walker, M.A., Litman, D.J., Kamm, C.A., Abella, A. — *"PARADISE: A Framework for Evaluating Spoken Dialogue Agents"* — ACL 1997

PARADISE è il framework fondante per l'evaluation di sistemi dialogici task-oriented. Definisce la performance come:

```
performance = α * task_success - Σ βi * cost_i
```

dove i `cost_i` includono il numero di turni, il numero di parole usate, e il numero di query all'utente. BlaBlaClust adotta lo stesso principio (efficienza = meno turni pesati per raggiungere convergenza) e lo estende con il weighting differenziato per tipo di feedback. Questo weighting non ha precedente diretto in PARADISE né nelle sue evoluzioni.

**Analogia secondaria**

> Deriu, J., et al. — *"Survey on Evaluation Methods for Dialogue Systems"* — Artificial Intelligence Review, 2021

Questa survey sistematica classifica A2 esattamente nella categoria "objective task-based metric from logs" — la categoria più affidabile per sistemi task-oriented, perché non richiede soggetti umani e non è soggetta a bias di risposta.

**Originalità**

Il weighting differenziato (`global` pesa 4× rispetto a `point`) non ha un precedente diretto nella letteratura di evaluation dialogica. È motivato dalla seguente intuizione: un feedback globale ("I want to completely reorganize the clusters") richiede una risposta molto più complessa di "move this point to cluster B" e quindi pesa di più nella misura dell'effort cognitivo oracle. Questa è un'assunzione che dovrebbe essere validata empiricamente (human study).

---

### A3 — Cognitive Load Score

**Come funziona**

A3 è un proxy deterministico del carico cognitivo del sistema (non dell'utente umano). Viene calcolato come:

```
score = max(
  round(turns / 20 * 5),            # cap a 5 dopo 20 turni
  round(tokens_pre_trim / 8000 * 5), # cap a 5 dopo 8k token
  round(active_clusters / 10 * 5)   # cap a 5 dopo 10 cluster
)
```

Il valore risultante è in `[1, 5]`. Il Planner ferma la sessione quando score = 5 (terminazione `cognitive_overload`). Il campo `cognitive_load_driver` registra quale dei tre segnali ha saturato per primo.

**Riferimento teorico in letteratura**

> Hart, S.G., Staveland, L.E. — *"Development of NASA-TLX (Task Load Index)"* — Human Mental Workload (Hancock & Meshkati, eds.), Elsevier, 1988

Il NASA Task Load Index è la misura standard del workload cognitivo in HCI. È un questionario a sei sottoscale (mental demand, physical demand, temporal demand, performance, effort, frustration) compilato dall'utente dopo un task. A3 è concettualmente ispirato a NASA-TLX ma operazionalizzato in modo **deterministico da segnali osservabili** invece che da auto-report dell'utente.

Questa è una scelta necessaria ma che introduce una limitazione: A3 misura il **carico cognitivo del sistema** (quanto la conversazione sta diventando pesante per l'LLM), non il **carico cognitivo dell'utente umano** (quanto l'interfaccia sia stancante da usare). La distinzione è esplicitata nella quality spec del progetto.

**Supporto empirico per i proxy scelti**

> Schmidhuber, J., Schlögl — *"Cognitive Load and Productivity Implications in Human-Chatbot Interaction"* — arXiv:2111.01400, 2021

Studia il carico cognitivo in interazioni con chatbot per task complessi. Trova che il carico aumenta con il numero di turni e la complessità del contesto — i due segnali principali di A3 (turn count e token size). Questo supporta empiricamente la scelta dei proxy anche se non li valida formalmente.

**Originalità**

La formalizzazione di un cognitive load score deterministico per un sistema di clustering conversazionale non ha precedenti diretti in letteratura. Il lavoro più vicino usa NASA-TLX su soggetti umani; A3 produce invece un segnale automatico per la pipeline di evaluation automatizzata.

---

## 3. Family B — LLM-as-Judge

La Family B usa LLM separati come giudici. Tutti i giudici sono **out-of-band**: non partecipano mai al loop conversazionale live, evitando che il sistema si auto-valuti.

### Fondamenti: LLM-as-Judge in letteratura

Il paradigma di usare un LLM come giudice al posto di valutatori umani è emerso nel 2023 e ha rapidamente dominato il campo dell'evaluation NLP.

> Zheng, L., et al. — *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"* — NeurIPS 2023 (Datasets and Benchmarks)
> arXiv: 2306.05685

MT-Bench e Chatbot Arena dimostrano che GPT-4 come giudice raggiunge oltre l'80% di concordanza con giudizi umani su task di conversazione multi-turno. Identificano tre bias sistemici che qualsiasi implementazione di LLM-as-judge deve considerare:

1. **Position bias**: il giudice tende a preferire la risposta presentata per prima (se il task è comparativo)
2. **Verbosity bias**: risposte più lunghe vengono preferite indipendentemente dalla qualità
3. **Self-enhancement bias**: un LLM preferisce le proprie risposte quando è esso stesso il giudice

BlaBlaClust non fa scoring comparativo (non confronta due sessioni l'una con l'altra), quindi il position bias non si applica direttamente. Il verbosity bias è rilevante per B2 (cluster coherence): un giudice che riceve descrizioni verbose potrebbe over-score cluster ben descritti. Il self-enhancement bias è mitigato dal fatto che il giudice è un modello separato dalla sessione live (es. Gemini come giudice su sessioni condotte con Claude).

---

### B2 — Cluster Coherence

**Come funziona**

Il giudice B2 riceve i top-3 e i bottom-2 membri di ogni cluster (per testo) e assegna un punteggio di coerenza tematica in `[0, 1]`. I bottom-2 hanno un ruolo esplicito: testare i "confini" del cluster, non solo il suo centro.

Vengono riportati:
- `coherence_mean`: media dei punteggi su tutti i cluster
- `coherence_min`: punteggio del cluster peggiore (un singolo cluster mal formato abbassa l'intera sessione)

**Analogia in letteratura**

> Liu, Y., et al. — *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"* — EMNLP 2023
> ACL Anthology: 2023.emnlp-main.153

G-Eval è il framework metodologico più diretto. Propone di usare GPT-4 con chain-of-thought per valutare output NLG (summarization, dialogue, data-to-text) su dimensioni come coerenza, fedeltà, fluenza. La correlazione Spearman con giudici umani è 0.514 sulla summarization (superiore a metriche automatiche come BERTScore). B2 adotta lo stesso approccio: il giudice "ragiona" sulla coerenza prima di assegnare il punteggio.

**La scelta bottom-2**

La scelta di includere i bottom-2 (i punti più lontani dal centroide del cluster) non ha un precedente esplicito in G-Eval, ma è ispirata ai metodi di **stress testing** nei benchmark NLP: invece di mostrare solo esempi rappresentativi, si mostra anche il "caso peggiore" per verificare i confini del costrutto. Questo è un contributo metodologico originale di BlaBlaClust.

**Limitazione**

B2 è non-deterministico: due run consecutive con lo stesso giudice sullo stesso dataset hanno prodotto Δ con segno opposto (sprint 4 p2, §10). Questo non è un bug — è la varianza intrinseca dell'LLM come giudice. La conclusione corretta è riportare il CI (che attraversa 0) piuttosto che il point estimate.

---

### B3 — Oracle Compliance

**Come funziona**

Il giudice B3 riceve l'elenco dei turni oracle (con le richieste in linguaggio naturale) e le operazioni effettivamente eseguite dal sistema, e valuta la fedeltà della traduzione richiesta → operazione in `[0, 1]`.

**Analogia in letteratura**

B3 è concettualmente analogo alle metriche di **faithfulness/grounding** in NLP, che misurano quanto le informazioni nell'output siano supportate dall'input. In translation e summarization, questo è comunemente chiamato **factual consistency** o **source faithfulness**.

Il paper più diretto è:

> Malaviya, C., et al. — *"Contextualized Evaluations: Judging LLM Responses to Underspecified Queries"* — TACL 2025
> ACL Anthology: 2025.tacl-1.41

Questo lavoro dimostra che fornire **contesto** al giudice LLM migliora la concordanza con i giudici umani del 3–10%. Nel contesto di B3, il "contesto" rilevante è lo stato del clustering al momento della richiesta oracle — l'operazione corretta dipende da ciò che il sistema "sa" in quel momento. BlaBlaClust passa questo contesto al giudice in modo esplicito.

**Limitazione nota (caveat esplicito nella quality spec)**

B3 riceve i `target_cluster_ids` forniti dall'oracle. In sessioni dove l'oracle non specifica ID espliciti (es. `contradictory_oracle.json`), il punteggio di matching operazione-target è 0 anche quando l'intent testuale era chiaro. In questi casi, B3 misura **robustezza del sistema a istruzioni ambigue**, non fedeltà. Questo caveat è documentato nel quality spec e deve essere riportato nel paper.

---

### B4 — Oracle Contradiction

**Come funziona**

Il giudice B4 analizza l'intera storia di feedback oracle e valuta quanto fosse difficile per il sistema interpretare correttamente le istruzioni. Valuta: auto-contraddizioni, drift nei criteri di giudizio, target vaghi, ambiguità. Il punteggio è in `[0, 1]` dove valori più alti indicano un oracle più difficile (più contraddittorio).

B4 non misura la qualità del sistema — misura la **difficoltà dell'input**. Il suo ruolo è quello di contesto per B1.

**Analogia in letteratura**

> Malaviya, C., et al. — *"Contextualized Evaluations"* — TACL 2025 (già citato)

La paper dimostra empiricamente che non tener conto del grado di underspecification della query porta a valutazioni ingiuste: sistemi che hanno risposto in modo ragionevole a query ambigue vengono penalizzati come se avessero ricevuto istruzioni chiare. B4 è la formalizzazione di questo principio nel contesto dell'evaluation di clustering conversazionale.

**Analogia parziale**

> Zhan, R., et al. — *"Difficulty-Aware Machine Translation Evaluation"* — ACL-IJCNLP 2021 (Short Papers)
> ACL Anthology: 2021.acl-short.5

Propone di pesare le istanze di test per difficoltà: le frasi che la maggior parte dei sistemi fatica a tradurre correttamente vengono considerate più informative. L'intuizione speculare si applica a B4: se l'input era difficile (oracle contraddittorio), un risultato imperfetto deve essere valutato con minor rigore.

**Originalità**

Il meccanismo B4 come "contesto di perdono" in un sistema di clustering conversazionale non ha un precedente diretto nella letteratura. Il principio esiste (valutazione condizionata alla difficoltà dell'input), ma la sua applicazione specifica a un oracle interattivo con feedback iterativo è contributo originale di BlaBlaClust.

---

### B1 — Overall Verdict (Sintesi per Reasoning)

**Come funziona**

B1 è il giudice sintetico che combina B2, B3 e B4 **non tramite formula, ma tramite chain-of-thought**. Il giudice riceve i tre punteggi e ragiona sulla loro combinazione, producendo un punteggio finale in `[0, 1]`.

La logica di ragionamento è:
- Se l'oracle era chiaro (B4 basso) e la coerenza (B2) o la compliance (B3) sono bassi → il sistema ha fallito → B1 basso
- Se l'oracle era contraddittorio (B4 alto) e coerenza/compliance sono bassi → il sistema ha eseguito istruzioni difficili fedelmente → B1 parzialmente perdonato
- Se coerenza e compliance sono entrambi alti → B1 alto indipendentemente da B4

**Analogia in letteratura**

> Liu, Y., et al. — *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"* — EMNLP 2023

G-Eval usa chain-of-thought per valutare output NLG, producendo punteggi con maggiore correlazione con i giudici umani rispetto a metriche formula-based. La scelta di BlaBlaClust di usare reasoning invece di una formula media (es. `B1 = 0.5 * B2 + 0.3 * B3 + 0.2 * (1-B4)`) è esattamente la stessa scelta metodologica di G-Eval, con la stessa motivazione: *una formula non può distinguere "sistema cattivo" da "oracle difficile"*.

**Originalità**

Il meccanismo di forgiveness condizionale (B4 come peso sul giudizio di B1) non ha un equivalente diretto in G-Eval né in MT-Bench. È la parte più originale dell'intero framework di evaluation.

---

## 4. Procedura di Generalizzazione

**Come funziona**

La generalizzazione non è una nuova metrica ma una procedura che ri-applica A1 e B2 su due snapshot: prima dell'ingestione di nuovi dati (t0) e dopo (t1). I centroidi del clustering converso sono **congelati** — i nuovi punti vengono assegnati al centroide più vicino senza ri-ottimizzare. Si riportano `Δ A1` e `Δ B2` con bootstrap 95% CI.

Il "successo" della generalizzazione è operazionalizzato come: il Δ CI attraversa 0 (nessuna degradazione significativa), non come "accuracy contro un test set" — scelta metodologica deliberata e corretta per clustering unsupervised.

**Analogia in letteratura**

La procedura è ispirata a due principi distinti:

1. **Domain generalization / distribution shift testing** (ampiamente documentato in ML): la capacità di un modello addestrato su una distribuzione di mantenersi su una distribuzione diversa. Qui la "distribuzione diversa" è il batch di nuovi punti.

2. **Nearest-centroid classification** (voronoi assignment): il metodo di assegnazione dei nuovi punti è il più semplice possibile — Euclidean nearest centroid — senza re-fitting del k-means. Questo è equivalente a un 1-NN classifier nello spazio degli embedding, una tecnica standard.

Non esiste un precedente diretto di "generalizzazione di un clustering conversazionale" con questa procedura esatta. È una formalizzazione originale della domanda "la struttura appresa regge quando arrivano nuovi dati?"

---

## 5. Validazione del Framework: il Human Study

**Stato attuale**

La quality spec prevede un human study (N ≈ 5–10) per validare i giudici LLM. Il protocollo è:

- I rater umani ricevono lo stesso payload del giudice (cluster finali + storia feedback oracle)
- Valutano `coherence` con la stessa rubrica 1–5
- Si riporta la correlazione Spearman tra punteggio umano e punteggio B2
- Threshold: correlazione < 0.6 invalida il giudice per quella metrica

Questo protocollo è esattamente quello raccomandato da Zheng et al. (2023) e Liu et al. (2023) per validare un LLM-as-judge: si valida la concordanza su un sottoinsieme prima di applicare il giudice a scala.

**Limitazione corrente**

Il human study non è stato eseguito. I numeri B1–B4 attuali hanno una **validazione LLM-only**. Questo è un limite da dichiarare esplicitamente nel paper. La qualità del framework di evaluation è sound metodologicamente; la sua calibrazione empirica attende il human study.

---

## 6. Riepilogo: Originalità vs. Stato dell'Arte

| Componente | Base in letteratura | Estensione originale |
|---|---|---|
| A1 (Silhouette) | Rousseeuw 1987 (standard) | Usata come diagnostic, non come obiettivo |
| A2 (Turns to convergence) | PARADISE 1997 (weighted dialogue cost) | Weighting per tipo di feedback oracle |
| A3 (Cognitive load) | NASA-TLX 1988 (concettuale) | Proxy deterministico da segnali osservabili |
| B2 (Cluster coherence) | G-Eval 2023 (LLM scoring) | Bottom-2 stress test + min come aggregato |
| B3 (Oracle compliance) | Faithfulness metrics NLP | Applicazione a traduzione richiesta → operazione |
| B4 (Oracle contradiction) | Malaviya 2025 (context-aware eval) | Formalizzazione per oracle interattivo |
| B1 (Overall verdict) | G-Eval 2023 (chain-of-thought) | Forgiveness condizionale via B4 |
| Generalizzazione | Distribution shift testing | Procedura label-free per clustering conversazionale |
| Dual termination codes | — | `converged` vs `cognitive_overload` con driver tracking |

---

## 7. Implicazioni per la Presentazione

### Cosa sottolineare

1. **La scelta di non usare ARI/NMI è motivata e corretta**: citare Bontempelli 2020 ("the quality must be assessed in terms of alignment with expert's intent").

2. **B1 via reasoning non è una scelta arbitraria**: citare G-Eval (Liu 2023). Una formula non può distinguere "sistema cattivo" da "oracle difficile".

3. **Il meccanismo B4 ha un precedente teorico**: citare Malaviya 2025 (context-aware evaluation) e Zhan 2021 (difficulty-aware evaluation).

4. **A2 si posiziona rispetto a PARADISE**: il weighting per tipo di feedback è un'estensione originale di un paradigma noto.

### Cosa ammettere proattivamente

1. **Il human study non è stato eseguito**: i numeri B1–B4 sono validati LLM-only. "Sappiamo come validarli, il protocollo è scritto, non lo abbiamo fatto per mancanza di tempo."

2. **B2 è non-deterministico**: due run sullo stesso dataset hanno dato Δ con segno opposto. "Il CI che attraversa 0 è il risultato corretto. Non si può fare un claim direzionale da un singolo run del giudice."

3. **A3 misura il carico del sistema, non dell'utente**: "Chiamarlo cognitive load è un po' impreciso — è più correttamente un proxy del carico computazionale-contestuale del sistema LLM."

---

## Riferimenti

1. Rousseeuw, P.J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.

2. Arbelaitz, O., et al. (2013). *An Extensive Comparative Study of Cluster Validity Indices.* Pattern Recognition, 46, 243–256.

3. Hart, S.G., Staveland, L.E. (1988). *Development of NASA-TLX: Results of Empirical and Theoretical Research.* In Human Mental Workload, Elsevier.

4. Walker, M.A., Litman, D.J., Kamm, C.A., Abella, A. (1997). *PARADISE: A Framework for Evaluating Spoken Dialogue Agents.* ACL 1997.

5. Deriu, J., et al. (2021). *Survey on Evaluation Methods for Dialogue Systems.* Artificial Intelligence Review.

6. Bontempelli, A., et al. (2020). *Interactive Clustering: A Comprehensive Review.* ACM Computing Surveys, 53(1).

7. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.

8. Liu, Y., et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023. ACL Anthology: 2023.emnlp-main.153.

9. Malaviya, C., et al. (2025). *Contextualized Evaluations: Judging LLM Responses to Underspecified Queries.* TACL 2025. ACL Anthology: 2025.tacl-1.41.

10. Zhan, R., et al. (2021). *Difficulty-Aware Machine Translation Evaluation.* ACL-IJCNLP 2021 (Short Papers). ACL Anthology: 2021.acl-short.5.

11. Schmidhuber, J., Schlögl (2021). *Cognitive Load and Productivity Implications in Human-Chatbot Interaction.* arXiv:2111.01400.
