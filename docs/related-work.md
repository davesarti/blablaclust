# Related Work and State-of-the-Art Positioning

**BlaBlaClust** — Conversational Clustering System
DOLSAS 2025-26, University of Trento

---

## Overview

This document contextualises BlaBlaClust within the current state of the art in
interactive and LLM-guided text clustering, drawing on five directly relevant
works. We examine how each work approaches the core challenges shared with our
system — embedding representation, oracle interaction, evaluation, and
convergence — and identify where BlaBlaClust sits in the design space.

---

## Related Work
We have selected 5 relevant publications from 2021 to 2026.

### Schild et al. (2021) / Schild (2024) — Interactive Clustering for Chatbot Design

> Erwan Schild, Marc Lossio-Ventura, Jean-Charles Lamirel, Mathieu Roche.
> *Concevoir un assistant conversationnel de manière itérative et semi-supervisée
> avec le clustering interactif.* EGC 2021. HAL hal-03133060. LORIA,
> Euro Information (Crédit Mutuel Alliance Fédérale).
>
> Erwan Schild. *Annotation collaborative par clustering interactif pour la
> conception d'assistants conversationnels.* PhD thesis, Université de Lorraine,
> defended 27 March 2024.

Schild et al. tackle the same core problem as BlaBlaClust — building a chatbot
intent taxonomy from scratch without a pre-defined label set — through an
iterative, constraint-based interactive clustering approach. This work is the
**direct pre-LLM precursor** to BlaBlaClust's interactive paradigm.

The pipeline alternates two phases in a loop:

1. **Automatic clustering**: the system clusters the text corpus without
   supervision.
2. **Constraint annotation**: a domain expert (the oracle) examines sampled
   document pairs and annotates each as either MUST-LINK (same intent) or
   CANNOT-LINK (different intent).
3. The system **re-clusters respecting the accumulated constraints** (e.g., via
   COP-KMeans).

No pre-defined taxonomy is required. The intent structure emerges from the
annotation sequence itself — the oracle needs only to judge similarity, not to
name or define categories in advance. This is a key conceptual parallel with
BlaBlaClust.

The 2024 PhD thesis expands the 2021 workshop paper into a full industrial
system: complete graphical interface, parameter optimisation study (cluster count,
sampling strategy, convergence thresholds), cost analysis (technical and human
annotation cost), multi-annotator conflict resolution, and quality indicators. The
industrial context (Crédit Mutuel Alliance Fédérale, a major French bank) provides
grounded, large-scale validation.

**Technical context**: this work predates both LLMs and transformer-based
embeddings. Text representations are traditional (bag-of-words or TF-IDF), and
constrained clustering uses classical algorithms. The paradigm is the same as
BlaBlaClust's; the machinery is entirely different.

---

### ClusterLLM (Zhang et al., 2023)

> Yuwei Zhang, Zihan Wang, Jingbo Shang. *ClusterLLM: Large Language Models as a
> Guide for Text Clustering.* EMNLP 2023. arXiv:2305.14871. UC San Diego.

ClusterLLM addresses the mismatch between LLMs (semantically powerful but
inaccessible as embedders) and small embedding models (locally runnable but
limited in semantic nuance). Rather than using LLMs to embed text directly, the
system uses them as a **relational oracle**: it poses pairwise and triplet
questions and uses the answers to fine-tune a small local embedder.

The pipeline has two stages:

**Stage 1 — Perspective.** The LLM answers ~1,024 triplet questions of the form
*"which of B or C is semantically closer to A?"*, sampled using an entropy-based
strategy that targets the most ambiguous instances. These answers are used as
contrastive training signal to fine-tune a small embedder (Instructor or E5),
pulling the embedding space toward the user-intended clustering perspective
(topic, intent, emotion, etc.).

**Stage 2 — Granularity.** The LLM answers pairwise questions (*"do A and B
belong to the same category?"*) sampled from a hierarchical cluster dendrogram.
The optimal number of clusters is the level of the hierarchy with the highest
consistency between LLM predictions and actual cluster membership.

Total cost: ~$0.60 per dataset using GPT-3.5. Evaluated on 14 datasets covering
intent discovery, topic mining, type discovery, and emotion detection.
ClusterLLM consistently improves over deep clustering and self-supervised
baselines.

---

### Dial-In LLM (Hong et al., 2025)

> Mengze Hong, Wailing Ng, Chen Jason Zhang, Yuanfeng Song, Di Jiang.
> *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer
> Service Dialogues.* EMNLP 2025. arXiv:2412.09049.
> Hong Kong Polytechnic University & WeBank.

Dial-In LLM targets intent clustering in large-scale Chinese customer service
dialogues, where cosine similarity fails in both directions: semantically
identical utterances can be far apart in embedding space, and semantically
distinct utterances can be close. The system replaces distance-based quality
metrics with two fine-tuned LLM judges.

**Coherence Evaluator** ($\mathcal{M}_\text{eval}$): a binary classifier that
labels each cluster as *good* (single coherent intent) or *bad* (mixed or
ambiguous). Accuracy >95% on unseen clusters (qwen14b). Used both as a
clustering guide and as a final quality metric (*Goodness score*).

**Intent Labeller** ($\mathcal{M}_\text{name}$): generates cluster labels in an
*Action-Objective* format (e.g., `inquire-insurance-fee`) tailored to dialogue
where every utterance has an action and a target.

The **iterative algorithm** proceeds as follows: at each iteration, the system
clusters the pool of unassigned sentences over a range of candidate $k$ values,
evaluates coherence, selects the $k$ that maximises the good/bad ratio, freezes
the good clusters, and returns bad-cluster sentences to the pool. This continues
until fewer than $\epsilon$ sentences remain unassigned.

A **post-correction** step embeds all cluster intent labels onto the unit
hypersphere $\mathbb{S}^{d-1}$ and merges clusters whose labels are closer than
a geodesic threshold $\theta = 0.8$, removing redundant clusters automatically.

The system was evaluated on a proprietary dataset of 55,085 sentences annotated
into 1,507 intent clusters from 100k+ real customer service calls — the largest
Chinese intent clustering benchmark released to date. Ground truth was
constructed by domain experts at WeBank, who manually labelled every utterance.
This was feasible because the customer service domain is **closed and stable**:
there is a finite and knowable set of customer intents.

---

### Perspectives (Fischer & Biemann, 2026)

> Tim Fischer, Chris Biemann. *Perspectives – Interactive Document Clustering in
> the Discourse Analysis Tool Suite.* arXiv:2602.15540v1, February 2026.
> University of Hamburg.

Perspectives is an interactive document clustering extension of the Discourse
Analysis Tool Suite (DATS), designed for Digital Humanities (DH) researchers who
need to explore large corpora through different analytical lenses without machine
learning expertise.

The pipeline is **aspect-focused**: before clustering, the user provides a
natural language instruction that steers the embedding (e.g., *"Identify the
topic"*, *"Summarize sentiment towards climate change"*). Optionally, an LLM
rewrites each document to emphasise the chosen aspect before embedding.

The core pipeline: LLM document rewriting → instruction-tuned embedding
(`multilingual-e5-large-instruct`, 500M) → UMAP dimensionality reduction →
HDBSCAN density-based clustering → cluster representation (c-TF-IDF keywords,
LLM-generated name and description, centroid, representative documents).

Human-in-the-loop refinement is provided through direct manipulation on an
interactive 2D document map: **Change Cluster**, **Add Cluster** (by selecting
documents or by typing a description), **Split**, **Merge**, **Remove**, and
**Accept Cluster Assignment**. When the user has validated enough examples, a
**Refine Model** operation triggers few-shot contrastive fine-tuning (SetFit +
LoRA, 2–16 labeled examples per cluster, one epoch), after which the entire
dataset is re-embedded with the updated model.

The system runs on a React + Plotly.js frontend with a FastAPI + Python backend,
using RQ workers for asynchronous embedding and clustering tasks — an
architecture directly comparable to BlaBlaClust's.

---

## Comparison with BlaBlaClust

### Interaction paradigm

The three works span a spectrum of how human intent enters the clustering loop:

| System | Interaction mode | Oracle role |
|---|---|---|
| **Schild et al. (2021)** | Binary MUST-LINK / CANNOT-LINK constraints | Domain expert, binary similarity judgments |
| **ClusterLLM** | Structured binary/triplet questions, batch | LLM, answers locally scoped questions |
| **Dial-In LLM** | None — fully automated | Fine-tuned LLM judge, no human needed |
| **Perspectives** | Direct UI manipulation (drag, assign, merge) | DH researcher acting on a visual map |
| **BlaBlaClust** | Free-form conversational dialogue | Human or LLM, per-turn natural language feedback |

BlaBlaClust occupies a distinct position: the oracle does not annotate binary
constraints (Schild), does not answer structured questions (ClusterLLM), does not
manipulate a visual interface (Perspectives), and is not replaced by an automated
judge (Dial-In). Instead, the oracle expresses preferences in free natural
language — describing what is wrong, suggesting a different grouping perspective,
or requesting a semantic axis — and the system interprets this and updates the
clustering accordingly.

The historical lineage is clear: Schild et al. established the paradigm of
letting the oracle *discover* a taxonomy through pairwise interaction rather than
specifying it upfront. BlaBlaClust inherits this core insight but replaces binary
constraint annotation with free-form natural language enabled by LLMs, lowering
the cognitive barrier and admitting richer implicit feedback per turn.

### Embedding architecture

All four systems use **fixed, pre-computed embeddings** as the base
representation — none modifies the embedding model during a session.

The key difference lies in how each system compensates for the inherent
limitations of generic embeddings:

| System | Embedding model | Compensation strategy | When |
|---|---|---|---|
| **Schild et al. (2021)** | Traditional (TF-IDF / bag-of-words) | Constrained clustering (MUST-LINK / CANNOT-LINK propagation) | Per iteration |
| **ClusterLLM** | Instructor / E5-large | Fine-tunes the embedder on LLM triplet feedback | Before clustering (offline, per dataset) |
| **Dial-In LLM** | BGE large (`bge-large-zh-v1.5`) | Fine-tuned LLM evaluates post-hoc cluster quality | After clustering (per iteration) |
| **Perspectives** | `multilingual-e5-large-instruct` | Instruction steers embedding; optional SetFit fine-tuning | Before clustering; after convergence |
| **BlaBlaClust** | `all-MiniLM-L6-v2` | Hybrid axis embedding (`D+1` vector) on oracle request | Before clustering (per turn, on demand) |

BlaBlaClust's **semantic reembedding** (`f_semantic_reembed.py`) constructs a
temporary hybrid space when the oracle requests a semantic axis: a scalar axis
score (from cosine pole comparison, or LLM batch scoring as fallback) is
appended to the original MiniLM embedding, weighted geometrically:

```
X = [ orig_norm × sqrt(1 - w),  axis_score × sqrt(w) ]    shape: (N, D+1)
```

This gives the oracle direct control over the clustering perspective at any turn,
without modifying the stored embeddings. The tradeoff is architectural: the axis
enters as a single additional dimension, while the remaining 384 dimensions still
encode the original topic geometry. An instruct-tuned embedding model (e.g.,
`e5-large-instruct`, `jina-embeddings-v3`) would reorient the *entire* embedding
space toward the requested axis rather than appending one scalar — a meaningful
qualitative difference that represents a clear future improvement path (see
[semantic-reembed-report.md](semantic-reembed-report.md)).

### Evaluation

The evaluation strategies reflect fundamentally different assumptions about
whether a ground truth exists.

**Dial-In LLM** and **ClusterLLM** have manually annotated ground truth labels,
enabling standard supervised clustering metrics (NMI, cluster accuracy). Dial-In
also measures **downstream accuracy**: a BERT classifier trained on the clustered
data is evaluated on held-out labelled examples, providing a direct measure of
practical utility. This is feasible because their domains are closed — there is
an objectively correct set of intents or categories to discover.

**Perspectives** evaluates with **KNN accuracy** on the 2D UMAP projection: a
k-nearest-neighbour classifier trained on 2D coordinates is used as a proxy for
how well the visual map separates the classes. Ground truth labels from the
dataset are used only for this evaluation step.

**BlaBlaClust** operates without any external ground truth. The correct
clustering is not knowable in advance — it is defined by the oracle during the
session. The evaluation suite therefore measures quality along dimensions that do
not require a reference labelling:

| Metric | What it measures | Analogue in literature |
|---|---|---|
| **A1 — Silhouette** | Geometric cluster separation | Sanity check in all three papers |
| **A2 — Turns / Termination** | Session efficiency | — |
| **A3 — Cognitive Load** | Oracle effort per turn | Not present in any of the three papers |
| **B2 — Coherence** | Intra-cluster semantic coherence (LLM judge) | Goodness score (Dial-In LLM) |
| **B3 — Compliance** | How closely the system followed oracle instructions | — |
| **B4 — Contradiction** | How often the system contradicted the oracle | — |

A3, B3, and B4 are specific to the conversational paradigm and have no
equivalent in the related work. B2 is conceptually identical to Dial-In's
Goodness score, but differs in implementation: Dial-In uses a domain-specific
fine-tuned binary classifier, while BlaBlaClust uses a general-purpose LLM
producing a continuous score with textual reasoning.

A notable distinction is that Dial-In's Goodness score serves **both as a
clustering guide and as a final metric**: it actively drives the iterative
algorithm by selecting which clusters to keep. In BlaBlaClust, B2 is computed
only at evaluation time; the oracle plays the guiding role during the session
itself.

---

## Original Contributions

Against this backdrop, BlaBlaClust's distinguishing contributions are:

1. **Conversational oracle interaction.** The oracle communicates in free natural
   language rather than annotating binary pairwise constraints (Schild),
   answering binary questions (ClusterLLM), manipulating a UI (Perspectives), or
   being replaced entirely by automation (Dial-In). This lowers the barrier to
   expert participation and encodes richer implicit feedback per turn. BlaBlaClust
   can be seen as the LLM-enabled evolution of Schild's paradigm: the same
   constraint-free taxonomy discovery, but with natural language replacing
   binary annotation.

2. **On-demand semantic axis steering.** The oracle can redirect the clustering
   perspective at any point during a session by requesting a semantic axis in
   natural language. The axis emerges from the dialogue rather than being
   specified upfront as in Perspectives.

3. **Conversational evaluation metrics.** A3 (cognitive load per turn), B3
   (compliance), and B4 (contradiction) are specific to the conversational
   paradigm and are not present in any of the three related works.

4. **No closed-world assumption.** Unlike Dial-In and ClusterLLM, BlaBlaClust
   does not require a pre-existing ground truth or a domain with a known,
   stable set of categories. The oracle defines correctness during the session.

---

## Future Directions Motivated by Related Work

**Instruct-tuned embedding for semantic reembedding.**
Replacing the current `MiniLM + cosine/LLM-fallback` pipeline with an
instruction-tuned embedding model (e.g., `e5-large-instruct`, `jina-embeddings-v3`)
would eliminate the LLM fallback entirely by reorienting the full embedding space
toward the requested axis, rather than appending a single scalar dimension.
Several providers offer this via API with no GPU or training required.

**Post-convergence fine-tuning from session history.**
Perspectives demonstrates that few-shot contrastive fine-tuning (SetFit + LoRA)
with 2–16 labeled examples per cluster improves downstream cluster quality.
BlaBlaClust's session log contains richer supervision signal than Perspectives
uses: every split operation is an explicit negative pair, every merge is an
explicit positive pair, and every requested semantic axis identifies a relevant
discriminative dimension. Mining this log for fine-tuning training data would be
a novel contribution beyond Perspectives' approach, which uses only final
validated assignments.

**Downstream accuracy as an additional evaluation metric.**
Following Dial-In LLM, if the clustered dataset is used for a downstream task
(e.g., training a classifier), measuring classification accuracy on held-out
examples provides a practical, externally grounded quality signal that
complements the intrinsic metrics currently implemented.

---

## References

- Schild, E., Lossio-Ventura, M., Lamirel, J.-C., & Roche, M. (2021). Concevoir
  un assistant conversationnel de manière itérative et semi-supervisée avec le
  clustering interactif. *EGC 2021*. HAL hal-03133060.
- Schild, E. (2024). Annotation collaborative par clustering interactif pour la
  conception d'assistants conversationnels. PhD thesis, Université de Lorraine.
- Zhang, Y., Wang, Z., & Shang, J. (2023). ClusterLLM: Large Language Models as
  a Guide for Text Clustering. *EMNLP 2023*. arXiv:2305.14871.
- Hong, M., Ng, W., Zhang, C. J., Song, Y., & Jiang, D. (2025). Dial-In LLM:
  Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service
  Dialogues. *EMNLP 2025*. arXiv:2412.09049.
- Fischer, T., & Biemann, C. (2026). Perspectives – Interactive Document
  Clustering in the Discourse Analysis Tool Suite. arXiv:2602.15540.
