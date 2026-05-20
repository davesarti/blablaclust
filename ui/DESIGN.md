# UI Design Reference — Conversational Clustering

Documento di riferimento per le scelte grafiche di `ui/index.html`.  
Aggiorna questo file ogni volta che cambi colori, font, layout o animazioni.

---

## Tema generale

**Modalità:** Light (warm/parchment)  
**Estetica:** Research tool editoriale — warm neutral, serif + mono, rarefatto  
**Filosofia:** Nessuna dipendenza esterna. Tutto inline in un singolo file HTML.

---

## Palette — colori di riferimento

Questi quattro colori sono la base di tutto. Non usare altri colori senza motivazione.

| Colore      | Hex       | Ruolo nel sistema                                    |
|-------------|-----------|------------------------------------------------------|
| Olive scuro | `#706D54` | Accento primario, `--accent`, `--text-muted`         |
| Marrone mid | `#A08963` | `--text-faint`, `--border-s`, dot load lvl 3         |
| Beige caldo | `#C9B194` | Glow animazioni, `msg-diff` border, radial welcome   |
| Grigio luce | `#DBDBDB` | `--surface` — sfondo card e pannelli principali      |

---

## Variabili CSS

### Superfici e sfondi

| Variabile     | Valore    | Derivazione / uso                              |
|---------------|-----------|------------------------------------------------|
| `--bg`        | `#f0ede6` | Bianco caldo derivato dalla palette — body bg  |
| `--surface`   | `#DBDBDB` | **PALETTE** — header, drawer, sidebar, card    |
| `--surface-2` | `#d0ccc5` | Superficie secondaria                          |
| `--surface-3` | `#c8c3bb` | Superficie terziaria — messaggi sistema        |
| `--border`    | `#bfbbb3` | Bordo sottile                                  |
| `--border-s`  | `#A08963` | **PALETTE** — bordo enfatizzato, scrollbar     |

### Testo

| Variabile      | Valore    | Uso                                             |
|----------------|-----------|------------------------------------------------ |
| `--text`       | `#1a1916` | Quasi-nero caldo — testo principale             |
| `--text-muted` | `#706D54` | **PALETTE** — testo secondario, label, desc     |
| `--text-faint` | `#A08963` | **PALETTE** — label mono uppercase, placeholder |

> `#A08963` ha contrasto ~2.4:1 su `--surface` — usarlo solo per elementi decorativi (label, non corpo testo).  
> `#706D54` ha contrasto ~4.4:1 su `--bg` — accettabile per testo secondario.

### Accento

| Variabile      | Valore               | Uso                                        |
|----------------|----------------------|--------------------------------------------|
| `--accent`     | `#706D54`            | **PALETTE** — bottoni primari, stat-turn   |
| `--accent-dim` | `rgba(112,109,84,.1)`| Sfondo msg utente, tag cluster selezionato |

Testo sui bottoni `--accent`: `#f5f2ec` (bianco caldo, contrasto ~4.4:1).  
Hover bottone: `#5a5844` (oliva più scura).

### Colori funzionali (status / feedback)

| Variabile    | Valore    | Uso                             |
|--------------|-----------|---------------------------------|
| `--green`    | `#3a7548` | Badge ACTIVE, live dot drawer   |
| `--green-bg` | `rgba(58,117,72,.09)` | Sfondo badge ACTIVE  |
| `--blue`     | `#345888` | Badge CONVERGED, btn CSV        |
| `--blue-bg`  | `rgba(52,88,136,.09)` | Sfondo badge CONVERGED |
| `--red`      | `#883838` | Btn interrompi sessione         |
| `--red-bg`   | `rgba(136,56,56,.09)` | Hover btn rosso        |

---

## Colori per cluster (nth-child)

Cinque accenti distinti, desaturati/earthy per armonizzare con la palette calda.

| Cluster | `--ca`    | `--ca-rgb`   | Tono           |
|---------|-----------|--------------|----------------|
| 1°      | `#8b4040` | `139,64,64`  | Terracotta     |
| 2°      | `#706D54` | `112,109,84` | Oliva (palette)|
| 3°      | `#3a5a88` | `58,90,136`  | Blu ardesia    |
| 4°      | `#3d6b48` | `61,107,72`  | Verde foresta  |
| 5°      | `#6a4880` | `106,72,128` | Malva          |

`--ca` controlla: bordo sinistro card, count pill, size bar fill, glow `::after`, hover `btn-ghost`.  
`--ca-rgb` necessario per `linear-gradient(90deg, rgba(var(--ca-rgb), .04), ...)` — CSS non permette hex dentro rgba().

---

## Tipografia

| Variabile  | Stack                                                            | Uso                                              |
|------------|------------------------------------------------------------------|--------------------------------------------------|
| `--serif`  | `Georgia, 'Times New Roman', serif`                              | h1 welcome, h2 modali, session name, stat-turn   |
| `--mono`   | `ui-monospace, 'SF Mono', Menlo, 'Cascadia Code', monospace`     | Label uppercase, numeri, badge, diff, tag cluster|
| `--sans`   | `-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif` | Corpo, descrizioni, bottoni, textarea    |

**Font size base:** 13px · **Line height:** 1.5

---

## Layout

```
┌─────────────────────────────────────────┐  ← #header (44px, fixed, --surface)
│ CC  |  [session name — serif]  [ACTIVE] │
├──────────────────────────┬──────────────┤
│                          │              │
│   #cluster-area          │  #sidebar    │
│   (flex: 1, --bg)        │  (200px)     │
│                          │  (--surface) │
├──────────────────────────┴──────────────┤
│ ● Conversazione          [toggle btn]   │  ← drawer topbar (30px)
│ [chat history — flex: 1]               │
│ [typing indicator]                      │
│ [cluster tag] [textarea] [INVIA]        │  ← drawer input
└─────────────────────────────────────────┘  ← #drawer (fixed, --surface)
```

| Variabile CSS  | Valore  | Cosa controlla                        |
|----------------|---------|---------------------------------------|
| `--header-h`   | `44px`  | Altezza header fisso                  |
| `--drawer-col` | `116px` | Altezza drawer collassato             |
| `--drawer-exp` | `380px` | Altezza drawer espanso                |
| `--sidebar-w`  | `200px` | Larghezza sidebar                     |
| `--r`          | `2px`   | Border-radius globale (quasi niente)  |

---

## Cluster cards

**Struttura visiva:**
```
┌─── [--ca left border 3px] ───────────────────────────────┐
│  [emoji] [nome cluster]                    [count pill]  │
│  descrizione breve                                       │
│  [══ size bar (% relativa al cluster più grande) ══]     │
│                                             espandi →   │
└──────────────────────────────────────────────────────────┘
```

- **Size bar**: `width = (count / maxCount) * 100%`, calcolata live in `renderClusters()`. Colore = `--ca`, `opacity: 0.55`.  
- **Gradient laterale** (`::after`): `linear-gradient(90deg, rgba(ca-rgb, .04), transparent)` su 50% della card.  
- **Hover**: `translateY(-1px)` + `box-shadow: 0 3px 14px rgba(112,109,84,.15)` + `--surface-2`  
- **Selected**: `color-mix(in srgb, var(--ca) 7%, var(--surface))` come sfondo

---

## Animazioni

| Nome           | Dove si usa             | Durata / Easing                              |
|----------------|-------------------------|----------------------------------------------|
| `markGlow`     | Welcome — `◈`           | `3.2s ease-in-out infinite` — pulse `#C9B194`|
| `modalIn`      | Apertura modali         | `0.2s cubic-bezier(0.4,0,0.2,1)` — scale + Y|
| `slideRight`   | Messaggi utente         | `0.18s ease-out` — translateX(6px→0)         |
| `slideLeft`    | Messaggi sistema        | `0.18s ease-out` — translateX(-6px→0)        |
| `typingBounce` | Typing indicator dots   | `1.2s ease-in-out infinite` — bounce, sfasato (0 / 0.18s / 0.36s) |
| `liveDot`      | Pallino verde topbar    | `2.6s ease-in-out infinite` — opacity 1→0.25 |
| hover cards    | `.cluster-card:hover`   | `transition: 0.15s` su bg, border, shadow, transform |
| drawer expand  | `#drawer`, `#main-area` | `transition: 0.28s cubic-bezier(0.4,0,0.2,1)` su height |

---

## Sfondo body

Griglia di puntini via CSS puro — olive molto sottile sul bianco caldo:

```css
background-image: radial-gradient(circle at 1px 1px, rgba(112,109,84,.06) 1px, transparent 0);
background-size: 28px 28px;
```

---

## Welcome screen

- `#welcome::before`: radial gradient `rgba(201,177,148,.18)` → transparent al 68%  
  (`#C9B194` = rgb 201,177,148 — palette beige caldo)
- Mark `◈`: `font-size: 56px`, colore `var(--accent)` (`#706D54`), glow con `#C9B194`  
- `.welcome-meta`: tre stat in `--mono 10px`, colore `--text-faint`, separati da divisori verticali `--border-s`

---

## Chat drawer

| Elemento          | Sfondo                     | Bordo                          | Testo         |
|-------------------|----------------------------|--------------------------------|---------------|
| Msg utente        | `rgba(112,109,84,.1)`      | `rgba(112,109,84,.2)`          | `--text`      |
| Msg sistema       | `--surface-3` (`#c8c3bb`)  | `--border`                     | `--text-muted`|
| Diff block        | `rgba(201,177,148,.15)`    | `2px solid #C9B194`            | `--text-muted`|
| Typing indicator  | `--surface-3`              | `--border`                     | —             |
| Cluster tag       | `--accent-dim`             | `rgba(112,109,84,.22)`         | `--accent`    |

---

## Cognitive Load dots

5 rettangoli (`18×8px`, `border-radius: 1px`). Scala verde→rosso, integrata con la palette calda:

| Classe | Colore    | Tono          |
|--------|-----------|---------------|
| `.d1`  | `#3a7548` | Verde foresta |
| `.d2`  | `#7a9040` | Verde oliva   |
| `.d3`  | `#A08963` | **PALETTE** — marrone mid |
| `.d4`  | `#906040` | Arancio terracotta |
| `.d5`  | `#883838` | Rosso mattone |

Dot non attivi: `--surface-3` con bordo `--border`.

---

## Scrollbar personalizzata

Applicate a `#cluster-area`, `#sidebar`, `#chat-history`, modal expand.

```css
width: 3px; track: transparent; thumb: var(--border-s) / border-radius 2px
```

---

## Backdrop modali

```css
background: rgba(26,25,22,.6);
backdrop-filter: blur(3px);
box-shadow: 0 12px 40px rgba(112,109,84,.18), 0 0 0 1px var(--border);
```

---

## Note tecniche

- **`color-mix()`**: usato per selected state delle card. Chrome 111+, Firefox 113+, Safari 16.2+. Degrada gracefully.
- **`backdrop-filter`**: supportato ovunque tranne Firefox senza flag.
- **`--ca-rgb`**: workaround obbligatorio — non si può usare `rgba(var(--ca), .04)` con valore hex.
- **Contrasti**: `#706D54` su `#f0ede6` = ~4.4:1 (AA per testo grande). `#A08963` su `--surface` = ~2.4:1 (solo decorativo).
- **Testo su bottoni**: `#f5f2ec` su `#706D54` = ~4.4:1 (AA).
- **No CDN / font esterni**: self-contained, funziona offline.
