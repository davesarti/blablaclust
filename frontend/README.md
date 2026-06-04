# BlaBlaClust — React Frontend

**Stack:** React 18 + TypeScript + Tailwind v4 + Vite + Plotly.js

Primary interface for the BlaBlaClust conversational clustering system.
See the [root README](../README.md) for full setup instructions.

## Quick start

```bash
cd frontend
npm install
npm run dev      # dev server → http://localhost:5173
npm run build    # production build
```

The backend must be running first:
```bash
PYTHONPATH=. python scripts/serve_ui.py   # from repo root
```

## Structure

```
src/
├── api/
│   └── client.ts              # Typed API client (all backend endpoints)
├── components/
│   ├── WelcomePage.tsx         # Landing page — dataset/session selector
│   ├── WorkspacePage.tsx       # Main workspace — cluster grid + chat sidebar
│   ├── ChatPanel.tsx           # Conversational oracle interface
│   ├── ClusterCard.tsx         # Cluster card with expand, pin, select
│   ├── AnalyticsPanel.tsx      # UMAP projection + turn-history visualisation
│   └── modals/
│       ├── DatasetsModal.tsx   # Upload, preview, delete datasets
│       ├── NewSessionModal.tsx # New session configuration
│       ├── EvalModal.tsx       # A1–B4 metrics for the current session
│       ├── UmapModal.tsx       # Geometry-aware UMAP per turn
│       ├── ExpandClusterModal.tsx  # Full point list with search and pin
│       ├── PersonaModal.tsx    # Persona session display
│       └── StopSessionModal.tsx    # Stop session + export
├── store/
│   └── AppContext.tsx          # Global app state (React context)
└── types/
    └── index.ts                # Shared TypeScript types
```

## Notes

- The legacy single-file HTML UI (`ui/index.html`) is still served at
  `http://localhost:8000/ui` for reference; see [`ui/README.md`](../ui/README.md).
- API base URL defaults to `http://localhost:8000` (Vite proxy configured in
  `vite.config.ts`).
