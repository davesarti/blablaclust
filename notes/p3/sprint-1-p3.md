# Sprint 1 — P3 (Core Engine)

## What I built:
- **Engine skeleton**: `src/engine/` — created the directory and one file per core function: `f_output.py`, `f_next_state.py`, `f_next_best_step.py`, `f_uncertainty.py`, `f_eval.py`. Each function has the correct signature with typed inputs and outputs but returns a hardcoded stub for now, so the rest of the team can integrate against it without waiting for the real logic
- **README**: `README.md` — wrote the project-level readme covering setup, project structure, environment variables and key development rules
- **AGENTS.md**: documented the agentic pattern (Planner/Executor) with a function map and architecture diagram showing the data flow between all f_* functions — mandatory grading component #4

## Challenges:
I had to understand how the different parts of the system connect before I could define the function signatures — specifically what types come from `schemas.py` (P1) and what comes from `harness.py` (P4). Setting up the Python environment also took some time due to SSL errors when running `pip install`, solved by using the CPU-only PyTorch index to avoid downloading large unnecessary NVIDIA packages.

## Next steps:
- Implement real logic for `f_next_state` once `prompts/f_next_state.txt` is ready from P4
- Implement `f_uncertainty` once P2 has populated the SoftAssignment table with the initial k-means clustering
