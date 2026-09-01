# CLAUDE.md — GNM Webcam Puppet

## Purpose
Drive a 3D head from a webcam in real time, built on [GNM Head](https://github.com/google/GNM), Google's open-source parametric head model. Two implementations of the same idea: a browser build (active) and the Python desktop prototype it was rebuilt from. Public repo, Apache-2.0.

## Navigation
| Path | Contents |
|------|----------|
| `web_puppet/` | **Active.** Browser build — TypeScript + WebGL, no Python at runtime |
| `web_puppet/src/` | `gnm.ts` forward pass · `tracker.ts`/`tracking.ts` MediaPipe · `fit.ts` identity · `retarget.ts` per-frame · `gpu.ts` render · `main.ts` wiring |
| `web_puppet/tools/` | `export_assets.py` (model → compact binary), `sync_web_assets.mjs`, `dump_test_vectors.py` |
| `web_puppet/README.md` | The build log — module status table, and every design decision with its reasoning |
| `webcam_puppet/` | **Superseded.** Python desktop prototype — still the reference implementation for parts not yet ported |
| `README.md` | User-facing setup and demo instructions |

## Tech Stack
- Browser: TypeScript, Vite, raw WebGL, MediaPipe Tasks Vision
- Python 3.10+ for the prototype and the asset-export tools
- `npm run dev` · `npm run build` (typecheck + build) · `npm test` (node --test) · `npm run sync-assets`
- Python prototype: `python -m webcam_puppet.app`; tests are `*_test.py` beside their module

## Conventions
- **`web_puppet/` is where new work goes.** `webcam_puppet/` is kept as the reference implementation and bug record — do not refactor it to match the browser version
- **Generated assets are never committed** — `gnm_head.bin`/`.json`, `testdata/`, `public/mediapipe/`, `face_landmarker.task` are all gitignored and rebuilt by the tools that make them
- The GNM model itself is not vendored; it is installed from Google's repo
- `web_puppet/README.md` is the build log, not a summary — when a module lands, update its row in the status table and record *why*, not just *what*
- Tests sit next to their source (`fit.test.ts` beside `fit.ts`)
- This repo is public: keep paths relative and examples reproducible by a stranger

## Current Focus
Browser build through **M4e** (corrective layer from ARKit scores). Next is **M4d** — ARKit blendshapes for blink, which landmark geometry never solved. Then **M5**: identity-capture flow and UI. `M4c` (per-frame expression + pose retarget) is marked *done, prototype* — it works but has not been hardened.
