# ADR 001: Virtual File System & Headless Compilation

## Date
August 2026

## Context
Originally, ArchIDE stored graph configurations locally in the browser or sent single flat JSON files. As custom sub-modules and multi-file projects were introduced, managing state purely in the frontend became unscalable. Furthermore, automated backend testing was difficult because it required a browser to generate the Graph IR payloads.

## Decision
1. **Virtual File System (VFS)**: We implemented a Python-backed disk storage system (`backend/storage.py`) that manages standard directories (`workspace/graphs` for `.arch` JSON, and `workspace/python` for `.py`). The frontend `vfsStore.ts` acts as a mirror, heavily reliant on Server-Sent Events (SSE) to update its state when disk files change.
2. **Headless Project Loader**: We built `backend/project_loader.py`, allowing the backend to crawl a directory of `.json` graphs and compile an entire project without the frontend.
3. **Double-Pass Kahn's Sort**: To support nested modules, the compiler sorts graphs topologically twice—first at the file-dependency level, then at the node-level within each file.

## Consequences
- **Positive**: Backend can be tested autonomously via `pytest`. The IDE supports arbitrary nested custom blocks. Frontend state is perfectly synchronized across multiple browser tabs.
- **Negative**: Adds complexity to React Flow UI state management. We had to implement drag-aware debouncing in `DnDCanvas.tsx` to prevent infinite SSE loops between frontend structural changes and backend disk writes.
