# ArchIDE - Agent Instructions

Welcome to ArchIDE. You are modifying a full-stack Next.js + PyTorch compilation IDE.

## Critical Guardrails
1. **Frontend State**: `DnDCanvas.tsx` is strictly UNCONTROLLED. State is synced to `vfsStore` via a debounced effect. To mutate nodes programmatically (e.g., from `PropertiesPanel`), you MUST use `useReactFlow().setNodes()`.
2. **Backend Authority**: The Python backend is the source of truth for block schemas. Frontend fetches from `/api/blocks`.
3. **Data Pipeline**: React Flow JSON -> `/api/compile` -> Kahn's Sort -> AST Generation -> Disk -> SSE -> VFS Store.

## Documentation Pointers
- **Architecture**: `docs/architecture.md` (System map, state philosophy)
- **API Contracts**: `docs/contracts.md` or `backend/models.py`
- **History/Why**: `docs/decisions/` (ADRs)
- **Changelog**: `docs/changelog/`

## Workflow Rules
- **DO NOT** manually document new blocks or maintain block lists (`blocks_status.md` is deprecated).
- **DO NOT** update documentation for bug fixes or local refactors.
- **Session Wrap-up**: When the session ends, follow `.agents/rules/session_wrapup.md` and use the `sync-docs` skill to update the changelog.
- **Tests**: Always run `npx tsc --noEmit` and `pytest backend/tests/` before completing a task.
