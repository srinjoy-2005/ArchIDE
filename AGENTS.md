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

## GitHub Issues & Backlog Protocol
- **Single Source of Truth**: All tasks, bugs, and feature backlog items are tracked exclusively via GitHub Issues (`gh issue`).
- **Discovery**: When identifying a bug, defect, or planned enhancement, run `gh issue list -R srinjoy-2005/ArchIDE` to verify if it is already tracked. If not, create an issue using `gh issue create` with standard labels (`bug`, `enhancement`, `task`, `area:*`, `priority:*`).
- **Traceability**: Reference issues in commit messages, PRs, and daily changelogs (e.g., `Fixes #X` or `Ref #X`).
- **Resolution**: Once a fix or feature is verified with passing tests (`npx tsc --noEmit` and `pytest backend/tests/`), close the issue via `gh issue close <issue-number> -R srinjoy-2005/ArchIDE`.

