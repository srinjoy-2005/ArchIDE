---
description: Guidelines for session wrap-up, documentation synchronization, and changelog maintenance in ArchIDE.
always_on: true
---

# ArchIDE Agent Guidelines: Session Wrap-up & Documentation Synchronization

## 1. Multi-Turn Work vs. Session Wrap-up
- **During Active Coding (Intermediate Turns)**: Focus purely on implementing the feature, fixing bugs, or refactoring code. Do NOT generate changelogs or rewrite documentation on every turn.
- **At Session Wrap-up**: When the developer indicates they are finished with their session (e.g. says "sync docs", "wrap up", "all done", "generate changelog"), execute the [`sync-docs`](../skills/sync-docs/SKILL.md) skill workflow or run `python scripts/sync.py`.

## 2. Documentation Synchronization Rules
- **Backend Block Changes**: If blocks are added or modified in `backend/blocks/`, update [`docs/backend/blocks_status.md`](../../docs/backend/blocks_status.md) (with port details) and [`docs/backend/block_registry_spec.md`](../../docs/backend/block_registry_spec.md).
- **Compiler/Pipeline Changes**: If `backend/compiler.py` or `backend/main.py` is updated, ensure [`docs/backend/compiler_design.md`](../../docs/backend/compiler_design.md) matches the active implementation.
- **Frontend/UI Changes**: If canvas logic, custom nodes, or state handling changes in `src/`, update [`docs/frontend/architecture.md`](../../docs/frontend/architecture.md).
- **Core Guardrails**: If architectural rules are introduced, update [`.agents/project_context.md`](../project_context.md).

## 3. Daily Changelog Standard
- All daily logs are stored in `docs/changelog/YYYY_MM_DD.md` (e.g. `docs/changelog/2026_08_14.md`).
- Multi-developer entries within the same day are organized under `## Author: <First Last>` headers.
- If a new date file is created, add an entry at the top of `## 📝 Changelog` in `docs/index.md`.
