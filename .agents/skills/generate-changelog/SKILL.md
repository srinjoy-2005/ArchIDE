---
name: generate-changelog
description: Automatically generates or appends to the daily developer changelog in docs/changelog/YYYY_MM_DD.md and updates docs/index.md.
---

# Generate Changelog Skill

> [!NOTE]
> For full end-of-session synchronization (including docs, context validation, tests, and commit messages), refer to the [`sync-docs`](../sync-docs/SKILL.md) skill.

## Changelog Format & Location
- **Filename**: `docs/changelog/YYYY_MM_DD.md` (e.g. `docs/changelog/2026_08_14.md`).
- **Sectioning**: Each author has a dedicated `## Author: <First Last>` section within the day's file.
- **Index Link**: If a new date file is created, add an entry at the top of `## 📝 Changelog` in `docs/index.md`.
