---
name: sync-docs
description: Synchronizes project documentation, updates/generates unified daily changelogs in docs/changelog/YYYY_MM_DD.md, validates agent context, runs test verification, and prepares clean commit messages at the end of a developer work session.
---

# Sync Docs & Session Wrap-up Skill

This skill is executed when a developer is ready to wrap up their work session or requests to sync documentation, changelogs, or context.

## Workflow

### 1. Gather Context & Git Diffs
- **Date**: Determine today's date in `YYYY_MM_DD` format (e.g. `2026_08_14`).
- **Author**: Identify the author's name from git config (`git config user.name`) or ask/confirm if ambiguous.
- **Inspect Modifications**: Run `git status` and `git diff` to identify modified files across `backend/`, `src/`, `docs/`, and `tests/`.

### 2. Auto-Update Documentation & Agent Context
Check if the changes require documentation updates:
- **New/Modified Blocks** (`backend/blocks/`): Ensure the block is listed in [`docs/backend/blocks_status.md`](../../docs/backend/blocks_status.md) with input/output handles and auto-inference behavior.
- **Compiler/Pipeline Changes** (`backend/compiler.py`): Ensure changes to shape inference or code generation are reflected in [`docs/backend/compiler_design.md`](../../docs/backend/compiler_design.md).
- **Frontend Architecture/UI** (`src/`): Ensure component state models or new UI features are updated in [`docs/frontend/architecture.md`](../../docs/frontend/architecture.md).
- **Project Guidelines/Guardrails**: If architectural rules or conventions changed, update [`.agents/project_context.md`](../project_context.md).

### 3. Record in Unified Daily Changelog
Target file: `docs/changelog/YYYY_MM_DD.md` (e.g. `docs/changelog/2026_08_14.md`).

- **If the file does NOT exist**:
  Create it with the following structure:
  ```markdown
  # Changelog: Month DD, YYYY

  ## Author: <First Last>

  ### 1. <Category / Feature Area>
  - **<Specific Change>**: <Concise explanation and impact>.
  - **<Specific Change>**: <Description>.
  ```
  Then update [`docs/index.md`](../../docs/index.md) by adding a link at the top of the `## 📝 Changelog` section.

- **If the file ALREADY exists**:
  - Check if a `## Author: <First Last>` section already exists for this author. If so, append new categorized bullet points under that section.
  - If it does not exist, append a new `## Author: <First Last>` section at the bottom.

### 4. Run Test Verification
Run backend verification to ensure no regressions were introduced:
```bash
pytest backend/tests/ -v
```

### 5. Suggest Conventional Commit Message
Formulate a clean, standardized git commit message based on the session's work:
- Format: `<type>(<scope>): <summary>` (e.g. `feat(blocks): add SiLU activation and update documentation`)
- Provide the message to the user along with a summary of updated files.
