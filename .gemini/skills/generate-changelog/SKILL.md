---
name: generate-changelog
description: Automatically generates or updates the daily developer changelog in the docs/changelog directory based on recent work or git diffs, and updates the docs/index.md to point to it.
---

# Generate Changelog Skill

You are an automated changelog generator for the ArchiDE project. Follow these strict steps when a user asks you to update or generate the changelog.

## 1. Gather Context
- **Date**: Determine the current local date (YYYY_MM_DD format).
- **Author**: Ask the user for their name if it isn't obvious from the conversation context. Create a snake_case version of their first name for the filename (e.g., `gourav`).
- **Changes**: Analyze the conversation history, or run `git status` and `git diff` to see what files were modified, added, or deleted today.

## 2. File Naming Convention
Changelogs MUST be saved in the `docs/changelog/` directory.
The filename format is: `YYYY_MM_DD_<author_name_snake_case>.md`.
Example: `2026_07_26_gourav.md`.

## 3. Formatting the Changelog
If the file does NOT exist, create it with this structure:
```markdown
# Changelog: Month DD, YYYY
**Author**: First Last

## 1. <High-Level Category>
- **<Specific Feature>**: <Detailed description of the change and its impact>.
- **<Specific Feature>**: <Detailed description>.
```

If the file ALREADY exists, simply append the new categories and bullet points to the end of the file. DO NOT overwrite the existing entries.

## 4. Updating the Documentation Index
If you created a **new** changelog file, you MUST update `docs/index.md`.
- Locate the `## 📝 Changelog` section at the bottom of `docs/index.md`.
- Add a new bullet point at the TOP of the changelog list (reverse chronological order):
  `*   **[Month DD, YYYY - First Last](changelog/YYYY_MM_DD_name.md)**: <Short 1-sentence summary of the major changes>.`

## 5. Execution Rules
- Always use specific tools (`write_to_file` and `replace_file_content`) to manipulate the markdown.
- Do NOT hallucinate changes. Only document what you can verify in the workspace or what the user explicitly tells you.
- Always inform the user once the changelog and the index have been updated.
