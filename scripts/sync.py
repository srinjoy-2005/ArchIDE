#!/usr/bin/env python3
"""
ArchIDE Project Sync Tool
Automates daily changelog generation, documentation index updating, test verification,
and commit message suggestions based on git diffs.
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

# Ensure UTF-8 output on all platforms
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
CHANGELOG_DIR = DOCS_DIR / "changelog"
INDEX_FILE = DOCS_DIR / "index.md"


def run_command(cmd, check=True):
    """Run shell command and return stdout string."""
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, shell=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        print(f"[Error] Command failed: {cmd}\n{res.stderr}")
    return res.stdout.strip()


def get_author_name():
    """Retrieve author name from git config or environment."""
    name = run_command("git config user.name", check=False)
    if not name:
        name = os.environ.get("USER", os.environ.get("USERNAME", "Developer"))
    return name


def get_git_diff_summary():
    """Analyze git status and modified files."""
    status = run_command("git status --porcelain", check=False)
    if not status:
        return []
    
    modified_files = []
    for line in status.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            modified_files.append((parts[0], parts[1]))
    return modified_files


def run_tests():
    """Execute backend pytest suite if pytest is installed."""
    print("\n[Testing] Running backend pytest suite...")
    cmd = f'"{sys.executable}" -m pytest backend/tests/ -v'
    res = subprocess.run(cmd, cwd=ROOT_DIR, shell=True)
    if res.returncode == 0:
        print("[Testing] All backend tests passed!")
        return True
    else:
        print("[Testing] Note: Run 'pip install pytest httpx' to enable local test runs.")
        return False


def sync_changelog(author, changes):
    """Create or append to today's daily changelog."""
    today = datetime.date.today()
    date_str = today.strftime("%Y_%m_%d")
    date_human = today.strftime("%B %d, %Y")
    
    CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    changelog_file = CHANGELOG_DIR / f"{date_str}.md"
    
    # Categorize modified files
    categories = {
        "Backend / Blocks": [f for _, f in changes if f.startswith("backend/blocks/")],
        "Compiler & Engine": [f for _, f in changes if f.startswith("backend/compiler") or f.startswith("backend/models") or f.startswith("backend/main")],
        "Frontend & UI": [f for _, f in changes if f.startswith("src/")],
        "Documentation": [f for _, f in changes if f.startswith("docs/")],
        "Tests & Tooling": [f for _, f in changes if f.startswith("backend/tests/") or f.startswith(".agents/") or f.startswith("scripts/") or f.startswith(".github/")],
    }
    
    content_lines = [f"## Author: {author}\n"]
    has_entries = False
    for cat_name, files in categories.items():
        if files:
            has_entries = True
            content_lines.append(f"### {cat_name}")
            for f in files:
                content_lines.append(f"- **{Path(f).name}**: Updated `{f}`.")
            content_lines.append("")

    if not has_entries:
        content_lines.append("### General Updates\n- **Project Sync**: Codebase maintenance and documentation sync.\n")

    section_text = "\n".join(content_lines).strip() + "\n"

    is_new_file = not changelog_file.exists()
    if is_new_file:
        header = f"# Changelog: {date_human}\n\n"
        changelog_file.write_text(header + section_text, encoding="utf-8")
        print(f"[Changelog] Created new daily changelog: {changelog_file.relative_to(ROOT_DIR)}")
        update_index_file(date_str, date_human)
    else:
        existing = changelog_file.read_text(encoding="utf-8")
        author_header = f"## Author: {author}"
        if author_header in existing:
            print(f"[Changelog] Author section '{author}' already recorded in {changelog_file.relative_to(ROOT_DIR)}")
        else:
            print(f"[Changelog] Appending author section to {changelog_file.relative_to(ROOT_DIR)}")
            changelog_file.write_text(existing.rstrip() + "\n\n---\n\n" + section_text, encoding="utf-8")


def update_index_file(date_str, date_human):
    """Insert new changelog link into docs/index.md."""
    if not INDEX_FILE.exists():
        return
    
    text = INDEX_FILE.read_text(encoding="utf-8")
    marker = "## 📝 Changelog\n"
    if marker in text:
        new_entry = f"*   **[{date_human}](changelog/{date_str}.md)**: Project updates and development log.\n"
        if f"changelog/{date_str}.md" not in text:
            idx = text.index(marker) + len(marker)
            updated = text[:idx] + new_entry + text[idx:]
            INDEX_FILE.write_text(updated, encoding="utf-8")
            print(f"[Index] Updated docs/index.md with {date_human} changelog link.")


def suggest_commit_message(changes):
    """Suggest conventional git commit message."""
    files = [f for _, f in changes]
    if any("backend/blocks" in f for f in files):
        prefix = "feat(blocks)"
    elif any("backend/compiler" in f for f in files):
        prefix = "feat(compiler)"
    elif any("src/" in f for f in files):
        prefix = "feat(ui)"
    elif any("docs/" in f for f in files):
        prefix = "docs"
    elif any("tests/" in f for f in files):
        prefix = "test"
    else:
        prefix = "chore"
        
    print("\n================ COMMIT SUGGESTION ================")
    print(f"{prefix}: update project components and sync documentation")
    print("===================================================\n")


def main():
    print(">> ArchIDE Project Wrap-up & Documentation Sync")
    author = get_author_name()
    print(f"Author: {author}")

    changes = get_git_diff_summary()
    if not changes:
        print("No uncommitted git changes detected.")
    else:
        print(f"Found {len(changes)} modified/untracked files.")

    sync_changelog(author, changes)
    tests_ok = run_tests()
    suggest_commit_message(changes)

    if tests_ok:
        print("[SUCCESS] Wrap-up and documentation sync completed successfully!")
    else:
        print("[INFO] Sync completed.")


if __name__ == "__main__":
    main()
