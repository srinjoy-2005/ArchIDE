# Unified Schema, Central Python Editor & Dedicated Inspector

We have aligned ArchIDE with the unified project specification and upgraded the IDE layout to support first-class Python files and central workspace code viewing.

---

## 1. Cleaned Layout & Dedicated Right Inspector
- **Removed the cramped `<> Code` tab from the Right Panel**:
  - The right sidebar is now a single-purpose, focused **Inspector** (`src/components/RightPanel.tsx`).
  - Devotes 100% of panel space to block configuration, dynamic hyperparameter forms, tensor shape analysis, and layer documentation.
- **Collapsible Control**: Retained the smooth right drawer toggle button (`PanelRightClose` / `PanelRightOpen`).

---

## 2. Central Workspace Dual View (`Graph` $\leftrightarrow$ `Python Code`)
- **Main Workspace Code Viewer** ([`src/components/CentralCodeEditor.tsx`](file:///d:/ML/ArchIDE/src/components/CentralCodeEditor.tsx)):
  - Built a full-height Python code editor in the central panel with line numbering gutter, syntax-highlighting, copy-to-clipboard, and direct `.py` file download.
- **Dual View Mode Switcher** ([`src/components/DnDCanvas.tsx`](file:///d:/ML/ArchIDE/src/components/DnDCanvas.tsx)):
  - Added a sleek `[ ☩ Graph | <> Python Code ]` segmented toggle in the tab strip.
  - Clicking `[ <> Python Code ]` or clicking "Export PyTorch" in the header immediately opens the compiled Python code in the central workspace.
  - Clicking `[ ☩ Graph ]` switches back to the visual ReactFlow canvas.

---

## 3. Unified Project & File Schema with Coordinates Preservation
Locked down and implemented the project specification across the VFS and serialization engine ([`src/lib/store.ts`](file:///d:/ML/ArchIDE/src/lib/store.ts)):

1. **Manifest File (`archide.project.json`)**:
   - `name`, `version`, `entry_point`, `folders`, and `files` paths.
2. **Graph Files (`main.json`, `blocks/conv/res_block.json`)**:
   - **Coordinates (`position: { x, y }`)**: Preserved across all nodes on export/import, ensuring shared graphs render with their author's visual layout.
   - **Parameters (`parameters: [...]`)**: Declares module constructor arguments (`in_channels`, `out_channels`, `stride`).
   - **Node Execution Semantics (`block_id`, `label`, `varName`, `paramValues`)**: Preserved cleanly and decoupled from UI layout metadata.

---

## 4. Enhanced File Explorer
- **Distinct Iconography**:
  - Graph files (`.json`, `.arch`): Rendered with cyan network graph icon (`Network`).
  - Python files (`.py`): Rendered with gold Python code icon (`FileCode`).
- **File Selection & Creation**:
  - Clicking a `.py` file automatically switches the central workspace into the Python Code Editor.
  - Clicking a graph file switches to the visual node canvas.

---

## 5. Verification
- **Frontend Build**: `npm run build` completed with **0 errors** (TypeScript & Next.js static pages generated clean).
- **Backend Tests**: `pytest backend/tests -v` passed all **19 / 19 tests** (100% green).
