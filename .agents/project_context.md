# ArchIDE - Project Context for Agents

This document provides critical context for agents operating within the ArchIDE repository. Read these guidelines before attempting any refactoring or feature additions.

## 1. Architecture Overview
ArchIDE is a full-stack node-based PyTorch architecture builder. 
- **Frontend**: Next.js (React), React Flow (`@xyflow/react`), TailwindCSS, Zustand.
- **Backend**: Python (FastAPI), PyTorch generation compiler logic.
- **Communication**: Frontend sends JSON payloads describing nodes and edges to the `/api/compile` endpoint, and fetches available block definitions from `/api/blocks`.

## 2. Technical Guidelines & Guardrails
### Frontend (React Flow & VFS)
- **UNCONTROLLED MODE IS REQUIRED**: The `ReactFlow` component in `src/components/DnDCanvas.tsx` is strictly designed to be **uncontrolled** (using `defaultNodes` and `defaultEdges` instead of `nodes={nodes}`). This allows internal React Flow state to be the single source of truth.
- **State Manipulation**: Do NOT use `useNodesState` or `useEdgesState`. To manipulate nodes from outside the canvas (like in `PropertiesPanel`), use `useReactFlow().setNodes()`.
- **Selection**: We rely on reading the internal store (e.g. `const selectedNode = useNodes().find(n => n.selected)`) rather than `useOnSelectionChange` local state tracking to ensure UI always stays perfectly synced with the canvas.
- **VFS Snapshotting**: Before switching tabs or files in `FileTabBar` or `FileExplorer`, call `updateFileState(activeFileId, getNodes(), getEdges())` to snapshot live canvas state to Zustand.

### Backend (FastAPI, Compiler & Submodules)
- **Topological Sorting**: `backend/compiler.py` uses Kahn's algorithm to resolve dependencies. Node cyclic dependencies throw a `ValueError` which gets propagated back as a `400 Bad Request`.
- **Submodule Constructor & Shape Propagation**: Custom submodules extract graph parameters (`GraphData.parameters`), forward arguments to instances (e.g. `ResBlock(in_channels=64, ...)`), and run isolated sub-graph passes in `infer_shapes` overriding input nodes with actual incoming tensor shapes.
- **Headless Project Loader**: `backend/project_loader.py` allows direct loading and compilation of manifest projects (`archide.project.json`) and directories of JSON graphs for automated testing.
- **Extending Blocks**: If adding a new block, define its schema and logic in `backend/blocks/` and register in `backend/registry.py`.

## 3. Component & Source Directory Mapping
- **Activity Bar & Sidebar**: `src/components/ActivityBar.tsx`, `src/components/FileExplorer.tsx` (VFS), `src/components/BlockLibrary.tsx`.
- **Canvas & Code Workspace**: `src/components/DnDCanvas.tsx`, `src/components/CentralCodeEditor.tsx`, `src/components/CustomNode.tsx`, `src/components/TensorEdge.tsx`.
- **Inspector & Properties**: `src/components/RightPanel.tsx`, `src/components/PropertiesPanel.tsx`.
- **Header & API Payload**: `src/components/Header.tsx`.
- **Compiler & Testing**: `backend/compiler.py`, `backend/project_loader.py`, `backend/tests/`.

## 4. Documentation References
Refer to `docs/index.md` for deep technical references on the Block Registry, Compiler Engine, and Roadmaps. Always update the relevant documentation if you alter architecture logic.
