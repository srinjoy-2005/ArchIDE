# ArchIDE - Project Context for Agents

This document provides critical context for agents operating within the ArchIDE repository. Read these guidelines before attempting any refactoring or feature additions.

## 1. Architecture Overview
ArchIDE is a full-stack node-based PyTorch architecture builder. 
- **Frontend**: Next.js (React), React Flow (`@xyflow/react`), TailwindCSS, Zustand.
- **Backend**: Python (FastAPI), PyTorch generation compiler logic.
- **Communication**: Frontend sends JSON payloads describing nodes and edges to the `/api/compile` endpoint, and fetches available block definitions from `/api/blocks`.

## 2. Technical Guidelines & Guardrails
### Frontend (React Flow)
- **UNCONTROLLED MODE IS REQUIRED**: The `ReactFlow` component in `src/app/page.tsx` (`DnDCanvas`) is strictly designed to be **uncontrolled** (using `defaultNodes` and `defaultEdges` instead of `nodes={nodes}`). This allows internal React Flow state to be the single source of truth.
- **State Manipulation**: Do NOT use `useNodesState` or `useEdgesState`. To manipulate nodes from outside the canvas (like in `PropertiesPanel`), use `useReactFlow().setNodes()` or simply mutate the node parameters.
- **Selection**: We rely on reading the internal store (e.g. `const selectedNode = useNodes().find(n => n.selected)`) rather than `useOnSelectionChange` local state tracking to ensure UI always stays perfectly synced with the canvas.

### Backend (FastAPI & Compiler)
- **Topological Sorting**: `backend/compiler.py` uses Kahn's algorithm to resolve dependencies. Node cyclic dependencies will throw a `ValueError` which gets propagated back to the frontend as a `400 Bad Request`.
- **String-Based Code Generation**: PyTorch code is generated structurally. Stateful layers (like `nn.Linear` or `nn.Conv2d`) are appended to `__init__`, while functional operations (like `Add` or `Split`) go straight into `forward()`.
- **Extending Blocks**: If adding a new block, define it in `backend/registry.py` under the `REGISTRY` list using the `BlockDef` Pydantic model (`backend/models.py`). 

## 3. What to Look Into & When
- **UI/UX Changes**: Look in `src/app/page.tsx` (the main layout, properties panel, and canvas) or `src/components/CustomNode.tsx`.
- **Compiler/Export Bugs**: Look in `backend/compiler.py`.
- **Adding New Nodes/Layers**: Look in `backend/registry.py` and `backend/compiler.py` (to define how it parses into PyTorch string blocks).
- **TypeScript Type Issues**: Refer to `backend/models.py` to see the Python Pydantic schemas that we mirror in the frontend.

## 4. Documentation References
Refer to the `docs/index.md` file for deep technical references on the Block Registry, Compiler Engine, and Roadmaps. Always update the relevant documentation if you alter architecture logic.
