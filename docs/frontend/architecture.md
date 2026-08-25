# ArchIDE: Frontend Architecture & Component Specification

This document provides the formal architecture and technical guidelines for the ArchIDE visual DAG builder frontend.

> [!NOTE]
> **Source Files** (modular structure as of Aug 25, 2026):
> - Layout Shell: [`src/app/page.tsx`](../../src/app/page.tsx)
> - Canvas + File Tabs: [`src/components/DnDCanvas.tsx`](../../src/components/DnDCanvas.tsx)
> - Custom Node: [`src/components/CustomNode.tsx`](../../src/components/CustomNode.tsx)
> - Tensor Edge: [`src/components/TensorEdge.tsx`](../../src/components/TensorEdge.tsx)
> - Properties Panel: [`src/components/PropertiesPanel.tsx`](../../src/components/PropertiesPanel.tsx)
> - Block Library Sidebar: [`src/components/BlockLibrary.tsx`](../../src/components/BlockLibrary.tsx)
> - Header + API Logic: [`src/components/Header.tsx`](../../src/components/Header.tsx)
> - Right Panel Shell: [`src/components/RightPanel.tsx`](../../src/components/RightPanel.tsx)
> - Code Viewer: [`src/components/CodePanel.tsx`](../../src/components/CodePanel.tsx)
> - Doc Popups: [`src/components/DocPanels.tsx`](../../src/components/DocPanels.tsx)
> - Shared Constants: [`src/lib/constants.ts`](../../src/lib/constants.ts)
> - Global Editor Store: [`src/lib/store.ts`](../../src/lib/store.ts)

---

## 1. Technical Stack & State Model

ArchIDE's frontend is built with **Next.js (App Router)**, **React Flow (`@xyflow/react`)**, **Zustand**, and **TailwindCSS**.

```mermaid
flowchart LR
    Palette[Sidebar Palette] -->|Drag & Drop| Canvas[DnDCanvas @xyflow/react]
    Canvas -->|Select Node| Inspector[PropertiesPanel]
    Canvas -->|Edge / Node Changes| Store[Zustand Store]
    Canvas -->|Static Check| API_Check[POST /api/check]
    Canvas -->|Compile Request| API_Compile[POST /api/compile]
    API_Check -->|Shapes & Errors| Store
    API_Compile -->|Synthesized PyTorch| CodeDrawer[Code Export Drawer]
```

### ⚠️ Critical Guardrail: Uncontrolled Canvas State
- **Uncontrolled React Flow is Required**: In [`src/components/DnDCanvas.tsx`](../../src/components/DnDCanvas.tsx), `ReactFlow` must strictly use `defaultNodes` and `defaultEdges` rather than controlled `nodes={nodes}` / `edges={edges}` props.
- **Why?**: Controlled state causes synchronization loops and drag latency with React Flow's internal spatial index.
- **State Manipulation**: To modify nodes from outside the canvas (e.g. from `PropertiesPanel`), use `const { setNodes } = useReactFlow()` and mutate node data immutably.
- **Reading Selected Nodes**: Selected state is read directly from the internal store via `useNodes().find(n => n.selected)` rather than tracking local selection state.

---

> [!NOTE]
> **Modular Architecture (Aug 25, 2026)**: The original monolithic `page.tsx` (1255 lines) was refactored into 8 focused component files plus `src/lib/constants.ts`. `page.tsx` is now a 43-line layout shell.

### 1. `DnDCanvas` + `FileTabBar` ([`src/components/DnDCanvas.tsx`](../../src/components/DnDCanvas.tsx))
- Renders the interactive grid canvas with `MiniMap`, `Controls`, and dot grid `Background`.
- `nodeTypes` and `edgeTypes` are defined at module level to prevent React Flow from unmounting nodes on re-render.
- Handles `onDrop`: Deserializes block definition JSON from `dataTransfer`, sets initial parameter defaults, computes canvas coordinates via `screenToFlowPosition`, and appends a `custom` node.
- Handles `onConnect`: Adds custom `tensor` edges between handles.
- Automatically clears `shapeErrorNodeId` when nodes or edges change.
- `FileTabBar` snapshots the active file's live React Flow state into Zustand via `updateFileState()` before switching tabs; `key={activeFileId}` forces a full remount with `defaultNodes`/`defaultEdges` for the new file.

### 2. `CustomNode` ([`src/components/CustomNode.tsx`](../../src/components/CustomNode.tsx))
- **Category Accent Strip**: Color-coded vertical indicator based on layer type.
- **Dynamic Handles**: Renders `Position.Left` target handles for `inputs` and `Position.Right` source handles for `outputs`. Hovering over handles displays the handle name and propagated tensor shape.
- **Error Badging**: If `shapeErrorNodeId === id`, highlights node with red border and renders an `AlertTriangle` "Shape mismatch" chip.
- **Variable Identifier Display**: Shows `→ {effectiveVar}` (e.g., custom name or auto-generated PyTorch variable name).
- **Hover Toolbar**: Quick duplicate and delete actions.
- **Shape Tooltip**: On hover, displays an elevated card listing input and output tensor dimensions (e.g. `[1×16×112×112]`).

### 3. `TensorEdge` ([`src/components/TensorEdge.tsx`](../../src/components/TensorEdge.tsx))
- Renders a smooth bezier curve with an invisible 16px hover hit area.
- Displays an `EdgeLabelRenderer` midpoint chip showing the active tensor shape flowing through the edge (retrieved from `useEditorStore((s) => s.nodeShapes)`).

### 4. `PropertiesPanel` ([`src/components/PropertiesPanel.tsx`](../../src/components/PropertiesPanel.tsx))
- **Node Identifier & VarName**: Allows editing node label and custom output variable name.
- **Three-Section Property Inspector**:
  1. *Tensor Shapes*: Displays read-only inferred port dimensions.
  2. *Hyperparameters*: Editable inputs (`int`, `float`, `string`, `bool`) for primary layer properties.
  3. *Advanced Properties*: Collapsible section for secondary settings (e.g. dilation, momentum, eps).
- **Dynamic Port Resizing**: When modifying `num_inputs` (on `AddBlock`) or `chunks` (on `SplitBlock`), dynamically updates node `inputs`/`outputs` arrays so the canvas immediately adds/removes connection handles.
- Falls back to `ModelSummaryDashboard` (graph stats + quick-start ConvNet loader) when no node is selected.

### 5. `Header` ([`src/components/Header.tsx`](../../src/components/Header.tsx))
- Houses all backend API logic: `buildPayload()` (constructs multi-graph JSON), `handleExport()` (`POST /api/compile`), `handleCheck()` (`POST /api/check`), `handleReset()`.
- Back-fills inferred shapes and auto-resolved params onto React Flow nodes after a successful `/api/check` response.
- Broadcasts request/response payloads via `BroadcastChannel` for the `/dev/payloads` inspector.

### 6. `BlockLibrary` ([`src/components/BlockLibrary.tsx`](../../src/components/BlockLibrary.tsx))
- Fetches live registry from `GET /api/blocks` on mount; falls back to `FALLBACK_BLOCKS` from `src/lib/constants.ts`.
- Derives **Custom Module** blocks on the fly from non-active open files in the Zustand store.
- Right-clicking a block (`BlockItem`) fetches `GET /api/blocks/{id}/docs` and opens `DocContextMenu`.

### 7. `DocContextMenu` & `DocDetailsPanel` ([`src/components/DocPanels.tsx`](../../src/components/DocPanels.tsx))
- Right-clicking any block in the sidebar fetches documentation from `GET /api/blocks/{id}/docs` and presents a popup summary or full slide-out documentation panel.

### 8. `RightPanel` ([`src/components/RightPanel.tsx`](../../src/components/RightPanel.tsx))
- Owns `rightTab` and `rightOpen` local state. Switches between `<PropertiesPanel />` and `<CodePanel />`.

### 9. `CodePanel` ([`src/components/CodePanel.tsx`](../../src/components/CodePanel.tsx))
- Reads `generatedCode` from Zustand store; provides copy-to-clipboard and download-as-`model.py` actions.

---

## 3. Global State: Zustand Store (`src/lib/store.ts`)

| State Field | Type | Description |
|---|---|---|
| `files` | `GraphFile[]` | List of all open files/graphs in the project |
| `activeFileId` | `string` | ID of the currently active tab/file |
| `generatedCode` | `string` | Current PyTorch module code string |
| `shapeErrorNodeId` | `string \| null` | Node ID with an active shape mismatch (highlights node red) |
| `nodeShapes` | `Record<string, Record<string, number[]>>` | Maps `node_id -> { port_id: shape_array }` returned by `/api/check` |
| `docMenuInfo` | `object \| null` | Coordinates and content for the right-click block documentation tooltip |
| `docPanelInfo` | `object \| null` | State for the slide-out full block documentation drawer |

---

## 4. Backend Communication Pipeline

1. **Palette Loading**: `GET /api/blocks` initializes the sidebar palette. The frontend also scans all `files` (except the active one) to generate "Custom Modules" on the fly based on their `Input` and `Output` nodes.
2. **Static Shape Check ("Run Static Tensor Check")**:
   - Sends recursive multi-graph payload `{ main_graph_id, graphs }` to `POST /api/check`.
   - On success (`200 OK`): Updates `nodeShapes` in store; shape chips appear on edges and nodes.
   - On error (`422 Unprocessable Content`): Reads `detail.node_id`, sets `shapeErrorNodeId`, and displays toast with error details.
3. **PyTorch Export ("Export PyTorch")**:
   - Sends `{ main_graph_id, graphs }` to `POST /api/compile`. Node `data` for custom modules includes `custom_module_id`.
   - On success: Stores compiled code in `generatedCode` and opens code viewer modal.
