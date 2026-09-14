"use client";

/**
 * src/components/DnDCanvas.tsx
 *
 * Contains two components:
 *
 * - FileTabBar: the multi-file tab strip above the canvas. Handles switching between
 *   graph files, creating new modules, and deleting files. Snapshots the live React Flow
 *   state into Zustand before switching so no edits are lost.
 *
 * - DnDCanvas: the main React Flow canvas. Runs in uncontrolled mode (defaultNodes /
 *   defaultEdges) per the architecture guardrails in .agents/project_context.md.
 *   Handles drag-and-drop node creation, edge validation (single-input port enforcement),
 *   node/edge change propagation, and orphan edge pruning on node delete.
 */

import React, { useCallback, useRef, useEffect } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  ControlButton,
  Background,
  BackgroundVariant,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type Edge,
  type Node,
  type NodeChange,
  type EdgeChange,
  SelectionMode,
  useReactFlow,
  useNodes,
  useEdges,
  ConnectionLineType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import CustomNode from './CustomNode';
import TensorEdge from './TensorEdge';
import { CentralCodeEditor } from './CentralCodeEditor';
import { QuickInsertModal } from './QuickInsertModal';
import { useEditorStore, useVFSStore } from '../lib/store';
import { X, FileCode, Network, Code2, Hand, MousePointer2, GitFork } from 'lucide-react';
import { getId, initialNodes, initialEdges, API_BASE } from '../lib/constants';
import { resolveFilePath } from '../lib/utils';

// Defined at module level to avoid re-creating objects on every render,
// which would cause React Flow to unmount and remount all nodes.
const nodeTypes = { custom: CustomNode };
const edgeTypes = {
  tensor: TensorEdge,
  default: TensorEdge,
  bezier: TensorEdge,
  step: TensorEdge,
  smoothstep: TensorEdge,
};

// ─── FileTabBar ───────────────────────────────────────────────────────────────

function FileTabBar() {
  const {
    files,
    folders,
    openTabIds,
    activeFileId,
    switchFile,
    closeTab,
    updateFileState
  } = useVFSStore();
  const { getNodes, getEdges } = useReactFlow();

  const handleSwitch = (id: string) => {
    if (id === activeFileId) return;
    // Snapshot live canvas state before switching so edits aren't lost
    updateFileState(activeFileId, getNodes(), getEdges());
    switchFile(id);
  };

  const handleClose = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    updateFileState(activeFileId, getNodes(), getEdges());
    closeTab(id);
  };

  // Helper to compute display path (e.g. "blocks/conv/res_block.json")
  const getDisplayPath = (file: typeof files[0]) => {
    const parent = folders.find((f) => f.id === file.parentId);
    return parent ? `${parent.name}/${file.name}` : file.name;
  };

  // Get the file objects for currently open tabs
  const openFiles = openTabIds
    .map((id) => files.find((f) => f.id === id))
    .filter((f): f is typeof files[0] => Boolean(f));

  return (
    <div className="flex items-center justify-between bg-[#121212] border-b border-[#282828] select-none h-9 px-1">
      {/* Tab list */}
      <div className="flex items-center gap-0 overflow-x-auto flex-1 h-full">
        {openFiles.map((f) => {
          const isActive = f.id === activeFileId;
          const displayPath = getDisplayPath(f);
          const isCode = f.fileType === 'code' || f.name.endsWith('.py');

          return (
            <div
              key={f.id}
              onClick={() => handleSwitch(f.id)}
              title={displayPath}
              className={`group relative flex items-center gap-2 px-3 py-1.5 h-full cursor-pointer text-[11.5px] transition-colors border-r border-[#222222] ${
                isActive
                  ? 'bg-[#1e1e1e] text-[#f0f0f0] font-medium border-t-2 border-t-[#2d8cf0]'
                  : 'bg-[#151515] text-[#7a7a7a] hover:bg-[#1a1a1a] hover:text-[#cccccc] border-t-2 border-t-transparent'
              }`}
            >
              {isCode ? (
                <FileCode className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-[#eab308]' : 'text-[#888]'}`} />
              ) : (
                <Network className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-[#38bdf8]' : 'text-[#555]'}`} />
              )}
              <span className="truncate max-w-[150px]">{displayPath}</span>
              <button
                onClick={(e) => handleClose(e, f.id)}
                title="Close tab"
                className="opacity-0 group-hover:opacity-100 hover:text-[#e54545] hover:bg-[#2e2e2e] p-0.5 rounded transition-all ml-0.5"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── DnDCanvas ────────────────────────────────────────────────────────────────

export function DnDCanvas() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const isCanvasHoveredRef = useRef(false);
  const lastMousePosRef = useRef<{ x: number; y: number } | null>(null);

  const { screenToFlowPosition, setNodes, setEdges, getNode, getNodes, getEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const clipboard = useEditorStore((s) => s.clipboard);
  const setClipboard = useEditorStore((s) => s.setClipboard);
  const canvasMode = useEditorStore((s) => s.canvasMode);
  const setCanvasMode = useEditorStore((s) => s.setCanvasMode);
  const edgeRouting = useEditorStore((s) => s.edgeRouting);
  const toggleEdgeRouting = useEditorStore((s) => s.toggleEdgeRouting);
  const sidebarOpen = useEditorStore((s) => s.sidebarOpen);
  const setSidebarOpen = useEditorStore((s) => s.setSidebarOpen);
  const activeSidebarView = useEditorStore((s) => s.activeSidebarView);
  const setActiveSidebarView = useEditorStore((s) => s.setActiveSidebarView);
  const toggleInspector = useEditorStore((s) => s.toggleInspector);
  const setQuickInsert = useEditorStore((s) => s.setQuickInsert);

  const handleToggleEdgeRouting = useCallback(() => {
    toggleEdgeRouting();
    setEdges((eds) => eds.map((e) => ({ ...e, type: 'tensor' })));
  }, [toggleEdgeRouting, setEdges]);

  const activeFileId = useVFSStore((s) => s.activeFileId);
  const files = useVFSStore((s) => s.files);
  const folders = useVFSStore((s) => s.folders);
  const isSaving = useVFSStore((s) => s.isSaving);
  const graphsFolderId = useVFSStore((s) => s.graphsFolderId);

  const activeFile = files.find((f) => f.id === activeFileId);
  const isCodeMode = activeFile?.fileType === 'code' || activeFile?.name.endsWith('.py') || activeFile?.name.endsWith('.toml');

  const initialCanvasEdges = React.useMemo(() => {
    return (activeFile?.edges || initialEdges).map((e) => ({
      ...e,
      type: 'tensor',
    }));
  }, [activeFile?.id, activeFile?.edges]);

  // Ref to track the last saved structural state to prevent infinite ping-pongs
  const lastSavedState = useRef<string>("");

  // Reset the saved state tracker when switching files so the next render triggers a save check
  useEffect(() => {
    lastSavedState.current = "";
  }, [activeFileId]);

  // Helper to strip transient React Flow state
  const getStrippedGraph = useCallback((nodesToStrip: Node[], edgesToStrip: Edge[]) => {
    return {
      nodes: nodesToStrip.map(n => ({ id: n.id, type: n.type, position: n.position, data: n.data })),
      edges: edgesToStrip.map(e => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle, type: e.type, animated: e.animated })),
    };
  }, []);

  // ─── Debounced Auto-Save (Live VFS Sync) ────────────────────────────────────
  useEffect(() => {
    if (!isSaving || !activeFile || isCodeMode) return;

    // 1. Prevent saves while actively dragging nodes
    if (nodes.some(n => n.dragging)) return;

    // Ensure we don't save an empty graph if it hasn't hydrated properly
    if (nodes.length === 0 && edges.length === 0 && activeFile.nodes.length > 0) return;

    // 2. Strip transient UI state to isolate structural/semantic data
    const strippedGraph = getStrippedGraph(nodes, edges);
    const currentStateStr = JSON.stringify({ ...strippedGraph, variables: activeFile.variables || [] });

    // 3. Deep equality check against the last saved state
    if (currentStateStr === lastSavedState.current) return;

    // Use a timeout to debounce saves after canvas interactions
    const handler = setTimeout(() => {
      // Use resolveFilePath to get the canonical file_id (handles folder renames correctly)
      const fullFileId = resolveFilePath(activeFile, folders, graphsFolderId);
      const fileNameWithoutExt = activeFile.name.replace(/\.[^/.]+$/, '');

      const payload = {
        file_id: fullFileId,
        content: {
          name: fileNameWithoutExt,
          nodes: strippedGraph.nodes,
          edges: strippedGraph.edges,
          variables: activeFile.variables || []
        }
      };

      // Mark as saved before fetching to immediately block stale re-saves
      lastSavedState.current = currentStateStr;

      fetch(`${API_BASE}/api/vfs/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(err => console.error('Auto-save failed:', err));
    }, 500);

    return () => clearTimeout(handler);
  }, [nodes, edges, activeFile, folders, isSaving, isCodeMode, getStrippedGraph, graphsFolderId]);

  // (SSE disk-to-canvas sync removed — no longer needed without Mirror Local)

  // ─── Canvas Shortcuts & Clipboard ──────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 1. Ignore if typing in an input, textarea, select, or editable element
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement ||
        (e.target as HTMLElement)?.isContentEditable
      ) {
        return;
      }

      // 2. Only fire shortcuts when canvas is active/selected (focused or mouse hovered)
      const isCanvasActive =
        isCanvasHoveredRef.current ||
        (canvasRef.current && canvasRef.current.contains(document.activeElement));
      if (!isCanvasActive) return;

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdKey = isMac ? e.metaKey : e.ctrlKey;

      // Tab: Quick search & insert layer at cursor
      if (e.key === 'Tab') {
        e.preventDefault();
        const pos = lastMousePosRef.current || { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        setQuickInsert({ isOpen: true, clientPos: pos });
        return;
      }

      // V: Toggle Variables panel
      if (!cmdKey && (e.key === 'v' || e.key === 'V')) {
        e.preventDefault();
        if (sidebarOpen && activeSidebarView === 'variables') {
          setSidebarOpen(false);
        } else {
          setActiveSidebarView('variables');
        }
        return;
      }

      // E: Toggle File Explorer
      if (!cmdKey && (e.key === 'e' || e.key === 'E')) {
        e.preventDefault();
        if (sidebarOpen && activeSidebarView === 'explorer') {
          setSidebarOpen(false);
        } else {
          setActiveSidebarView('explorer');
        }
        return;
      }

      // I: Toggle Inspector
      if (!cmdKey && (e.key === 'i' || e.key === 'I')) {
        e.preventDefault();
        toggleInspector();
        return;
      }

      // Cmd+C / Ctrl+C: Copy selected nodes
      if (cmdKey && (e.key === 'c' || e.key === 'C')) {
        const selectedNodes = getNodes().filter((n: Node) => n.selected);
        if (selectedNodes.length === 0) return;
        
        const selectedIds = new Set(selectedNodes.map((n: Node) => n.id));
        const innerEdges = getEdges().filter((edge: Edge) => selectedIds.has(edge.source) && selectedIds.has(edge.target));
        
        setClipboard({ nodes: selectedNodes, edges: innerEdges });
      }

      // Cmd+V / Ctrl+V: Paste copied nodes
      if (cmdKey && (e.key === 'v' || e.key === 'V')) {
        if (!clipboard || clipboard.nodes.length === 0) return;

        // Create ID mapping from old to new
        const idMap: Record<string, string> = {};
        clipboard.nodes.forEach((n: Node) => { idMap[n.id] = getId(); });

        const pastedNodes = clipboard.nodes.map((n: Node) => ({
          ...n,
          id: idMap[n.id],
          selected: true,
          position: { x: n.position.x + 40, y: n.position.y + 40 }
        }));

        const pastedEdges = clipboard.edges.map((e: Edge) => ({
          ...e,
          id: getId(),
          source: idMap[e.source],
          target: idMap[e.target]
        }));

        // Deselect current nodes
        setNodes((nds: Node[]) => nds.map((n) => ({ ...n, selected: false } as Node)).concat(pastedNodes as Node[]));
        setEdges((eds: Edge[]) => eds.map((e) => ({ ...e, selected: false } as Edge)).concat(pastedEdges as Edge[]));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    getNodes,
    getEdges,
    setNodes,
    setEdges,
    clipboard,
    setClipboard,
    sidebarOpen,
    activeSidebarView,
    setSidebarOpen,
    setActiveSidebarView,
    toggleInspector,
    setQuickInsert
  ]);

  // Prevent connecting a single-input port that already has an incoming edge
  const isValidConnection = useCallback(
    (connection: Connection | Edge) => {
      if (connection.source === connection.target) return false;
      const targetNode = getNode(connection.target);
      if (!targetNode) return true;
      const inputs = (targetNode.data?.inputs as any[]) || [];
      const targetPort = inputs.find((p) => p.id === connection.targetHandle);
      if (targetPort && !targetPort.is_list) {
        const existing = getEdges().find(
          (e) => e.target === connection.target && e.targetHandle === connection.targetHandle
        );
        if (existing && existing.id !== (connection as any).id) return false;
      }
      return true;
    },
    [getNode, getEdges]
  );

  const onConnect = useCallback(
    (params: Connection | Edge) =>
      setEdges((eds: Edge[]) => addEdge({ ...params, animated: false, type: 'tensor' } as Edge, eds)),
    [setEdges]
  );

  // Clear shape error highlight whenever the graph is structurally edited
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const isStructuralEdit = changes.some(c => c.type === 'remove' || c.type === 'add');
      if (isStructuralEdit) {
        setShapeErrorNodeId(null);
      }
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    [setNodes, setShapeErrorNodeId]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const isStructuralEdit = changes.some(c => c.type === 'remove' || c.type === 'add');
      if (isStructuralEdit) {
        setShapeErrorNodeId(null);
      }
      setEdges((eds) => applyEdgeChanges(changes, eds));
    },
    [setEdges, setShapeErrorNodeId]
  );

  // Prune dangling edges when nodes are deleted via keyboard
  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      const deletedIds = new Set(deleted.map((n) => n.id));
      setEdges((eds) => eds.filter((e) => !deletedIds.has(e.source) && !deletedIds.has(e.target)));
    },
    [setEdges]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Creates a new node on drop, initialising paramValues from block defaults
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const type = event.dataTransfer.getData('application/reactflow');
      const label = event.dataTransfer.getData('application/label');
      const blockDefStr = event.dataTransfer.getData('application/blockDef');
      if (!type) return;

      const blockDef = blockDefStr ? JSON.parse(blockDefStr) : {};
      const initialParamValues: any = {};
      if (blockDef.params) {
        blockDef.params.forEach((p: any) => { initialParamValues[p.name] = p.default; });
      }

      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });

      const newNode = {
        id: getId(),
        type,
        position,
        data: {
          block_id: blockDef.id,
          label,
          description: `A ${label} layer`,
          params: blockDef.params || [],
          paramValues: initialParamValues,
          inputs: blockDef.inputs || [],
          outputs: blockDef.outputs || [],
          is_functional: blockDef.is_functional || false,
          varName: '',
          custom_module_id: blockDef.custom_module_id,
        },
      };

      setNodes((nds: Node[]) => nds.concat(newNode as unknown as Node));
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div
      className="flex-1 relative flex flex-col h-full overflow-hidden outline-none"
      ref={canvasRef}
      tabIndex={0}
      onMouseEnter={() => { isCanvasHoveredRef.current = true; }}
      onMouseLeave={() => { isCanvasHoveredRef.current = false; }}
      onMouseMove={(e) => {
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      }}
    >
      <FileTabBar />
      <div className="flex-1 relative overflow-hidden">
        {isCodeMode ? (
          <CentralCodeEditor />
        ) : (
          <ReactFlow
            key={activeFileId}
            defaultNodes={activeFile?.nodes || initialNodes}
            defaultEdges={initialCanvasEdges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultEdgeOptions={{ type: 'tensor' }}
            connectionLineType={edgeRouting === 'step' ? ConnectionLineType.SmoothStep : ConnectionLineType.Bezier}
            onConnect={onConnect}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodesDelete={onNodesDelete}
            isValidConnection={isValidConnection}
            onDrop={onDrop}
            onDragOver={onDragOver}
            deleteKeyCode={['Backspace', 'Delete']}
            panOnDrag={canvasMode === 'pan'}
            selectionOnDrag={canvasMode === 'select'}
            selectionMode={SelectionMode.Partial}
            fitView
          >
            <Controls className="!bg-[#252525] !border-[#3a3a3a] !rounded-[3px]" style={{ bottom: 16, left: 16 }}>
              <ControlButton
                onClick={() => setCanvasMode('pan')}
                title="Pan Tool (Hand)"
                className="hover:!bg-[#333] transition-colors"
                style={{ backgroundColor: canvasMode === 'pan' ? '#333' : 'transparent', color: canvasMode === 'pan' ? '#38bdf8' : '#777' }}
              >
                <Hand className="w-3.5 h-3.5" />
              </ControlButton>
              <ControlButton
                onClick={() => setCanvasMode('select')}
                title="Select Tool (Window)"
                className="hover:!bg-[#333] transition-colors"
                style={{ backgroundColor: canvasMode === 'select' ? '#333' : 'transparent', color: canvasMode === 'select' ? '#38bdf8' : '#777' }}
              >
                <MousePointer2 className="w-3.5 h-3.5" />
              </ControlButton>
              <ControlButton
                onClick={handleToggleEdgeRouting}
                title={`Edge Routing: ${edgeRouting === 'step' ? 'Stepped / Orthogonal (SimulIDE)' : 'Smooth Bezier'} (Click to switch)`}
                className="hover:!bg-[#333] transition-colors"
                style={{ backgroundColor: edgeRouting === 'step' ? '#333' : 'transparent', color: edgeRouting === 'step' ? '#38bdf8' : '#777' }}
              >
                <GitFork className="w-3.5 h-3.5" />
              </ControlButton>
            </Controls>
            <MiniMap
              nodeColor={() => '#2d8cf0'}
              maskColor="rgba(18,18,18,0.8)"
              className="!bg-[#1e1e1e] !border !border-[#3a3a3a] !rounded-[3px]"
              style={{ bottom: 16, right: 16 }}
            />
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#333" />
          </ReactFlow>
        )}
      </div>
      <QuickInsertModal />
    </div>
  );
}
