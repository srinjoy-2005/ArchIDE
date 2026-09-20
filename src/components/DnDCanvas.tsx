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

import React, { useCallback, useRef } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
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
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import CustomNode from './CustomNode';
import TensorEdge from './TensorEdge';
import { CentralCodeEditor } from './CentralCodeEditor';
import { useEditorStore } from '../lib/store';
import { Plus, X, FileCode, Network, Code2 } from 'lucide-react';
import { getId, initialNodes, initialEdges } from '../lib/constants';

// Defined at module level to avoid re-creating objects on every render,
// which would cause React Flow to unmount and remount all nodes.
const nodeTypes = { custom: CustomNode };
const edgeTypes = { tensor: TensorEdge };

// ─── FileTabBar ───────────────────────────────────────────────────────────────

function FileTabBar() {
  const {
    files,
    folders,
    openTabIds,
    activeFileId,
    activeViewMode,
    setActiveViewMode,
    switchFile,
    createFile,
    closeTab,
    updateFileState
  } = useEditorStore();
  const { getNodes, getEdges } = useReactFlow();

  const handleSwitch = (id: string) => {
    if (id === activeFileId) return;
    // Snapshot live canvas state before switching so edits aren't lost
    updateFileState(activeFileId, getNodes(), getEdges());
    switchFile(id);
  };

  const handleCreate = () => {
    updateFileState(activeFileId, getNodes(), getEdges());
    const name = prompt('Enter new file name (e.g. attention.json or layer.py):');
    if (name) createFile(name);
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
        <button
          onClick={handleCreate}
          title="New File"
          className="ml-1 p-1 rounded hover:bg-[#252525] text-[#666] hover:text-[#d4d4d4] transition-colors flex items-center justify-center"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Dual View Mode Switcher: Graph vs Python Code */}
      <div className="flex items-center bg-[#181818] p-0.5 rounded border border-[#2a2a2a] mx-2 flex-shrink-0">
        <button
          onClick={() => setActiveViewMode('graph')}
          className={`flex items-center gap-1 px-2.5 py-1 text-[11px] rounded transition-colors ${
            activeViewMode === 'graph'
              ? 'bg-[#2a2a2a] text-[#ffffff] font-medium shadow-sm'
              : 'text-[#777] hover:text-[#bbb]'
          }`}
          title="Visual Node Graph Canvas"
        >
          <Network className="w-3 h-3 text-[#38bdf8]" />
          <span>Graph</span>
        </button>
        <button
          onClick={() => setActiveViewMode('code')}
          className={`flex items-center gap-1 px-2.5 py-1 text-[11px] rounded transition-colors ${
            activeViewMode === 'code'
              ? 'bg-[#2a2a2a] text-[#ffffff] font-medium shadow-sm'
              : 'text-[#777] hover:text-[#bbb]'
          }`}
          title="Python Code Editor"
        >
          <Code2 className="w-3 h-3 text-[#eab308]" />
          <span>Python Code</span>
        </button>
      </div>
    </div>
  );
}

// ─── DnDCanvas ────────────────────────────────────────────────────────────────

export function DnDCanvas() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, setNodes, setEdges, getNode, getEdges } = useReactFlow();
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const activeFileId = useEditorStore((s) => s.activeFileId);
  const activeViewMode = useEditorStore((s) => s.activeViewMode);
  const files = useEditorStore((s) => s.files);
  const activeFile = files.find((f) => f.id === activeFileId);

  const isCodeMode = activeViewMode === 'code' || activeFile?.fileType === 'code' || activeFile?.name.endsWith('.py');

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

  // Clear shape error highlight whenever the graph is edited
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setShapeErrorNodeId(null);
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    [setNodes, setShapeErrorNodeId]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setShapeErrorNodeId(null);
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
    <div className="flex-1 relative flex flex-col h-full overflow-hidden" ref={canvasRef}>
      <FileTabBar />
      <div className="flex-1 relative overflow-hidden">
        {isCodeMode ? (
          <CentralCodeEditor />
        ) : (
          <ReactFlow
            key={activeFileId}
            defaultNodes={activeFile?.nodes || initialNodes}
            defaultEdges={activeFile?.edges || initialEdges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onConnect={onConnect}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodesDelete={onNodesDelete}
            isValidConnection={isValidConnection}
            onDrop={onDrop}
            onDragOver={onDragOver}
            deleteKeyCode={['Backspace', 'Delete']}
            fitView
          >
            <Controls className="!bg-[#252525] !border-[#3a3a3a] !rounded-[3px]" style={{ bottom: 16, left: 16 }} />
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
    </div>
  );
}
