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
import { useEditorStore } from '../lib/store';
import { Plus, X } from 'lucide-react';
import { getId, initialNodes, initialEdges } from '../lib/constants';

// Defined at module level to avoid re-creating objects on every render,
// which would cause React Flow to unmount and remount all nodes.
const nodeTypes = { custom: CustomNode };
const edgeTypes = { tensor: TensorEdge };

// ─── FileTabBar ───────────────────────────────────────────────────────────────

function FileTabBar() {
  const { files, activeFileId, switchFile, createFile, deleteFile, updateFileState } = useEditorStore();
  const { getNodes, getEdges } = useReactFlow();

  const handleSwitch = (id: string) => {
    if (id === activeFileId) return;
    // Snapshot live canvas state before switching so edits aren't lost
    updateFileState(activeFileId, getNodes(), getEdges());
    switchFile(id);
  };

  const handleCreate = () => {
    updateFileState(activeFileId, getNodes(), getEdges());
    const name = prompt('Enter new module name:');
    if (name) createFile(name);
  };

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this module?')) {
      deleteFile(id);
    }
  };

  return (
    <div className="flex items-center gap-1 px-2 pt-2 pb-1 bg-[#181818] border-b border-[#363636] overflow-x-auto flex-shrink-0">
      {files.map((f) => (
        <div
          key={f.id}
          onClick={() => handleSwitch(f.id)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-t-[4px] cursor-pointer text-[11px] transition-colors border border-b-0 ${
            f.id === activeFileId
              ? 'bg-[#1e1e1e] border-[#363636] text-[#e2e2e2]'
              : 'bg-[#252525] border-transparent text-[#888] hover:bg-[#2a2a2a] hover:text-[#d4d4d4]'
          }`}
        >
          <span>{f.name}</span>
          {files.length > 1 && (
            <button onClick={(e) => handleDelete(e, f.id)} className="text-[#555] hover:text-[#e54545]">
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      ))}
      <button
        onClick={handleCreate}
        className="ml-1 p-1 rounded hover:bg-[#2a2a2a] text-[#888] hover:text-[#d4d4d4] transition-colors border border-transparent"
      >
        <Plus className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ─── DnDCanvas ────────────────────────────────────────────────────────────────

export function DnDCanvas() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, setNodes, setEdges, getNode, getEdges } = useReactFlow();
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const activeFileId = useEditorStore((s) => s.activeFileId);
  const files = useEditorStore((s) => s.files);
  const activeFile = files.find((f) => f.id === activeFileId);

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
    <div className="flex-1 relative flex flex-col" ref={canvasRef}>
      <FileTabBar />
      <div className="flex-1 relative">
        {/* key=activeFileId forces a full remount when switching files,
            applying the new defaultNodes/defaultEdges for that tab */}
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
      </div>
    </div>
  );
}
