"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  Connection,
  Edge,
  Node,
  NodeChange,
  EdgeChange,
  ReactFlowProvider,
  useReactFlow,
  useNodes,
  useEdges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import CustomNode from "../components/CustomNode";
import TensorEdge from "../components/TensorEdge";
import { useEditorStore } from "../lib/store";
import {
  Search,
  Copy,
  Check,
  Download,
  Play,
  RotateCcw,
  ChevronRight,
  Plus,
  PanelRightClose,
  PanelRightOpen,
  Layers,
  Code2,
  Settings2,
  AlertCircle,
  Box,
  Sparkles,
  ArrowRightLeft,
  Brain,
  Activity,
} from "lucide-react";

// ─── Constants ─────────────────────────────────────────────────────────────────

const nodeTypes = { custom: CustomNode };
const edgeTypes = { tensor: TensorEdge };
const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];
let nodeCounter = 100;
const getId = () => `node_${nodeCounter++}`;

// Backend base URL — srinjoy_branch runs the API on port 8001
const API_BASE = "http://localhost:8001";

// ─── Fallback blocks (used when backend is unreachable) ────────────────────────

const FALLBACK_BLOCKS: any[] = [
  { id: "input",   name: "Input",   category: "Core Layers", is_functional: true,  color: "#4ade80", inputs: [],                                             outputs: [{ id: "out", name: "Output" }],                     params: [{ name: "shape", type: "string", default: "(1,3,224,224)" }] },
  { id: "output",  name: "Output",  category: "Core Layers", is_functional: true,  color: "#f87171", inputs: [{ id: "in", name: "Return Value" }],            outputs: [],                                                  params: [] },
  { id: "linear",  name: "Linear",  category: "Core Layers", is_functional: false, color: "#818cf8", inputs: [{ id: "in", name: "Input" }],                   outputs: [{ id: "out", name: "Output" }],                     params: [{ name: "in_features", type: "int", default: 128 }, { name: "out_features", type: "int", default: 64 }] },
  { id: "conv2d",  name: "Conv2D",  category: "Core Layers", is_functional: false, color: "#60a5fa", inputs: [{ id: "in", name: "Input" }],                   outputs: [{ id: "out", name: "Output" }],                     params: [{ name: "in_channels", type: "int", default: 3 }, { name: "out_channels", type: "int", default: 16 }, { name: "kernel_size", type: "int", default: 3 }] },
  { id: "relu",    name: "ReLU",    category: "Activations", is_functional: false, color: "#a78bfa", inputs: [{ id: "in", name: "Input" }],                   outputs: [{ id: "out", name: "Output" }],                     params: [] },
  { id: "softmax", name: "Softmax", category: "Activations", is_functional: false, color: "#c084fc", inputs: [{ id: "in", name: "Input" }],                   outputs: [{ id: "out", name: "Output" }],                     params: [{ name: "dim", type: "int", default: 1 }] },
  { id: "add",     name: "Add",     category: "Tensor Ops",  is_functional: true,  color: "#fb923c", inputs: [{ id: "in_0", name: "Input 1" }, { id: "in_1", name: "Input 2" }], outputs: [{ id: "out", name: "Out" }], params: [{ name: "num_inputs", type: "int", default: 2 }] },
  { id: "split",   name: "Split",   category: "Tensor Ops",  is_functional: true,  color: "#34d399", inputs: [{ id: "in", name: "Input" }],                   outputs: [{ id: "out_0", name: "Chunk 1" }, { id: "out_1", name: "Chunk 2" }], params: [{ name: "chunks", type: "int", default: 2 }, { name: "dim", type: "int", default: 0 }] },
];

// ─── Category icon helper ────────────────────────────────────────────────────────

function CategoryIcon({ category }: { category: string }) {
  const cls = "w-3 h-3 flex-shrink-0";
  const c = category.toLowerCase();
  if (c.includes("core"))       return <Box className={cls} style={{ color: "#60a5fa" }} />;
  if (c.includes("activation")) return <Sparkles className={cls} style={{ color: "#a78bfa" }} />;
  if (c.includes("tensor"))     return <ArrowRightLeft className={cls} style={{ color: "#fb923c" }} />;
  if (c.includes("pool"))       return <Activity className={cls} style={{ color: "#34d399" }} />;
  return <Brain className={cls} style={{ color: "#6b7280" }} />;
}

// ─── Model Summary Dashboard (shown in inspector when no node is selected) ───────

function ModelSummaryDashboard() {
  const { setNodes, setEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();

  const loadStarter = () => {
    const n: Node[] = [
      { id: "s1", type: "custom", position: { x: 60,  y: 160 }, data: { block_id: "input",  label: "Input",  is_functional: true,  params: [{ name: "shape", type: "string", default: "(1,3,224,224)" }], paramValues: { shape: "(1,3,224,224)" }, inputs: [], outputs: [{ id: "out", name: "Output" }] } },
      { id: "s2", type: "custom", position: { x: 270, y: 160 }, data: { block_id: "conv2d", label: "Conv2D", is_functional: false, params: [{ name: "in_channels", type: "int", default: 3 }, { name: "out_channels", type: "int", default: 16 }, { name: "kernel_size", type: "int", default: 3 }], paramValues: { in_channels: 3, out_channels: 16, kernel_size: 3 }, inputs: [{ id: "in", name: "Input" }], outputs: [{ id: "out", name: "Output" }] } },
      { id: "s3", type: "custom", position: { x: 480, y: 160 }, data: { block_id: "relu",   label: "ReLU",   is_functional: false, params: [], paramValues: {}, inputs: [{ id: "in", name: "Input" }], outputs: [{ id: "out", name: "Output" }] } },
      { id: "s4", type: "custom", position: { x: 680, y: 160 }, data: { block_id: "linear", label: "Linear", is_functional: false, params: [{ name: "in_features", type: "int", default: 16 }, { name: "out_features", type: "int", default: 10 }], paramValues: { in_features: 16, out_features: 10 }, inputs: [{ id: "in", name: "Input" }], outputs: [{ id: "out", name: "Output" }] } },
      { id: "s5", type: "custom", position: { x: 880, y: 160 }, data: { block_id: "output", label: "Output", is_functional: true,  params: [], paramValues: {}, inputs: [{ id: "in", name: "Return Value" }], outputs: [] } },
    ];
    const e: Edge[] = [
      { id: "e12", source: "s1", sourceHandle: "out", target: "s2", targetHandle: "in", type: "tensor" },
      { id: "e23", source: "s2", sourceHandle: "out", target: "s3", targetHandle: "in", type: "tensor" },
      { id: "e34", source: "s3", sourceHandle: "out", target: "s4", targetHandle: "in", type: "tensor" },
      { id: "e45", source: "s4", sourceHandle: "out", target: "s5", targetHandle: "in", type: "tensor" },
    ];
    setNodes(n);
    setEdges(e);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Stats row */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] p-3">
          <div className="text-[9px] uppercase tracking-wider text-[#666] mb-1">Layers</div>
          <div className="text-xl font-semibold text-[#e2e2e2] font-mono">{nodes.length}</div>
        </div>
        <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] p-3">
          <div className="text-[9px] uppercase tracking-wider text-[#666] mb-1">Connections</div>
          <div className="text-xl font-semibold text-[#e2e2e2] font-mono">{edges.length}</div>
        </div>
      </div>

      {/* Guide */}
      <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] p-3 text-[11px] text-[#888] space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-[#555] mb-2">Canvas Guide</div>
        <div className="flex justify-between"><span>Drag layer</span><span className="text-[#aaa] font-mono">→ drop on grid</span></div>
        <div className="flex justify-between"><span>Connect ports</span><span className="text-[#aaa] font-mono">→ drag handle</span></div>
        <div className="flex justify-between"><span>Delete</span><span className="text-[#aaa] font-mono">Backspace / Del</span></div>
        <div className="flex justify-between"><span>Select node</span><span className="text-[#aaa] font-mono">→ click</span></div>
      </div>

      {/* Quick load */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#555] mb-2">Quick Start</div>
        <button
          onClick={loadStarter}
          className="w-full flex items-center justify-between bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#3a3a3a] hover:border-[#505050] rounded-[3px] px-3 py-2 text-[11px] text-[#aaa] hover:text-[#e2e2e2] transition-colors group"
        >
          <div className="flex items-center gap-2">
            <Brain className="w-3.5 h-3.5 text-[#2d8cf0]" />
            <span>Load ConvNet Pipeline</span>
          </div>
          <ChevronRight className="w-3 h-3 text-[#555] group-hover:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  );
}

// ─── Properties / Inspector panel ────────────────────────────────────────────────

// ParamInput: Adobe-themed wrapper around srinjoy's ParamInput logic
function ParamInput({ param, value, onChange }: { param: any; value: any; onChange: (name: string, value: any) => void }) {
  const inputClass = param.read_only
    ? "w-full bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-2 py-1.5 text-[12px] text-[#555] font-mono cursor-not-allowed"
    : "w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] px-2 py-1.5 text-[12px] text-[#e2e2e2] font-mono outline-none transition-colors";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-[#aaa] capitalize flex items-center gap-1.5">
          {param.name.replace(/_/g, ' ')}
          {param.read_only && (
            <span className="text-[9px] font-mono text-[#555] bg-[#252525] border border-[#363636] px-1 py-px rounded-sm">inferred</span>
          )}
        </label>
        <span className="text-[9px] font-mono text-[#555]">{param.type}</span>
      </div>
      <input
        type={param.type === 'int' || param.type === 'float' ? 'number' : 'text'}
        className={inputClass}
        value={value ?? param.default}
        readOnly={param.read_only}
        disabled={param.read_only}
        title={param.description || ''}
        onChange={(e) => {
          if (!param.read_only) {
            onChange(param.name, param.type === 'int' ? parseInt(e.target.value) : param.type === 'float' ? parseFloat(e.target.value) : e.target.value);
          }
        }}
      />
    </div>
  );
}

// PropertiesPanel: srinjoy's logic (section-based params, num_inputs/chunks dynamic handles)
// with our Adobe-style UI skin and the extra varName field
function PropertiesPanel() {
  const { setNodes } = useReactFlow();
  const nodes = useNodes();
  const selectedNode = nodes.find((n) => n.selected) || null;

  // srinjoy's exact handleParamChange logic (incl. dynamic input/output resizing)
  const handleParamChange = (paramName: string, value: any) => {
    if (!selectedNode) return;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === selectedNode.id) {
          const newData: any = {
            ...n.data,
            paramValues: { ...(n.data.paramValues as any || {}), [paramName]: value },
          };
          if (paramName === 'num_inputs') {
            const num = parseInt(value) || 2;
            newData.inputs = Array.from({ length: num }, (_, i) => ({ id: `in_${i}`, name: `Input ${i + 1}` }));
          } else if (paramName === 'chunks') {
            const num = parseInt(value) || 2;
            newData.outputs = Array.from({ length: num }, (_, i) => ({ id: `out_${i + 1}`, name: `Chunk ${i + 1}` }));
          }
          return { ...n, data: newData };
        }
        return n;
      })
    );
  };

  if (!selectedNode) return <ModelSummaryDashboard />;

  const params = (selectedNode.data.params as any[]) || [];
  const paramValues = (selectedNode.data.paramValues as any) || {};

  // srinjoy's section categorisation
  const shapeParams    = params.filter((p: any) => p.section === 'shape');
  const basicParams    = params.filter((p: any) => !p.section || p.section === 'basic');
  const advancedParams = params.filter((p: any) => p.section === 'advanced');

  return (
    <div className="flex flex-col gap-4">
      {/* Node header — our UI */}
      <div className="pb-3 border-b border-[#363636]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[9px] uppercase tracking-wider text-[#555]">Selected Layer</span>
          <span className="text-[9px] font-mono text-[#555] bg-[#1e1e1e] border border-[#363636] px-1.5 py-px rounded-sm">{selectedNode.id}</span>
        </div>
        <div className="flex flex-col gap-3">
          {/* Label */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-[#888]">Label</label>
            <input
              type="text"
              className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] px-2 py-1.5 text-[12px] text-[#e2e2e2] outline-none transition-colors"
              value={(selectedNode.data.label as string) || ""}
              onChange={(e) =>
                setNodes((nds) =>
                  nds.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, label: e.target.value } } : n)
                )
              }
            />
          </div>
          {/* Variable name — our feature */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <label className="text-[10px] text-[#888]">Output variable name</label>
              <span className="text-[9px] font-mono text-[#555]">identifier</span>
            </div>
            <input
              type="text"
              spellCheck={false}
              placeholder={`auto: x_${selectedNode.id.replace(/-/g, "_")}`}
              className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] px-2 py-1.5 text-[12px] text-[#e2e2e2] font-mono outline-none transition-colors placeholder-[#444]"
              value={(selectedNode.data.varName as string) || ""}
              onChange={(e) =>
                setNodes((nds) =>
                  nds.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, varName: e.target.value } } : n)
                )
              }
            />
            <span className="text-[9px] text-[#555]">
              Leave blank to auto-generate. Used as the tensor variable in compiled PyTorch code.
            </span>
          </div>
        </div>
      </div>

      {/* Inferred Shapes section (srinjoy) — our theme */}
      {selectedNode.data.inferredShapes && Object.keys(selectedNode.data.inferredShapes).length > 0 && (
        <div className="flex flex-col gap-3 border-b border-[#363636] pb-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Tensor Shapes</div>
          {Object.entries(selectedNode.data.inferredShapes as Record<string, any>).map(([port, shape]) => (
            <div key={port} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <label className="text-[11px] text-[#aaa] capitalize flex items-center gap-1.5">
                  Port: {port.replace(/_/g, ' ')}
                  <span className="text-[9px] font-mono text-[#555] bg-[#252525] border border-[#363636] px-1 py-px rounded-sm">inferred</span>
                </label>
              </div>
              <input
                type="text"
                className="w-full bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-2 py-1.5 text-[12px] text-[#555] font-mono cursor-not-allowed"
                value={JSON.stringify(shape)}
                readOnly
                disabled
              />
            </div>
          ))}
        </div>
      )}

      {/* Fallback for legacy shape params (if any remain) */}
      {shapeParams.length > 0 && (!selectedNode.data.inferredShapes || Object.keys(selectedNode.data.inferredShapes).length === 0) && (
        <div className="flex flex-col gap-3 border-b border-[#363636] pb-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Tensor Shapes</div>
          {shapeParams.map((p: any) => (
            <ParamInput key={p.name} param={p} value={paramValues[p.name]} onChange={handleParamChange} />
          ))}
        </div>
      )}

      {/* Basic parameters (srinjoy) — our theme */}
      {basicParams.length > 0 && (
        <div className="flex flex-col gap-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Hyperparameters</div>
          {basicParams.map((p: any) => (
            <ParamInput key={p.name} param={p} value={paramValues[p.name]} onChange={handleParamChange} />
          ))}
        </div>
      )}

      {basicParams.length === 0 && shapeParams.length === 0 && (
        <div className="text-[11px] text-[#555] italic bg-[#1e1e1e] border border-[#363636] rounded-[3px] px-3 py-2">
          No configurable parameters.
        </div>
      )}

      {/* Advanced parameters — collapsible (srinjoy) — our theme */}
      {advancedParams.length > 0 && (
        <details className="group">
          <summary className="text-[10px] font-semibold text-[#555] uppercase tracking-wider cursor-pointer hover:text-[#d4d4d4] transition-colors list-none flex items-center gap-1.5">
            <span className="group-open:rotate-90 transition-transform inline-block text-[8px]">▶</span>
            Advanced Properties
          </summary>
          <div className="flex flex-col gap-3 mt-3 pl-2 border-l-2 border-[#363636]">
            {advancedParams.map((p: any) => (
              <ParamInput key={p.name} param={p} value={paramValues[p.name]} onChange={handleParamChange} />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

// ─── Drag-and-drop canvas ─────────────────────────────────────────────────────────

function DnDCanvas() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, setNodes, setEdges } = useReactFlow();
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);

  const onConnect = useCallback(
    (params: Connection | Edge) =>
      setEdges((eds: Edge[]) => addEdge({ ...params, animated: false, type: 'tensor' } as Edge, eds)),
    [setEdges]
  );

  // Clear shape error state when user edits graph
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

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  // srinjoy's exact onDrop logic — initialises paramValues from blockDef defaults
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData("application/reactflow");
      const label = event.dataTransfer.getData("application/label");
      const blockDefStr = event.dataTransfer.getData("application/blockDef");

      if (!type) return;

      const blockDef = blockDefStr ? JSON.parse(blockDefStr) : {};

      // Initialise paramValues with default values (srinjoy)
      const initialParamValues: any = {};
      if (blockDef.params) {
        blockDef.params.forEach((p: any) => {
          initialParamValues[p.name] = p.default;
        });
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

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
          varName: "",
        },
      };

      setNodes((nds: Node[]) => nds.concat(newNode as unknown as Node));
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div className="flex-1 relative" ref={canvasRef}>
      <ReactFlow
        defaultNodes={initialNodes}
        defaultEdges={initialEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onConnect={onConnect}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onDrop={onDrop}
        onDragOver={onDragOver}
        deleteKeyCode={["Backspace", "Delete"]}
        fitView
      >
        <Controls
          className="!bg-[#252525] !border-[#3a3a3a] !rounded-[3px]"
          style={{ bottom: 16, left: 16 }}
        />
        <MiniMap
          nodeColor={() => "#2d8cf0"}
          maskColor="rgba(18,18,18,0.8)"
          className="!bg-[#1e1e1e] !border !border-[#3a3a3a] !rounded-[3px]"
          style={{ bottom: 16, right: 16 }}
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="#333"
        />
      </ReactFlow>
    </div>
  );
}

// ─── Block list item ───────────────────────────────────────────────────────────────

const BlockItem = ({ blockDef }: { blockDef: any }) => {
  const onDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData("application/reactflow", "custom");
    event.dataTransfer.setData("application/label", blockDef.name);
    event.dataTransfer.setData("application/blockDef", JSON.stringify(blockDef));
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div
      onDragStart={onDragStart}
      draggable
      className="flex items-center gap-2.5 bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#3a3a3a] hover:border-[#505050] rounded-[3px] px-3 py-2 cursor-grab active:cursor-grabbing transition-colors group"
    >
      <div
        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{ background: blockDef.color || "#4a4a4a" }}
      />
      <span className="text-[12px] text-[#c8c8c8] group-hover:text-[#e2e2e2] transition-colors truncate">
        {blockDef.name}
      </span>
    </div>
  );
};

// ─── Header ───────────────────────────────────────────────────────────────────────

function Header() {
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();
  const setGeneratedCode = useEditorStore((s) => s.setGeneratedCode);
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const setNodeShapes = useEditorStore((s) => s.setNodeShapes);

  const [compiling, setCompiling] = useState(false);
  const [checkStatus, setCheckStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');
  const [checkMsg, setCheckMsg] = useState('');

  // srinjoy's buildPayload — reads live nodes/edges; includes our varName field
  const buildPayload = () => {
    const currentNodes = getNodes();
    const currentEdges = getEdges();
    return {
      nodes: currentNodes.map((n) => ({
        id: n.id,
        data: {
          block_id: n.data.block_id || "",
          label: n.data.label,
          is_functional: n.data.is_functional || false,
          paramValues: n.data.paramValues || {},
          varName: (n.data.varName as string) || "",
        },
      })),
      edges: currentEdges.map((e) => ({
        source: e.source,
        sourceHandle: e.sourceHandle || "",
        target: e.target,
        targetHandle: e.targetHandle || "",
      })),
    };
  };

  // srinjoy's handleExport — uses port 8001, returns code or compiler error
  const handleExport = async () => {
    setCompiling(true);
    setShapeErrorNodeId(null);
    setGeneratedCode("# Compiling via Python Backend Engine...");
    try {
      const response = await fetch(`${API_BASE}/api/compile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const data = await response.json();
      if (response.ok) {
        setGeneratedCode(data.code);
        setShapeErrorNodeId(null);
      } else {
        // Distinguish ShapeMismatch (422) from other compiler errors
        if (data.detail?.error === "ShapeMismatch") {
          setShapeErrorNodeId(data.detail.node_id);
          setGeneratedCode(
            `# ❌ Shape Mismatch at "${data.detail.node_label}":\n# ${data.detail.message}`
          );
        } else {
          setGeneratedCode(`# ❌ Compiler Error:\n# ${JSON.stringify(data.detail)}`);
        }
      }
    } catch (err: any) {
      setGeneratedCode(`# ❌ Network Error:\n# Could not reach backend: ${err.message}`);
    } finally {
      setCompiling(false);
    }
  };

  // srinjoy's handleCheck — runs shape check on port 8001 and back-fills inferred params
  const handleCheck = async () => {
    setCheckStatus('checking');
    setCheckMsg('');
    try {
      const response = await fetch(`${API_BASE}/api/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const data = await response.json();
      if (response.ok) {
        setCheckStatus('ok');
        setCheckMsg('Compatible ✓');
        // Store raw shapes for hover tooltips on nodes and edges
        setNodeShapes(data.node_shapes ?? {});
        // Back-fill inferred shape params and auto-resolved parameters on each node (srinjoy)
        setNodes((nds) => nds.map((n) => {
          const shapes = data.node_shapes?.[n.id];
          const newParams = data.node_params?.[n.id];
          if (!shapes && !newParams) return n;

          const params = (n.data.params as any[]) || [];
          const updatedValues = { ...(n.data.paramValues as any) };

          if (shapes) {
            params.forEach((p: any) => {
              if (p.section === 'shape') {
                if (p.name === 'output_shape' && shapes['out']) updatedValues['output_shape'] = JSON.stringify(shapes['out']);
                if (p.name === 'input_shape'  && shapes['in'])  updatedValues['input_shape']  = JSON.stringify(shapes['in']);
              }
            });
          }

          if (newParams) {
            Object.assign(updatedValues, newParams);
          }
          return { ...n, data: { ...n.data, paramValues: updatedValues, inferredShapes: shapes } };
        }));
      } else {
        setCheckStatus('error');
        const errMsg = typeof data.detail === 'string' ? data.detail : data.detail?.message;
        setCheckMsg(errMsg || 'Shape mismatch detected.');
      }
    } catch (err: any) {
      setCheckStatus('error');
      setCheckMsg(`Cannot reach backend: ${err.message}`);
    }
  };

  const handleReset = () => {
    if (confirm("Clear the canvas?")) {
      setNodes([]);
      setEdges([]);
      setShapeErrorNodeId(null);
      setNodeShapes({});
      setGeneratedCode("");
    }
  };

  const checkBtnClass = {
    idle:     'text-[#888] hover:text-[#d4d4d4] border-[#363636] hover:border-[#505050] bg-[#252525]',
    checking: 'text-[#555] border-[#363636] bg-[#1e1e1e] cursor-not-allowed',
    ok:       'text-[#4ade80] border-[#4ade80]/50 bg-[#4ade80]/10',
    error:    'text-[#e54545] border-[#e54545]/50 bg-[#e54545]/10',
  }[checkStatus];

  return (
    <header
      className="flex items-center justify-between px-4 py-0 z-20 flex-shrink-0"
      style={{ height: 40, background: "#1e1e1e", borderBottom: "1px solid #363636" }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="w-5 h-5 rounded-sm bg-[#2d8cf0] flex items-center justify-center">
          <Layers className="w-3 h-3 text-white" />
        </div>
        <span className="text-[13px] font-semibold text-[#d4d4d4] tracking-tight">ArchiDE</span>
        <span className="text-[10px] font-mono text-[#555] border border-[#363636] px-1.5 py-px rounded-sm">PyTorch</span>
      </div>

      {/* Status + actions */}
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-mono text-[#666]">
          {nodes.length} layers · {edges.length} edges
        </span>

        <div className="w-px h-4 bg-[#363636]" />

        {checkMsg && (
          <span className={`text-[10px] pr-2 ${checkStatus === 'ok' ? 'text-[#4ade80]' : 'text-[#e54545]'}`}>
            {checkMsg}
          </span>
        )}

        <button
          onClick={handleCheck}
          disabled={checkStatus === 'checking'}
          className={`flex items-center gap-1.5 text-[11px] font-medium transition-colors px-2 py-1 rounded-sm border ${checkBtnClass}`}
        >
          {checkStatus === 'checking' ? 'Checking...' : 'Check Shapes'}
        </button>

        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 text-[11px] text-[#888] hover:text-[#e2e2e2] transition-colors px-2 py-1 rounded-sm hover:bg-[#2a2a2a] ml-1"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>

        <button
          onClick={handleExport}
          disabled={compiling}
          className="flex items-center gap-1.5 text-[11px] font-medium text-white bg-[#2d8cf0] hover:bg-[#3a97f5] disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-3 py-1.5 rounded-sm"
        >
          <Play className="w-3 h-3 fill-white" />
          {compiling ? "Compiling…" : "Export PyTorch"}
        </button>
      </div>
    </header>
  );
}

// ─── Root component ───────────────────────────────────────────────────────────────

export default function Home() {
  const generatedCode = useEditorStore((s) => s.generatedCode);
  const [registry, setRegistry] = useState<any[]>(FALLBACK_BLOCKS);
  const [search, setSearch] = useState("");
  const [rightTab, setRightTab] = useState<"inspector" | "code">("inspector");
  const [rightOpen, setRightOpen] = useState(true);
  const [copied, setCopied] = useState(false);

  // srinjoy's registry fetch — port 8001, groups by category
  useEffect(() => {
    fetch(`${API_BASE}/api/blocks`)
      .then((r) => r.json())
      .then((data) => { if (Array.isArray(data) && data.length) setRegistry(data); })
      .catch(() => {}); // fall back to FALLBACK_BLOCKS
  }, []);

  // Group by category (srinjoy pattern)
  const categories: Record<string, any[]> = {};
  registry
    .filter((b) => b.name.toLowerCase().includes(search.toLowerCase()))
    .forEach((block) => {
      if (!categories[block.category]) categories[block.category] = [];
      categories[block.category].push(block);
    });

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([generatedCode], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "model.py";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col" style={{ background: "#181818", color: "#d4d4d4" }}>
        <Header />

        <div className="flex flex-1 overflow-hidden">

          {/* ── Left sidebar: block library ─────────────────────────── */}
          <aside
            className="flex flex-col flex-shrink-0 overflow-hidden"
            style={{ width: 220, background: "#1e1e1e", borderRight: "1px solid #363636" }}
          >
            {/* Sidebar header */}
            <div
              className="flex items-center justify-between px-3 flex-shrink-0"
              style={{ height: 32, borderBottom: "1px solid #363636" }}
            >
              <span className="text-[9px] uppercase tracking-wider text-[#555]">Layer Library</span>
            </div>

            {/* Search */}
            <div className="px-2 pt-2 pb-1 flex-shrink-0">
              <div className="flex items-center gap-1.5 bg-[#252525] border border-[#3a3a3a] rounded-[3px] px-2 py-1">
                <Search className="w-3 h-3 text-[#555]" />
                <input
                  type="text"
                  placeholder="Search layers..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="bg-transparent text-[11px] text-[#c8c8c8] placeholder-[#444] outline-none flex-1 min-w-0"
                />
              </div>
            </div>

            {/* Block list — grouped by category */}
            <div className="flex-1 overflow-y-auto px-2 py-1 space-y-3">
              {Object.keys(categories).length === 0 ? (
                <div className="text-[11px] text-[#555] italic px-1 pt-2">
                  {search ? "No matching layers." : "Loading layers…"}
                </div>
              ) : (
                Object.entries(categories).map(([category, blocks]) => (
                  <div key={category}>
                    <div className="flex items-center gap-1.5 px-1 py-1">
                      <CategoryIcon category={category} />
                      <span className="text-[9px] uppercase tracking-wider text-[#555]">{category}</span>
                    </div>
                    <div className="flex flex-col gap-1">
                      {blocks.map((block: any) => (
                        <BlockItem key={block.id} blockDef={block} />
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </aside>

          {/* ── Canvas ──────────────────────────────────────────────── */}
          <DnDCanvas />

          {/* ── Right panel: inspector + code ───────────────────────── */}
          {rightOpen && (
            <aside
              className="flex flex-col flex-shrink-0 overflow-hidden"
              style={{ width: 280, background: "#1e1e1e", borderLeft: "1px solid #363636" }}
            >
              {/* Tab bar */}
              <div
                className="flex items-center flex-shrink-0"
                style={{ height: 32, borderBottom: "1px solid #363636" }}
              >
                <button
                  onClick={() => setRightTab("inspector")}
                  className={`flex items-center gap-1.5 px-3 h-full text-[11px] border-b-2 transition-colors ${
                    rightTab === "inspector"
                      ? "text-[#d4d4d4] border-[#2d8cf0]"
                      : "text-[#666] border-transparent hover:text-[#aaa]"
                  }`}
                >
                  <Settings2 className="w-3 h-3" />
                  Inspector
                </button>
                <button
                  onClick={() => setRightTab("code")}
                  className={`flex items-center gap-1.5 px-3 h-full text-[11px] border-b-2 transition-colors ${
                    rightTab === "code"
                      ? "text-[#d4d4d4] border-[#2d8cf0]"
                      : "text-[#666] border-transparent hover:text-[#aaa]"
                  }`}
                >
                  <Code2 className="w-3 h-3" />
                  Code
                </button>
                <div className="flex-1" />
                <button
                  onClick={() => setRightOpen(false)}
                  className="p-1 mr-1 text-[#555] hover:text-[#aaa] transition-colors"
                  title="Close panel"
                >
                  <PanelRightClose className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Inspector tab */}
              {rightTab === "inspector" && (
                <div className="flex-1 overflow-y-auto p-3">
                  <PropertiesPanel />
                </div>
              )}

              {/* Code tab */}
              {rightTab === "code" && (
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Code actions */}
                  <div
                    className="flex items-center gap-1.5 px-3 flex-shrink-0"
                    style={{ height: 32, borderBottom: "1px solid #363636" }}
                  >
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#e2e2e2] transition-colors"
                    >
                      {copied ? <Check className="w-3 h-3 text-[#4ade80]" /> : <Copy className="w-3 h-3" />}
                      {copied ? "Copied" : "Copy"}
                    </button>
                    <div className="w-px h-3 bg-[#363636]" />
                    <button
                      onClick={handleDownload}
                      className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#e2e2e2] transition-colors"
                    >
                      <Download className="w-3 h-3" />
                      Download
                    </button>
                  </div>

                  {/* Code content */}
                  <div className="flex-1 overflow-auto p-3">
                    <pre
                      className="text-[11px] font-mono leading-relaxed"
                      style={{ color: generatedCode.startsWith("# ❌") ? "#e54545" : "#7ec8e3" }}
                    >
                      {generatedCode || "# Export PyTorch code will appear here after compiling."}
                    </pre>
                  </div>
                </div>
              )}
            </aside>
          )}

          {/* Collapsed panel toggle */}
          {!rightOpen && (
            <button
              onClick={() => setRightOpen(true)}
              className="flex-shrink-0 flex items-center justify-center text-[#555] hover:text-[#aaa] transition-colors"
              style={{ width: 24, background: "#1e1e1e", borderLeft: "1px solid #363636" }}
              title="Open panel"
            >
              <PanelRightOpen className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </ReactFlowProvider>
  );
}
