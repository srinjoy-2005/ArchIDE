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
const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];
let nodeCounter = 100;
const getId = () => `node_${nodeCounter++}`;

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

// ─── Shape hint map ─────────────────────────────────────────────────────────────

function getShapeHint(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("conv2d"))  return "(B,C,H,W) → (B,C',H',W')";
  if (n.includes("linear"))  return "(B,In) → (B,Out)";
  if (n.includes("relu") || n.includes("softmax")) return "elementwise activation";
  if (n.includes("input"))   return "model input tensor";
  if (n.includes("output"))  return "model output";
  if (n.includes("add"))     return "tensor addition";
  if (n.includes("split"))   return "split along dim";
  return "pytorch block";
}

// ─── Category icon helper ────────────────────────────────────────────────────────

function CategoryIcon({ category }: { category: string }) {
  const cls = "w-3 h-3 flex-shrink-0";
  const c = category.toLowerCase();
  if (c.includes("core"))       return <Box className={cls} style={{ color: "#60a5fa" }} />;
  if (c.includes("activation")) return <Sparkles className={cls} style={{ color: "#a78bfa" }} />;
  if (c.includes("tensor"))     return <ArrowRightLeft className={cls} style={{ color: "#fb923c" }} />;
  return <Brain className={cls} style={{ color: "#6b7280" }} />;
}

// ─── Model Summary Dashboard ─────────────────────────────────────────────────────

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
      { id: "e12", source: "s1", sourceHandle: "out", target: "s2", targetHandle: "in", animated: true, style: { stroke: "#4a4a4a" } },
      { id: "e23", source: "s2", sourceHandle: "out", target: "s3", targetHandle: "in", animated: true, style: { stroke: "#4a4a4a" } },
      { id: "e34", source: "s3", sourceHandle: "out", target: "s4", targetHandle: "in", animated: true, style: { stroke: "#4a4a4a" } },
      { id: "e45", source: "s4", sourceHandle: "out", target: "s5", targetHandle: "in", animated: true, style: { stroke: "#4a4a4a" } },
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

function PropertiesPanel() {
  const { setNodes } = useReactFlow();
  const nodes = useNodes();
  const selectedNode = nodes.find((n) => n.selected) || null;

  const handleParamChange = (paramName: string, value: any) => {
    if (!selectedNode) return;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === selectedNode.id) {
          const newData: any = {
            ...n.data,
            paramValues: { ...((n.data.paramValues as any) || {}), [paramName]: value },
          };
          if (paramName === "num_inputs") {
            const num = parseInt(value) || 2;
            newData.inputs = Array.from({ length: num }, (_, i) => ({ id: `in_${i}`, name: `Input ${i + 1}` }));
          } else if (paramName === "chunks") {
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

  const shapeParams   = params.filter((p: any) => p.section === 'shape');
  const basicParams   = params.filter((p: any) => !p.section || p.section === 'basic');
  const advancedParams = params.filter((p: any) => p.section === 'advanced');

  return (
    <div className="flex flex-col gap-4">
      {/* Node header */}
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
          {/* Variable name */}
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
              Leave blank to auto-generate. Used as the tensor variable in the compiled PyTorch code.
            </span>
          </div>
        </div>
      </div>

      {/* Shape section — always visible */}
      {shapeParams.length > 0 && (
        <div className="flex flex-col gap-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Tensor Shapes</div>
          {shapeParams.map((p: any) => (
            <ParamInput key={p.name} param={p} value={paramValues[p.name]} onChange={handleParamChange} />
          ))}
        </div>
      )}

      {/* Basic parameters */}
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
          No parameters to configure.
        </div>
      )}

      {/* Advanced parameters — collapsible */}
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
      setEdges((eds: Edge[]) => addEdge({ ...params, animated: true, style: { stroke: "#4a4a4a" } } as Edge, eds)),
    [setEdges]
  );

  // TODO 5 — clear shape error state when user edits graph
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

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/reactflow");
      const blockDefStr = event.dataTransfer.getData("application/blockDef");
      if (!type) return;

      const blockDef = blockDefStr ? JSON.parse(blockDefStr) : {};
      const initialParamValues: any = {};
      (blockDef.params || []).forEach((p: any) => { initialParamValues[p.name] = p.default; });

      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setNodes((nds: Node[]) =>
        nds.concat({
          id: getId(),
          type,
          position,
          data: {
            block_id: blockDef.id,
            label: blockDef.name,
            params: blockDef.params || [],
            paramValues: initialParamValues,
            inputs: blockDef.inputs || [],
            outputs: blockDef.outputs || [],
            is_functional: blockDef.is_functional || false,
          },
        } as unknown as Node)
      );
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div className="flex-1 relative overflow-hidden" ref={canvasRef} style={{ background: "#1a1a1a" }}>
      <ReactFlow
        defaultNodes={initialNodes}
        defaultEdges={initialEdges}
        nodeTypes={nodeTypes}
        onConnect={onConnect}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onDrop={onDrop}
        onDragOver={onDragOver}
        deleteKeyCode={["Backspace", "Delete"]}
        defaultEdgeOptions={{ animated: true, style: { stroke: "#4a4a4a", strokeWidth: 1.5 } }}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Controls position="bottom-right" className="!m-3" />
        <MiniMap
          nodeColor={() => "#404040"}
          maskColor="rgba(26, 26, 26, 0.8)"
          className="!m-3 !rounded-[3px]"
        />
        <Background variant={BackgroundVariant.Dots} color="#2e2e2e" gap={20} size={1} />
      </ReactFlow>
    </div>
  );
}

// ─── Left sidebar — block item ────────────────────────────────────────────────────

function BlockItem({ blockDef }: { blockDef: any }) {
  const { setNodes } = useReactFlow();

  const onDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData("application/reactflow", "custom");
    event.dataTransfer.setData("application/blockDef", JSON.stringify(blockDef));
    event.dataTransfer.effectAllowed = "move";
  };

  const handleAdd = () => {
    const initialParamValues: any = {};
    (blockDef.params || []).forEach((p: any) => { initialParamValues[p.name] = p.default; });
    setNodes((nds: Node[]) =>
      nds.concat({
        id: getId(),
        type: "custom",
        position: { x: 220 + Math.random() * 60, y: 120 + Math.random() * 60 },
        data: {
          block_id: blockDef.id,
          label: blockDef.name,
          params: blockDef.params || [],
          paramValues: initialParamValues,
          inputs: blockDef.inputs || [],
          outputs: blockDef.outputs || [],
          is_functional: blockDef.is_functional || false,
        },
      } as unknown as Node)
    );
  };

  const shapeHint = getShapeHint(blockDef.name);
  const firstTwoParams = (blockDef.params || []).slice(0, 2);

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className="group flex flex-col gap-1 bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#363636] hover:border-[#505050] rounded-[3px] px-3 py-2 cursor-grab active:cursor-grabbing transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CategoryIcon category={blockDef.category} />
          <span className="text-[12px] font-medium text-[#d4d4d4] group-hover:text-[#e2e2e2] transition-colors">
            {blockDef.name}
          </span>
        </div>
        <button
          onClick={handleAdd}
          title="Add to canvas"
          className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded text-[#888] hover:text-[#2d8cf0]"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>

      <span className="text-[9px] text-[#555] font-mono">{shapeHint}</span>

      {firstTwoParams.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-0.5">
          {firstTwoParams.map((p: any) => (
            <span key={p.name} className="text-[9px] font-mono text-[#666] bg-[#252525] border border-[#363636] px-1.5 py-px rounded-sm">
              {p.name.replace("_channels","").replace("_features","") || p.name}={p.default}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Header() {
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const setGeneratedCode = useEditorStore((s) => s.setGeneratedCode);
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const nodes = useNodes();
  const edges = useEdges();
  
  const [compiling, setCompiling] = useState(false);
  const [checkStatus, setCheckStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');
  const [checkMsg, setCheckMsg] = useState('');

  const buildPayload = () => {
    // We use getNodes/getEdges to get the latest state safely inside handlers
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

  const handleExport = async () => {
    setCompiling(true);
    setShapeErrorNodeId(null);
    setGeneratedCode("# Compiling...");

    try {
      const response = await fetch("http://localhost:8000/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const data = await response.json();

      if (response.ok) {
        setGeneratedCode(data.code);
        setShapeErrorNodeId(null);
      } else {
        // Check for ShapeMismatch error from 422
        if (data.detail?.error === "ShapeMismatch") {
          setShapeErrorNodeId(data.detail.node_id);
          setGeneratedCode(
            `# ❌ Shape Mismatch at "${data.detail.node_label}":\n# ${data.detail.message}`
          );
        } else {
          setGeneratedCode(`# ❌ Compiler Error:\n# ${data.detail}`);
        }
      }
    } catch (err: any) {
      setGeneratedCode(`# ❌ Network Error:\n# Could not reach backend: ${err.message}`);
    } finally {
      setCompiling(false);
    }
  };

  const handleReset = () => {
    if (confirm("Clear the canvas?")) {
      setNodes([]);
      setEdges([]);
      setShapeErrorNodeId(null);
    }
  };

  const handleCheck = async () => {
    setCheckStatus('checking');
    setCheckMsg('');
    try {
      const response = await fetch("http://localhost:8001/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload())
      });
      const data = await response.json();
      if (response.ok) {
        setCheckStatus('ok');
        setCheckMsg('Compatible ✓');
        // Update inferred shape params and auto-resolved parameters on each node
        setNodes(nds => nds.map(n => {
          const shapes = data.node_shapes?.[n.id];
          const newParams = data.node_params?.[n.id];
          if (!shapes && !newParams) return n;
          
          const params = (n.data.params as any[]) || [];
          const updatedValues = { ...(n.data.paramValues as any) };
          
          // Apply shapes
          if (shapes) {
            params.forEach((p: any) => {
              if (p.section === 'shape') {
                if (p.name === 'output_shape' && shapes['out']) updatedValues['output_shape'] = JSON.stringify(shapes['out']);
                if (p.name === 'input_shape'  && shapes['in'])  updatedValues['input_shape']  = JSON.stringify(shapes['in']);
              }
            });
          }
          
          // Apply auto-inferred parameters (e.g. in_features = 64)
          if (newParams) {
            Object.assign(updatedValues, newParams);
          }
          
          return { ...n, data: { ...n.data, paramValues: updatedValues } };
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
          className="flex items-center gap-1.5 text-[11px] text-[#888] hover:text-[#e2e2e2] transition-colors px-2 py-1 rounded-sm hover:bg-[#2a2a2a] ml-2"
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
  const [catFilter, setCatFilter] = useState("All");
  const [rightTab, setRightTab] = useState<"inspector" | "code">("inspector");
  const [rightOpen, setRightOpen] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/blocks")
      .then((r) => r.json())
      .then((data) => { if (Array.isArray(data) && data.length) setRegistry(data); })
      .catch(() => {});
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([generatedCode], { type: "text/plain" }));
    a.download = "archide_model.py";
    a.click();
  };

  // Build filtered, grouped registry
  const allCategories = Array.from(new Set(registry.map((b) => b.category)));
  const filtered = registry.filter((b) => {
    const matchSearch = b.name.toLowerCase().includes(search.toLowerCase());
    const matchCat = catFilter === "All" || b.category === catFilter;
    return matchSearch && matchCat;
  });
  const grouped: Record<string, any[]> = {};
  filtered.forEach((b) => {
    if (!grouped[b.category]) grouped[b.category] = [];
    grouped[b.category].push(b);
  });

  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col overflow-hidden" style={{ background: "#1a1a1a", fontFamily: "var(--font-inter), sans-serif" }}>
        <Header />

        <div className="flex flex-1 overflow-hidden relative">
          {/* ── Left Sidebar ─────────────────────────────────────────────── */}
          <aside
            className="flex flex-col overflow-hidden z-10 flex-shrink-0"
            style={{ width: 240, background: "#252525", borderRight: "1px solid #363636" }}
          >
            {/* Search */}
            <div className="px-3 py-2" style={{ borderBottom: "1px solid #363636" }}>
              <div className="relative">
                <Search className="w-3 h-3 absolute left-2 top-2.5 text-[#555]" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search layers…"
                  className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] pl-7 pr-2 py-1.5 text-[11px] text-[#d4d4d4] placeholder-[#555] outline-none transition-colors"
                />
              </div>
            </div>

            {/* Category pills */}
            <div className="flex items-center gap-1 px-3 py-2 overflow-x-auto" style={{ borderBottom: "1px solid #363636" }}>
              {["All", ...allCategories].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setCatFilter(cat)}
                  className={`text-[10px] px-2 py-0.5 rounded-sm whitespace-nowrap transition-colors border ${
                    catFilter === cat
                      ? "bg-[#2d8cf0] border-[#2d8cf0] text-white"
                      : "bg-[#1e1e1e] border-[#363636] text-[#888] hover:text-[#d4d4d4] hover:border-[#505050]"
                  }`}
                >
                  {cat === "Core Layers" ? "Core" : cat === "Tensor Ops" ? "Tensor" : cat}
                </button>
              ))}
            </div>

            {/* Block list */}
            <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-5">
              {Object.keys(grouped).length === 0 ? (
                <div className="text-[11px] text-[#555] text-center pt-8">No layers match</div>
              ) : (
                Object.entries(grouped).map(([cat, blocks]) => (
                  <div key={cat} className="flex flex-col gap-1.5">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1.5">
                        <CategoryIcon category={cat} />
                        <span className="text-[9px] uppercase tracking-wider text-[#555] font-semibold">{cat}</span>
                      </div>
                      <span className="text-[9px] font-mono text-[#444]">{blocks.length}</span>
                    </div>
                    {blocks.map((b) => <BlockItem key={b.id} blockDef={b} />)}
                  </div>
                ))
              )}
            </div>
          </aside>

          {/* ── Canvas ───────────────────────────────────────────────────── */}
          <DnDCanvas />

          {/* ── Right panel collapse toggle (when closed) ─────────────── */}
          {!rightOpen && (
            <button
              onClick={() => setRightOpen(true)}
              className="absolute right-3 top-3 z-30 p-1.5 bg-[#252525] border border-[#363636] hover:border-[#505050] rounded-[3px] text-[#888] hover:text-[#d4d4d4] transition-colors shadow-lg"
            >
              <PanelRightOpen className="w-3.5 h-3.5" />
            </button>
          )}

          {/* ── Right Sidebar ─────────────────────────────────────────── */}
          {rightOpen && (
            <aside
              className="flex flex-col flex-shrink-0 z-10 overflow-hidden"
              style={{ width: 280, background: "#252525", borderLeft: "1px solid #363636" }}
            >
              {/* Tab bar */}
              <div
                className="flex items-center px-1 py-1 gap-0.5 flex-shrink-0"
                style={{ borderBottom: "1px solid #363636", background: "#1e1e1e" }}
              >
                <button
                  onClick={() => setRightTab("inspector")}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 text-[11px] font-medium rounded-sm transition-colors ${
                    rightTab === "inspector"
                      ? "bg-[#252525] text-[#d4d4d4]"
                      : "text-[#666] hover:text-[#aaa]"
                  }`}
                >
                  <Settings2 className="w-3 h-3" />
                  Inspector
                </button>
                <button
                  onClick={() => setRightTab("code")}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 text-[11px] font-medium rounded-sm transition-colors ${
                    rightTab === "code"
                      ? "bg-[#252525] text-[#d4d4d4]"
                      : "text-[#666] hover:text-[#aaa]"
                  }`}
                >
                  <Code2 className="w-3 h-3" />
                  PyTorch Code
                </button>
                <button
                  onClick={() => setRightOpen(false)}
                  className="p-1.5 text-[#555] hover:text-[#aaa] rounded-sm transition-colors ml-0.5"
                >
                  <PanelRightClose className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Tab content */}
              {rightTab === "inspector" ? (
                <div className="flex-1 p-4 overflow-y-auto">
                  <PropertiesPanel />
                </div>
              ) : (
                <div className="flex-1 flex flex-col min-h-0" style={{ background: "#181818" }}>
                  {/* Code toolbar */}
                  <div
                    className="flex items-center justify-between px-3 py-2 flex-shrink-0"
                    style={{ borderBottom: "1px solid #363636", background: "#1e1e1e" }}
                  >
                    <span className="text-[10px] font-mono text-[#555]">nn.Module · output.py</span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={handleCopy}
                        className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#d4d4d4] bg-[#252525] hover:bg-[#2a2a2a] border border-[#363636] px-2 py-0.5 rounded-sm transition-colors"
                      >
                        {copied ? <><Check className="w-3 h-3 text-[#4ade80]" /><span className="text-[#4ade80]">Copied</span></> : <><Copy className="w-3 h-3" /><span>Copy</span></>}
                      </button>
                      <button
                        onClick={handleDownload}
                        className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#d4d4d4] bg-[#252525] hover:bg-[#2a2a2a] border border-[#363636] px-2 py-0.5 rounded-sm transition-colors"
                      >
                        <Download className="w-3 h-3" />
                        <span>.py</span>
                      </button>
                    </div>
                  </div>

                  {/* Code body */}
                  <div className="flex-1 overflow-auto p-4">
                    <pre className="text-[11px] font-mono text-[#9da3ae] leading-relaxed whitespace-pre-wrap">
                      {generatedCode}
                    </pre>
                  </div>
                </div>
              )}
            </aside>
          )}
        </div>
      </div>
    </ReactFlowProvider>
  );
}
