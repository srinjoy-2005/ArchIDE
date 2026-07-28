"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  ReactFlowProvider,
  useReactFlow,
  useOnSelectionChange,
  useNodes
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import CustomNode from "../components/CustomNode";
import { useEditorStore } from "../lib/store";

const nodeTypes = { custom: CustomNode };

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

let id = 2;
const getId = () => `node_${id++}`;

function PropertiesPanel() {
  const { setNodes } = useReactFlow();
  const nodes = useNodes();
  
  // Directly find the selected node from the store
  const selectedNode = nodes.find(n => n.selected) || null;

  const handleParamChange = (paramName: string, value: any) => {
    if (!selectedNode) return;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === selectedNode.id) {
          const newData: any = {
            ...n.data,
            paramValues: {
              ...(n.data.paramValues as any || {}),
              [paramName]: value
            }
          };

          // If the param is num_inputs or chunks, dynamically generate input/output ports!
          if (paramName === 'num_inputs') {
            const num = parseInt(value) || 2;
            newData.inputs = Array.from({ length: num }, (_, i) => ({
              id: `in_${i}`, name: `Input ${i + 1}`
            }));
          } else if (paramName === 'chunks') {
            const num = parseInt(value) || 2;
            newData.outputs = Array.from({ length: num }, (_, i) => ({
              id: `out_${i}`, name: `Chunk ${i + 1}`
            }));
          }

          return {
            ...n,
            data: newData
          };
        }
        return n;
      })
    );
  };

  if (!selectedNode) {
    return <div className="text-slate-500 text-sm">Select a block to edit its properties.</div>;
  }

  const params = (selectedNode.data.params as any[]) || [];
  const paramValues = (selectedNode.data.paramValues as any) || {};

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1 mb-2 border-b border-slate-700 pb-4">
        <label className="text-xs text-slate-400 capitalize">Node Name (Label)</label>
        <input
          type="text"
          className="bg-slate-800 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-blue-500 transition-colors"
          value={selectedNode.data.label as string}
          onChange={(e) => {
            setNodes(nds => nds.map(n => n.id === selectedNode.id ? { ...n, data: { ...n.data, label: e.target.value } } : n));
          }}
        />
      </div>

      <div className="font-semibold text-slate-200">Block Parameters</div>
      {params.length === 0 ? (
        <div className="text-sm text-slate-500">No parameters to configure.</div>
      ) : (
        params.map(param => (
          <div key={param.name} className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 capitalize">{param.name.replace(/_/g, ' ')}</label>
            <input
              type={param.type === 'int' || param.type === 'float' ? 'number' : 'text'}
              className="bg-slate-800 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-blue-500 transition-colors"
              value={paramValues[param.name] ?? param.default}
              onChange={(e) => handleParamChange(param.name, param.type === 'int' ? parseInt(e.target.value) : e.target.value)}
            />
          </div>
        ))
      )}
    </div>
  );
}

function DnDCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, setNodes, setEdges } = useReactFlow();

  const onConnect = useCallback(
    (params: Connection | Edge) => setEdges((eds: Edge[]) => addEdge(params, eds)),
    [setEdges]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData("application/reactflow");
      const label = event.dataTransfer.getData("application/label");
      const blockDefStr = event.dataTransfer.getData("application/blockDef");

      if (typeof type === "undefined" || !type) {
        return;
      }
      
      const blockDef = blockDefStr ? JSON.parse(blockDefStr) : {};

      // Initialize paramValues with default values
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
          is_functional: blockDef.is_functional || false
        },
      };

      setNodes((nds: Node[]) => nds.concat(newNode as unknown as Node));
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div className="flex-1 bg-slate-950 relative" ref={reactFlowWrapper}>
      <ReactFlow
        defaultNodes={initialNodes}
        defaultEdges={initialEdges}
        nodeTypes={nodeTypes}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        deleteKeyCode={["Backspace", "Delete"]}
        fitView
        className="react-flow-dark"
      >
        <Controls />
        <MiniMap
          nodeColor={(n: Node) => "#3b82f6"}
          maskColor="rgba(15, 23, 42, 0.7)"
        />
        <Background color="#334155" gap={16} />
      </ReactFlow>
    </div>
  );
}

const BlockItem = ({ blockDef }: { blockDef: any }) => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.setData("application/label", blockDef.name);
    event.dataTransfer.setData("application/blockDef", JSON.stringify(blockDef));
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div
      onDragStart={(event) => onDragStart(event, "custom")}
      draggable
      className="p-3 rounded-md border border-slate-700 cursor-grab hover:border-blue-500 hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all font-semibold text-slate-200"
      style={{ backgroundColor: blockDef.color || '#1e293b' }}
    >
      {blockDef.name}
    </div>
  );
};

const Header = () => {
  const { getNodes, getEdges } = useReactFlow();
  const setGeneratedCode = useEditorStore((state) => state.setGeneratedCode);

  const handleExport = async () => {
    const nodes = getNodes();
    const edges = getEdges();
    
    // Prepare the payload for FastAPI
    const payload = {
      nodes: nodes.map(n => ({
        id: n.id,
        data: {
          block_id: n.data.block_id || "",
          label: n.data.label,
          is_functional: n.data.is_functional || false,
          paramValues: n.data.paramValues || {}
        }
      })),
      edges: edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle || "",
        target: e.target,
        targetHandle: e.targetHandle || ""
      }))
    };

    setGeneratedCode("# Compiling via Python Backend Engine...");

    try {
      const response = await fetch("http://localhost:8000/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      
      if (response.ok) {
        setGeneratedCode(data.code);
      } else {
        setGeneratedCode(`# COMPILER ERROR\n${data.detail}`);
      }
    } catch (err: any) {
      setGeneratedCode(`# NETWORK ERROR\nFailed to reach compiler backend: ${err.message}`);
    }
  };

  return (
    <header className="flex items-center justify-between bg-slate-900 border-b border-slate-800 p-4 shadow-xl z-10">
      <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 text-transparent bg-clip-text">ArchiDE</h1>
      <button 
        onClick={handleExport}
        className="rounded-md bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-500 transition-colors shadow-lg shadow-blue-600/20"
      >
        Export PyTorch
      </button>
    </header>
  );
};

export default function Home() {
  const generatedCode = useEditorStore((state) => state.generatedCode);
  const [registry, setRegistry] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/blocks")
      .then(res => res.json())
      .then(data => setRegistry(data))
      .catch(err => console.error("Failed to fetch blocks", err));
  }, []);

  // Group blocks by category
  const categories: Record<string, any[]> = {};
  registry.forEach(block => {
    if (!categories[block.category]) {
      categories[block.category] = [];
    }
    categories[block.category].push(block);
  });

  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col">
        <Header />
        
        <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-slate-900 border-r border-slate-800 p-4 flex flex-col gap-6 overflow-y-auto z-10 shadow-2xl shadow-black/50">
          {Object.keys(categories).length === 0 ? (
             <div className="text-sm text-slate-400 animate-pulse">Loading blocks...</div>
          ) : (
            Object.keys(categories).map(category => (
              <div key={category}>
                <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">{category}</h2>
                <div className="flex flex-col gap-2">
                  {categories[category].map((block: any) => (
                    <BlockItem key={block.id} blockDef={block} />
                  ))}
                </div>
              </div>
            ))
          )}
        </aside>

        {/* Canvas */}
        <DnDCanvas />

        {/* Right Sidebar (Split between Properties and Code) */}
        <aside className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col z-10 shadow-2xl h-full overflow-hidden">
          {/* Properties Panel (Top Half) */}
          <div className="flex-1 border-b border-slate-800 flex flex-col min-h-0">
             <div className="p-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md">
                <h2 className="text-sm font-semibold text-slate-300">Properties</h2>
             </div>
             <div className="flex-1 p-4 overflow-y-auto">
                <PropertiesPanel />
             </div>
          </div>
          
          {/* Code Preview (Bottom Half) */}
          <div className="flex-1 flex flex-col min-h-0">
             <div className="p-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md">
                <h2 className="text-sm font-semibold text-slate-300">Generated Code</h2>
             </div>
             <div className="flex-1 p-4 overflow-auto text-sm font-mono text-indigo-300 bg-[#0c1017]">
               <pre>{generatedCode}</pre>
             </div>
          </div>
        </aside>
      </div>
    </div>
    </ReactFlowProvider>
  );
}
