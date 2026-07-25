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
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import CustomNode from "../components/CustomNode";
import { generatePyTorchCode } from "../lib/codegen";
import { useEditorStore } from "../lib/store";

const nodeTypes = { custom: CustomNode };

const initialNodes: Node[] = [
  { id: "1", type: "custom", position: { x: 50, y: 50 }, data: { label: "Input", description: "Entry point of the model" } },
];
const initialEdges: Edge[] = [];

let id = 2;
const getId = () => `node_${id++}`;

function DnDCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  // Expose nodes/edges to the store or handle globally if needed, 
  // but for now, Header will read them via useReactFlow().getNodes()

  // Make Backspace deletion explicitly work
  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    // Handled by React Flow default
  }, []);

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

      if (typeof type === "undefined" || !type) {
        return;
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: getId(),
        type,
        position,
        data: { label, description: `A ${label} layer` },
      };

      setNodes((nds: Node[]) => nds.concat(newNode as unknown as Node));
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div className="flex-1 bg-slate-950 relative" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        deleteKeyCode={["Backspace", "Delete"]}
        fitView
        className="react-flow-dark"
      >
        <Controls />
        <MiniMap
          nodeColor={(n: Node) => {
            return "#3b82f6";
          }}
          maskColor="rgba(15, 23, 42, 0.7)"
        />
        <Background color="#334155" gap={16} />
      </ReactFlow>
    </div>
  );
}

const BlockItem = ({ label }: { label: string }) => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.setData("application/label", label);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div
      onDragStart={(event) => onDragStart(event, "custom")}
      draggable
      className="bg-slate-800 p-3 rounded-md border border-slate-700 cursor-grab hover:border-blue-500 hover:bg-slate-700 hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all font-semibold text-slate-200"
    >
      {label}
    </div>
  );
};

const Header = () => {
  const { getNodes, getEdges } = useReactFlow();
  const setGeneratedCode = useEditorStore((state) => state.setGeneratedCode);

  const handleExport = () => {
    const nodes = getNodes();
    const edges = getEdges();
    const code = generatePyTorchCode(nodes, edges);
    setGeneratedCode(code);
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

  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col">
        <Header />
        
        <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-slate-900 border-r border-slate-800 p-4 flex flex-col gap-6 overflow-y-auto z-10 shadow-2xl shadow-black/50">
          <div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Core Layers</h2>
            <div className="flex flex-col gap-2">
              <BlockItem label="Input" />
              <BlockItem label="Linear" />
              <BlockItem label="Conv2D" />
            </div>
          </div>
          
          <div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Activations</h2>
            <div className="flex flex-col gap-2">
              <BlockItem label="ReLU" />
              <BlockItem label="Sigmoid" />
              <BlockItem label="Tanh" />
              <BlockItem label="Softmax" />
            </div>
          </div>

          <div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Regularization</h2>
            <div className="flex flex-col gap-2">
              <BlockItem label="Dropout" />
              <BlockItem label="BatchNorm2d" />
            </div>
          </div>
          
          <div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Pooling</h2>
            <div className="flex flex-col gap-2">
              <BlockItem label="MaxPool2d" />
              <BlockItem label="AvgPool2d" />
            </div>
          </div>
        </aside>

        {/* Canvas */}
        <DnDCanvas />

        {/* Code Preview */}
        <aside className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col z-10 shadow-2xl">
          <div className="p-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md">
            <h2 className="text-sm font-semibold text-slate-300">Generated Code</h2>
          </div>
          <div className="flex-1 p-4 overflow-auto text-sm font-mono text-indigo-300 bg-[#0c1017]">
            <pre>
{generatedCode}
            </pre>
          </div>
        </aside>
      </div>
    </div>
    </ReactFlowProvider>
  );
}
