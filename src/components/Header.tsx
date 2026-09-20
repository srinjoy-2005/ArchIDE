"use client";

/**
 * src/components/Header.tsx
 *
 * The top application header bar. Contains:
 * - Brand logo and version badge
 * - Live graph stats (layer count, edge count)
 * - "Check Shapes" button — calls /api/check, back-fills inferred shapes onto nodes,
 *   and updates the nodeShapes store slice used by hover tooltips on edges/nodes.
 * - "Reset" button — clears the canvas with confirmation
 * - "Export PyTorch" button — calls /api/compile and writes the result to the code panel
 * - Debug button (dev only) — opens the payload inspector at /dev/payloads
 *
 * API logic (buildPayload, handleExport, handleCheck) lives here because it needs
 * direct access to both the live React Flow state (getNodes/getEdges) and the
 * multi-file Zustand store to construct the multi-graph payload.
 */

import { useState } from 'react';
import { useReactFlow, useNodes, useEdges } from '@xyflow/react';
import { useEditorStore } from '../lib/store';
import { Layers, Play, RotateCcw, Bug } from 'lucide-react';
import { API_BASE } from '../lib/constants';

export function Header() {
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();
  const setGeneratedCode  = useEditorStore((s) => s.setGeneratedCode);
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const setNodeShapes     = useEditorStore((s) => s.setNodeShapes);
  const setActiveViewMode = useEditorStore((s) => s.setActiveViewMode);
  const files             = useEditorStore((s) => s.files);
  const activeFileId      = useEditorStore((s) => s.activeFileId);

  const [compiling,    setCompiling]    = useState(false);
  const [checkStatus,  setCheckStatus]  = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');
  const [checkMsg,     setCheckMsg]     = useState('');

  /** Build the multi-graph JSON payload for /api/compile and /api/check */
  const buildPayload = () => {
    const currentNodes = getNodes();
    const currentEdges = getEdges();
    const graphs: any = {};
    for (const f of files) {
      if (f.fileType === 'code' || f.name.endsWith('.py')) continue;
      const isCurrent = f.id === activeFileId;
      const nList = isCurrent ? currentNodes : f.nodes;
      const eList = isCurrent ? currentEdges : f.edges;
      graphs[f.id] = {
        name: f.name.replace(/\.[^/.]+$/, ""),
        parameters: f.parameters || [],
        nodes: nList.map((n) => ({
          id: n.id,
          position: n.position || { x: 100, y: 100 },
          data: {
            block_id: n.data.block_id || '',
            label: n.data.label,
            is_functional: n.data.is_functional || false,
            paramValues: n.data.paramValues || {},
            varName: (n.data.varName as string) || '',
            custom_module_id: (n.data.custom_module_id as string) || '',
          },
        })),
        edges: eList.map((e) => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle || '',
          target: e.target,
          targetHandle: e.targetHandle || '',
        })),
      };
    }
    return { main_graph_id: activeFileId, graphs };
  };

  /** Broadcast payload to the dev tools panel (/dev/payloads) via BroadcastChannel */
  const broadcastPayload = (endpoint: string, reqPayload: any, resPayload: any) => {
    if (typeof window !== 'undefined') {
      try {
        const channel = new BroadcastChannel('archide_payloads');
        channel.postMessage({ endpoint, request: reqPayload, response: resPayload, timestamp: Date.now() });
        channel.close();
      } catch (e) {
        console.error('Broadcast failed', e);
      }
    }
  };

  const handleExport = async () => {
    setCompiling(true);
    setShapeErrorNodeId(null);
    setGeneratedCode('# Compiling via Python Backend Engine...');
    try {
      const payload = buildPayload();
      const response = await fetch(`${API_BASE}/api/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      broadcastPayload('/api/compile', payload, data);

      if (response.ok) {
        setGeneratedCode(data.code);
        setShapeErrorNodeId(null);
        setActiveViewMode('code');
      } else {
        if (data.detail?.error === 'ShapeMismatch') {
          setShapeErrorNodeId(data.detail.node_id);
          setGeneratedCode(`# ❌ Shape Mismatch at "${data.detail.node_label}":\n# ${data.detail.message}`);
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

  const handleCheck = async () => {
    setCheckStatus('checking');
    setCheckMsg('');
    try {
      const payload = buildPayload();
      const response = await fetch(`${API_BASE}/api/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      broadcastPayload('/api/check', payload, data);

      if (response.ok) {
        setCheckStatus('ok');
        setCheckMsg('Compatible ✓');
        setNodeShapes(data.node_shapes ?? {});
        // Back-fill inferred shapes and auto-resolved params onto each node
        setNodes((nds) =>
          nds.map((n) => {
            const shapes    = data.node_shapes?.[n.id];
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
            if (newParams) Object.assign(updatedValues, newParams);
            return { ...n, data: { ...n.data, paramValues: updatedValues, inferredShapes: shapes } };
          })
        );
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
    if (confirm('Clear the canvas?')) {
      setNodes([]);
      setEdges([]);
      setShapeErrorNodeId(null);
      setNodeShapes({});
      setGeneratedCode('');
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
      style={{ height: 40, background: '#1e1e1e', borderBottom: '1px solid #363636' }}
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
          suppressHydrationWarning
          onClick={handleCheck}
          disabled={checkStatus === 'checking'}
          className={`flex items-center gap-1.5 text-[11px] font-medium transition-colors px-2 py-1 rounded-sm border ${checkBtnClass}`}
        >
          {checkStatus === 'checking' ? 'Checking...' : 'Check Shapes'}
        </button>
        <button
          suppressHydrationWarning
          onClick={handleReset}
          className="flex items-center gap-1.5 text-[11px] text-[#888] hover:text-[#e2e2e2] transition-colors px-2 py-1 rounded-sm hover:bg-[#2a2a2a] ml-1"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
        <button
          suppressHydrationWarning
          onClick={handleExport}
          disabled={compiling}
          className="flex items-center gap-1.5 text-[11px] font-medium text-white bg-[#2d8cf0] hover:bg-[#3a97f5] disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-3 py-1.5 rounded-sm"
        >
          <Play className="w-3 h-3 fill-white" />
          {compiling ? 'Compiling…' : 'Export PyTorch'}
        </button>
        {process.env.NODE_ENV === 'development' && (
          <button
            onClick={() => window.open('/dev/payloads', '_blank')}
            className="flex items-center gap-1.5 text-[11px] text-[#888] hover:text-[#e2e2e2] transition-colors px-2 py-1.5 rounded-sm hover:bg-[#2a2a2a] ml-1 border border-[#3a3a3a]"
            title="Open Dev Tools"
          >
            <Bug className="w-3.5 h-3.5" />
            Debug
          </button>
        )}
      </div>
    </header>
  );
}
