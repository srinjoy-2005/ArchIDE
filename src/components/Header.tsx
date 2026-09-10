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

import { useState, useRef } from 'react';
import { useReactFlow, useNodes, useEdges } from '@xyflow/react';
import { useEditorStore, useVFSStore } from '../lib/store';
import { Layers, Play, RotateCcw, Bug, Save, Upload, FolderOpen } from 'lucide-react';
import { API_BASE } from '../lib/constants';
import { resolveFilePath, getVFSFilePath } from '../lib/utils';

export function Header() {
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();
  const setGeneratedCode  = useEditorStore((s) => s.setGeneratedCode);
  const setShapeErrorNodeId = useEditorStore((s) => s.setShapeErrorNodeId);
  const setNodeShapes     = useEditorStore((s) => s.setNodeShapes);

  const folders           = useVFSStore((s) => s.folders);
  const handleCompiledFiles = useVFSStore((s) => s.handleCompiledFiles);
  const files             = useVFSStore((s) => s.files);
  const activeFileId      = useVFSStore((s) => s.activeFileId);
  const entryFileId       = useVFSStore((s) => s.entryFileId);
  const isSaving          = useVFSStore((s) => s.isSaving);
  const setIsSaving       = useVFSStore((s) => s.setIsSaving);
  const overwriteFilesFromVFS = useVFSStore((s) => s.overwriteFilesFromVFS);
  const restoreVFSFromFiles = useVFSStore((s) => s.restoreVFSFromFiles);
  const graphsFolderId    = useVFSStore((s) => s.graphsFolderId);

  const [compiling,    setCompiling]    = useState(false);
  const [checkStatus,  setCheckStatus]  = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');
  const [checkMsg,     setCheckMsg]     = useState('');

  // Save State path — persisted in localStorage
  const [savePath, setSavePath] = useState<string>(() =>
    (typeof window !== 'undefined' && localStorage.getItem('archide_save_path')) || 'workspace'
  );
  const [savePathEditing, setSavePathEditing] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'ok' | 'error'>('idle');

  // Load State path — persisted in localStorage
  const [loadPath, setLoadPath] = useState<string>(() =>
    (typeof window !== 'undefined' && localStorage.getItem('archide_load_path')) || 'workspace'
  );
  const [loadPathEditing, setLoadPathEditing] = useState(false);
  const [loadStatus, setLoadStatus] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');

  const savePathRef = useRef<HTMLInputElement>(null);
  const loadPathRef = useRef<HTMLInputElement>(null);

  /** Save State: mirror exact browser VFS files to disk at savePath, pruning redundant files */
  const handleSaveState = async () => {
    setSaveStatus('saving');
    // Enable live auto-save from this point forward
    setIsSaving(true);
    try {
      const currentFiles = useVFSStore.getState().files;
      const currentFolders = useVFSStore.getState().folders;
      const filesMap: Record<string, any> = {};

      for (const f of currentFiles) {
        const vfsRelPath = getVFSFilePath(f, currentFolders);
        if (f.fileType === 'code' || f.name.endsWith('.py') || f.name.endsWith('.toml')) {
          filesMap[vfsRelPath] = f.compiledCode || '';
        } else {
          const isCurrentActive = f.id === activeFileId;
          const name = f.name.replace(/\.[^/.]+$/, '');
          filesMap[vfsRelPath] = {
            name,
            nodes: isCurrentActive ? nodes : f.nodes,
            edges: isCurrentActive ? edges : f.edges,
            variables: f.variables || [],
          };
        }
      }

      const res = await fetch(`${API_BASE}/api/state/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dir: savePath,
          files: filesMap,
        }),
      });
      if (res.ok) {
        setSaveStatus('ok');
        setTimeout(() => setSaveStatus('idle'), 2500);
      } else {
        setSaveStatus('error');
        setTimeout(() => setSaveStatus('idle'), 3000);
      }
    } catch {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };

  /** Load State: read exact files from backend and mirror into browser VFS */
  const handleLoadState = async () => {
    setLoadStatus('loading');
    try {
      const res = await fetch(`${API_BASE}/api/state/load?dir=${encodeURIComponent(loadPath)}`);
      if (!res.ok) {
        setLoadStatus('error');
        setTimeout(() => setLoadStatus('idle'), 3000);
        return;
      }
      const data = await res.json();
      if (data.files && Object.keys(data.files).length > 0) {
        restoreVFSFromFiles(data.files);
      } else {
        if (data.graphs) overwriteFilesFromVFS(data.graphs);
        if (data.python) handleCompiledFiles(data.python);
      }

      // Sync active graph into uncontrolled React Flow canvas
      const updatedActive = useVFSStore.getState().files.find(
        (f) => f.id === useVFSStore.getState().activeFileId
      );
      if (updatedActive && updatedActive.fileType !== 'code') {
        setNodes(updatedActive.nodes || []);
        setEdges(updatedActive.edges || []);
      }

      setLoadStatus('ok');
      setTimeout(() => setLoadStatus('idle'), 2500);
    } catch {
      setLoadStatus('error');
      setTimeout(() => setLoadStatus('idle'), 3000);
    }
  };

  /** Build the multi-graph JSON payload for /api/compile and /api/check */
  const buildPayload = () => {
    const currentNodes = getNodes();
    const currentEdges = getEdges();
    const graphs: Record<string, any> = {};
    const file_paths: Record<string, string> = {};
    
    // Pass 1: Build ID to Path mapping
    const idToPath: Record<string, string> = {};
    for (const f of files) {
      if (f.fileType === 'code' || f.name.endsWith('.py') || f.name.endsWith('.toml')) continue;
      idToPath[f.id] = resolveFilePath(f, folders, graphsFolderId);
    }

    // Pass 2: Build Graphs payload
    for (const f of files) {
      if (f.fileType === 'code' || f.name.endsWith('.py') || f.name.endsWith('.toml')) continue;

      const fullPath = idToPath[f.id];
      file_paths[fullPath] = fullPath;

      const isCurrent = f.id === activeFileId;
      const nList = isCurrent ? currentNodes : f.nodes;
      const eList = isCurrent ? currentEdges : f.edges;

      const variables = f.variables || [];

      graphs[fullPath] = {
        name: f.name.replace(/\.[^/.]+$/, ''),
        variables,
        parameters: [],
        nodes: nList.map((n) => {
          let cid = (n.data.custom_module_id as string) || '';
          if (cid && idToPath[cid]) {
            cid = idToPath[cid];
          }
          return {
            id: n.id,
            position: n.position || { x: 100, y: 100 },
            data: {
              block_id: n.data.block_id || '',
              label: n.data.label,
              is_functional: n.data.is_functional || false,
              paramValues: n.data.paramValues || {},
              varName: (n.data.varName as string) || '',
              custom_module_id: cid,
            },
          };
        }),
        edges: eList.map((e) => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle || '',
          target: e.target,
          targetHandle: e.targetHandle || '',
        })),
      };
    }

    const entryFile = files.find((f) => f.id === entryFileId);
    const mainGraphPath = entryFile ? resolveFilePath(entryFile, folders, graphsFolderId) : entryFileId;

    return { main_graph_id: mainGraphPath, graphs, file_paths };
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
    setCheckStatus('checking');
    setCheckMsg('');
    setGeneratedCode('# Compiling via Python Backend Engine...');
    try {
      const payload = buildPayload();

      // ── Guard: block compile if any param has a dangling @var: binding (null) ──
      const danglingNodes: string[] = [];
      for (const [graphId, graphData] of Object.entries(payload.graphs as any)) {
        for (const node of (graphData as any).nodes || []) {
          const pv = node.data?.paramValues || {};
          const hasDangling = Object.values(pv).some((v) => v === null || v === undefined);
          if (hasDangling) {
            danglingNodes.push(`${node.data?.label || node.id} (in ${graphId})`);
          }
        }
      }
      if (danglingNodes.length > 0) {
        setGeneratedCode(
          `# ⛔ Compile blocked: ${danglingNodes.length} node(s) have unresolved parameters or 'Not Set' fields.\n` +
          `# Please rebind, fill in, or reset the following nodes:\n` +
          danglingNodes.map((n) => `#   • ${n}`).join('\n')
        );
        setCompiling(false);
        return;
      }

      const response = await fetch(`${API_BASE}/api/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      broadcastPayload('/api/compile', payload, data);

      if (response.ok) {
        setCheckStatus('ok');
        setCheckMsg('Compatible ✓');
        setGeneratedCode('# Compiled successfully. Check the python/ folder.');
        setShapeErrorNodeId(null);
        handleCompiledFiles(data.files || {});
        
        setNodeShapes(data.node_shapes ?? {});
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
            return { ...n, data: { ...n.data, paramValues: updatedValues, inferredShapes: shapes, inferredParams: newParams } };
          })
        );
      } else {
        setCheckStatus('error');
        // Clear all previous inferred shapes/params on error
        setNodeShapes({});
        setNodes((nds) =>
          nds.map((n) => {
            const updatedValues = { ...(n.data.paramValues as any) };
            delete updatedValues['output_shape'];
            delete updatedValues['input_shape'];
            return {
              ...n,
              data: { ...n.data, paramValues: updatedValues, inferredShapes: undefined, inferredParams: undefined },
            };
          })
        );

        if (data.detail?.error === 'ShapeMismatch') {
          setShapeErrorNodeId(data.detail.node_id);
          const msg = `# ❌ Shape Mismatch at "${data.detail.node_label}":\n# ${data.detail.message}`;
          setGeneratedCode(msg);
          setCheckMsg(data.detail.message);
        } else {
          setGeneratedCode(`# ❌ Compiler Error:\n# ${JSON.stringify(data.detail)}`);
          setCheckMsg('Compiler Error');
        }
      }
    } catch (err: any) {
      setCheckStatus('error');
      setCheckMsg(`Cannot reach backend: ${err.message}`);
      // Clear shapes on network error too
      setNodeShapes({});
      setNodes((nds) =>
        nds.map((n) => {
          const updatedValues = { ...(n.data.paramValues as any) };
          delete updatedValues['output_shape'];
          delete updatedValues['input_shape'];
          return {
            ...n,
            data: { ...n.data, paramValues: updatedValues, inferredShapes: undefined, inferredParams: undefined },
          };
        })
      );
      setGeneratedCode(`# ❌ Network Error:\n# Could not reach backend: ${err.message}`);
    } finally {
      setCompiling(false);
    }
  };

  const handleCheck = async () => {
    setCheckStatus('checking');
    setCheckMsg('');
    setShapeErrorNodeId(null);
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
            return { ...n, data: { ...n.data, paramValues: updatedValues, inferredShapes: shapes, inferredParams: newParams } };
          })
        );
      } else {
        setCheckStatus('error');
        // Clear all previous inferred shapes/params on error
        setNodeShapes({});
        setNodes((nds) =>
          nds.map((n) => {
            const updatedValues = { ...(n.data.paramValues as any) };
            delete updatedValues['output_shape'];
            delete updatedValues['input_shape'];
            return {
              ...n,
              data: { ...n.data, paramValues: updatedValues, inferredShapes: undefined, inferredParams: undefined },
            };
          })
        );
        if (data.detail?.error === 'ShapeMismatch') {
          setShapeErrorNodeId(data.detail.node_id);
          setCheckMsg(data.detail.message);
        } else {
          const errMsg = typeof data.detail === 'string' ? data.detail : data.detail?.message;
          setCheckMsg(errMsg || 'Shape mismatch detected.');
        }
      }
    } catch (err: any) {
      setCheckStatus('error');
      // Clear shapes on network error too
      setNodeShapes({});
      setNodes((nds) =>
        nds.map((n) => {
          const updatedValues = { ...(n.data.paramValues as any) };
          delete updatedValues['output_shape'];
          delete updatedValues['input_shape'];
          return {
            ...n,
            data: { ...n.data, paramValues: updatedValues, inferredShapes: undefined, inferredParams: undefined },
          };
        })
      );
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
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            compiling
              ? 'bg-[#2a2a2a] text-[#888]'
              : 'bg-[#2d8cf0] hover:bg-[#3b9cff] text-white'
          }`}
        >
          {compiling ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {compiling ? 'Compiling...' : 'Compile'}
        </button>

        {/* Save State button + path input */}
        <div className="relative flex items-center gap-0.5 ml-1">
          {savePathEditing ? (
            <input
              ref={savePathRef}
              autoFocus
              value={savePath}
              onChange={(e) => {
                setSavePath(e.target.value);
                localStorage.setItem('archide_save_path', e.target.value);
              }}
              onBlur={() => setSavePathEditing(false)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === 'Escape') setSavePathEditing(false); }}
              className="text-[10px] font-mono bg-[#1a1a1a] border border-[#505050] text-[#d4d4d4] rounded-sm px-1.5 py-1 w-44 outline-none"
              placeholder="e.g. workspace or /abs/path"
            />
          ) : (
            <>
              <button
                suppressHydrationWarning
                onClick={handleSaveState}
                disabled={saveStatus === 'saving'}
                title={`Save to: ${savePath}`}
                className={`flex items-center gap-1.5 text-[11px] transition-colors px-2 py-1.5 rounded-sm border ${
                  saveStatus === 'ok'
                    ? 'text-[#4ade80] border-[#4ade80]/50 bg-[#4ade80]/10'
                    : saveStatus === 'error'
                    ? 'text-[#e54545] border-[#e54545]/50 bg-[#e54545]/10'
                    : saveStatus === 'saving'
                    ? 'text-[#555] border-[#363636] bg-[#1e1e1e] cursor-not-allowed'
                    : 'text-[#888] hover:text-[#e2e2e2] hover:bg-[#2a2a2a] border-[#3a3a3a]'
                }`}
              >
                <Save className="w-3.5 h-3.5" />
                {saveStatus === 'saving' ? 'Saving…' : saveStatus === 'ok' ? 'Saved ✓' : 'Save State'}
              </button>
              <button
                onClick={() => setSavePathEditing(true)}
                title="Edit save path"
                className="p-1 text-[#555] hover:text-[#888] transition-colors"
              >
                <FolderOpen className="w-3 h-3" />
              </button>
            </>
          )}
        </div>

        {/* Load State button + path input */}
        <div className="relative flex items-center gap-0.5">
          {loadPathEditing ? (
            <input
              ref={loadPathRef}
              autoFocus
              value={loadPath}
              onChange={(e) => {
                setLoadPath(e.target.value);
                localStorage.setItem('archide_load_path', e.target.value);
              }}
              onBlur={() => setLoadPathEditing(false)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === 'Escape') setLoadPathEditing(false); }}
              className="text-[10px] font-mono bg-[#1a1a1a] border border-[#505050] text-[#d4d4d4] rounded-sm px-1.5 py-1 w-44 outline-none"
              placeholder="e.g. workspace or /abs/path"
            />
          ) : (
            <>
              <button
                suppressHydrationWarning
                onClick={handleLoadState}
                disabled={loadStatus === 'loading'}
                title={`Load from: ${loadPath}`}
                className={`flex items-center gap-1.5 text-[11px] transition-colors px-2 py-1.5 rounded-sm border ${
                  loadStatus === 'ok'
                    ? 'text-[#4ade80] border-[#4ade80]/50 bg-[#4ade80]/10'
                    : loadStatus === 'error'
                    ? 'text-[#e54545] border-[#e54545]/50 bg-[#e54545]/10'
                    : loadStatus === 'loading'
                    ? 'text-[#555] border-[#363636] bg-[#1e1e1e] cursor-not-allowed'
                    : 'text-[#888] hover:text-[#e2e2e2] hover:bg-[#2a2a2a] border-[#3a3a3a]'
                }`}
              >
                <Upload className="w-3.5 h-3.5" />
                {loadStatus === 'loading' ? 'Loading…' : loadStatus === 'ok' ? 'Loaded ✓' : 'Load State'}
              </button>
              <button
                onClick={() => setLoadPathEditing(true)}
                title="Edit load path"
                className="p-1 text-[#555] hover:text-[#888] transition-colors"
              >
                <FolderOpen className="w-3 h-3" />
              </button>
            </>
          )}
        </div>

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
