"use client";

/**
 * src/components/PropertiesPanel.tsx
 *
 * The "Inspector" tab content for the right panel. Contains three sub-components:
 *
 * - ParamInput: a single labelled form field for editing a block parameter.
 * - ModelSummaryDashboard: shown when no node is selected — displays graph stats,
 *   a canvas usage guide, and a "Load ConvNet Pipeline" quick-start button.
 * - PropertiesPanel: shown when a node is selected — renders its label, output
 *   variable name, connected input tensors, inferred shapes, and hyperparameters
 *   (split into basic and collapsible advanced sections).
 */

import { useReactFlow, useNodes, useEdges } from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import { Brain, ChevronRight } from 'lucide-react';
import { PARAM_TYPE_HANDLERS, type ParamTypeName } from '../lib/paramTypes';

// ─── ParamInput ───────────────────────────────────────────────────────────────

function ParamInput({
  param,
  value,
  onChange,
}: {
  param: any;
  value: any;
  onChange: (name: string, value: any) => void;
}) {
  const handler = PARAM_TYPE_HANDLERS[param.type as ParamTypeName] || PARAM_TYPE_HANDLERS.string;
  
  const isAutoInferEnabled = param.auto_infer;
  const isChecked = isAutoInferEnabled && (value === -1 || value === "?" || value === -1.0);
  const isReadOnly = param.read_only || isChecked;

  const inputClass = isReadOnly
    ? 'w-full bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-2 py-1.5 text-[12px] text-[#555] font-mono cursor-not-allowed'
    : 'w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] px-2 py-1.5 text-[12px] text-[#e2e2e2] font-mono outline-none transition-colors';

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (isReadOnly) return;
    const rawVal = handler.inputType === 'checkbox' ? e.target.checked : e.target.value;
    
    if (isAutoInferEnabled && String(rawVal) === '-1') {
      onChange(param.name, -1);
      return;
    }
    
    if (handler.isValid(rawVal as any)) {
      onChange(param.name, handler.coerce(rawVal as any));
    } else {
      onChange(param.name, rawVal); 
    }
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      onChange(param.name, param.type === 'string' ? '?' : -1);
    } else {
      onChange(param.name, param.default ?? 1);
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-[#aaa] capitalize flex items-center gap-1.5">
          {param.name.replace(/_/g, ' ')}
          {param.read_only && (
            <span className="text-[9px] font-mono text-[#555] bg-[#252525] border border-[#363636] px-1 py-px rounded-sm">
              inferred
            </span>
          )}
        </label>
        <div className="flex items-center gap-2">
          {isAutoInferEnabled && (
            <label className="flex items-center gap-1 text-[9px] text-[#888] cursor-pointer">
              <input 
                type="checkbox" 
                checked={isChecked} 
                onChange={handleCheckboxChange}
                className="w-2.5 h-2.5 accent-[#2d8cf0]"
              />
              Auto-Infer
            </label>
          )}
          <span className="text-[9px] font-mono text-[#555]">{param.type}</span>
        </div>
      </div>
      
      {handler.inputType === 'checkbox' ? (
         <label className="flex items-center gap-2 text-[12px] text-[#e2e2e2]">
           <input type="checkbox" checked={!!value} onChange={handleChange} disabled={isReadOnly} />
           {value ? 'True' : 'False'}
         </label>
      ) : (
        <input
          type={handler.inputType}
          step={handler.step}
          className={inputClass}
          value={value ?? param.default}
          readOnly={isReadOnly}
          disabled={isReadOnly}
          title={param.description || ''}
          onChange={handleChange}
        />
      )}
    </div>
  );
}

// ─── ModelSummaryDashboard ────────────────────────────────────────────────────

function ModelSummaryDashboard() {
  const { setNodes, setEdges } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();

  const loadStarter = () => {
    const n: Node[] = [
      { id: 's1', type: 'custom', position: { x: 60,   y: 160 }, data: { block_id: 'input',   label: 'Input',   is_functional: true,  params: [{ name: 'shape', type: 'string', default: '(1,3,224,224)' }], paramValues: { shape: '(1,3,224,224)' }, inputs: [], outputs: [{ id: 'out', name: 'Output' }] } },
      { id: 's2', type: 'custom', position: { x: 270,  y: 160 }, data: { block_id: 'conv2d',  label: 'Conv2D',  is_functional: false, params: [{ name: 'in_channels', type: 'int', default: 3 }, { name: 'out_channels', type: 'int', default: 16 }, { name: 'kernel_size', type: 'int', default: 3 }], paramValues: { in_channels: 3, out_channels: 16, kernel_size: 3 }, inputs: [{ id: 'in', name: 'Input' }], outputs: [{ id: 'out', name: 'Output' }] } },
      { id: 's3', type: 'custom', position: { x: 480,  y: 160 }, data: { block_id: 'relu',    label: 'ReLU',    is_functional: false, params: [], paramValues: {}, inputs: [{ id: 'in', name: 'Input' }], outputs: [{ id: 'out', name: 'Output' }] } },
      { id: 's4', type: 'custom', position: { x: 670,  y: 160 }, data: { block_id: 'flatten', label: 'Flatten', is_functional: true,  params: [{ name: 'start_dim', type: 'int', default: 1 }, { name: 'end_dim', type: 'int', default: -1 }], paramValues: { start_dim: 1, end_dim: -1 }, inputs: [{ id: 'in', name: 'Input' }], outputs: [{ id: 'out', name: 'Output' }] } },
      { id: 's5', type: 'custom', position: { x: 880,  y: 160 }, data: { block_id: 'linear',  label: 'Linear',  is_functional: false, params: [{ name: 'in_features', type: 'int', default: 128 }, { name: 'out_features', type: 'int', default: 10 }], paramValues: { in_features: -1, out_features: 10 }, inputs: [{ id: 'in', name: 'Input' }], outputs: [{ id: 'out', name: 'Output' }] } },
      { id: 's6', type: 'custom', position: { x: 1090, y: 160 }, data: { block_id: 'output',  label: 'Output',  is_functional: true,  params: [], paramValues: {}, inputs: [{ id: 'in', name: 'Return Value' }], outputs: [] } },
    ];
    const e: Edge[] = [
      { id: 'e12', source: 's1', sourceHandle: 'out', target: 's2', targetHandle: 'in', type: 'tensor' },
      { id: 'e23', source: 's2', sourceHandle: 'out', target: 's3', targetHandle: 'in', type: 'tensor' },
      { id: 'e34', source: 's3', sourceHandle: 'out', target: 's4', targetHandle: 'in', type: 'tensor' },
      { id: 'e45', source: 's4', sourceHandle: 'out', target: 's5', targetHandle: 'in', type: 'tensor' },
      { id: 'e56', source: 's5', sourceHandle: 'out', target: 's6', targetHandle: 'in', type: 'tensor' },
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

      {/* Canvas guide */}
      <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] p-3 text-[11px] text-[#888] space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-[#555] mb-2">Canvas Guide</div>
        <div className="flex justify-between"><span>Drag layer</span><span className="text-[#aaa] font-mono">→ drop on grid</span></div>
        <div className="flex justify-between"><span>Connect ports</span><span className="text-[#aaa] font-mono">→ drag handle</span></div>
        <div className="flex justify-between"><span>Delete</span><span className="text-[#aaa] font-mono">Backspace / Del</span></div>
        <div className="flex justify-between"><span>Select node</span><span className="text-[#aaa] font-mono">→ click</span></div>
      </div>

      {/* Quick start */}
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

// ─── PropertiesPanel ──────────────────────────────────────────────────────────

export function PropertiesPanel() {
  const { setNodes } = useReactFlow();
  const nodes = useNodes();
  const edges = useEdges();
  const selectedNode = nodes.find((n) => n.selected) || null;

  // Handles param changes including dynamic input/output resizing for variadic blocks
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

  // Fallback: show dashboard when nothing is selected
  if (!selectedNode) return <ModelSummaryDashboard />;

  const params       = (selectedNode.data.params as any[]) || [];
  const paramValues  = (selectedNode.data.paramValues as any) || {};
  const shapeParams  = params.filter((p: any) => p.section === 'shape');
  const basicParams  = params.filter((p: any) => !p.section || p.section === 'basic');
  const advancedParams = params.filter((p: any) => p.section === 'advanced');
  const inferredShapes = (selectedNode.data.inferredShapes as Record<string, any>) || {};
  const incomingEdges  = edges.filter((e) => e.target === selectedNode.id);

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
              value={(selectedNode.data.label as string) || ''}
              onChange={(e) =>
                setNodes((nds) =>
                  nds.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, label: e.target.value } } : n)
                )
              }
            />
          </div>
          {/* Output variable name */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <label className="text-[10px] text-[#888]">Output variable name</label>
              <span className="text-[9px] font-mono text-[#555]">identifier</span>
            </div>
            <input
              type="text"
              spellCheck={false}
              placeholder={`auto: x_${selectedNode.id.replace(/-/g, '_')}`}
              className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#2d8cf0] rounded-[3px] px-2 py-1.5 text-[12px] text-[#e2e2e2] font-mono outline-none transition-colors placeholder-[#444]"
              value={(selectedNode.data.varName as string) || ''}
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

      {/* Connected incoming tensors */}
      {incomingEdges.length > 0 && (
        <div className="flex flex-col gap-2 border-b border-[#363636] pb-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-[#555]">Connected Inputs</span>
            <span className="text-[9px] font-mono text-[#2d8cf0] bg-[#1e1e1e] border border-[#363636] px-1.5 py-px rounded-sm">
              {incomingEdges.length} {incomingEdges.length === 1 ? 'tensor' : 'tensors'}
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            {incomingEdges.map((e, idx) => {
              const srcNode = nodes.find((n) => n.id === e.source);
              const srcLabel = (srcNode?.data?.label as string) || e.source;
              const srcVar = (srcNode?.data?.varName as string) || `x_${e.source.replace(/-/g, '_')}`;
              return (
                <div key={e.id} className="flex items-center justify-between bg-[#1e1e1e] border border-[#3a3a3a] px-2.5 py-1.5 rounded-[3px] text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono text-[#666]">#{idx + 1}</span>
                    <span className="text-[#e2e2e2] font-medium">{srcLabel}</span>
                  </div>
                  <span className="font-mono text-[10px] text-[#2d8cf0]">{srcVar}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Inferred shapes from last /api/check */}
      {Object.keys(inferredShapes).length > 0 && (
        <div className="flex flex-col gap-3 border-b border-[#363636] pb-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Tensor Shapes</div>
          {Object.entries(inferredShapes).map(([port, shape]) => (
            <div key={port} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <label className="text-[11px] text-[#aaa] capitalize flex items-center gap-1.5">
                  Port: {port.replace(/_/g, ' ')}
                  <span className="text-[9px] font-mono text-[#555] bg-[#252525] border border-[#363636] px-1 py-px rounded-sm">inferred</span>
                </label>
              </div>
              <input type="text" className="w-full bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-2 py-1.5 text-[12px] text-[#555] font-mono cursor-not-allowed" value={JSON.stringify(shape)} readOnly disabled />
            </div>
          ))}
        </div>
      )}

      {/* Fallback for legacy shape params */}
      {shapeParams.length > 0 && Object.keys(inferredShapes).length === 0 && (
        <div className="flex flex-col gap-3 border-b border-[#363636] pb-3">
          <div className="text-[10px] uppercase tracking-wider text-[#555]">Tensor Shapes</div>
          {shapeParams.map((p: any) => (
            <ParamInput key={p.name} param={p} value={paramValues[p.name]} onChange={handleParamChange} />
          ))}
        </div>
      )}

      {/* Basic hyperparameters */}
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
