import React, { memo } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import { Trash2, Copy, AlertTriangle } from 'lucide-react';
import { useEditorStore } from '../lib/store';

// ─── Category accent colors (subtle left-strip only) ─────────────────────────
const CATEGORY_COLORS: Record<string, string> = {
  input:   '#4ade80', // green
  output:  '#f87171', // red
  conv:    '#60a5fa', // blue
  linear:  '#818cf8', // indigo
  relu:    '#a78bfa', // violet
  softmax: '#c084fc', // purple
  add:     '#fb923c', // orange
  split:   '#34d399', // emerald
  default: '#6b7280', // gray
};

function getAccentColor(label: string = ''): string {
  const l = label.toLowerCase();
  for (const key of Object.keys(CATEGORY_COLORS)) {
    if (l.includes(key)) return CATEGORY_COLORS[key];
  }
  return CATEGORY_COLORS.default;
}

function getParamSummary(paramValues: Record<string, any> = {}): string {
  const parts: string[] = [];
  if (paramValues.in_channels !== undefined && paramValues.out_channels !== undefined) {
    parts.push(`${paramValues.in_channels}→${paramValues.out_channels}ch`);
  } else if (paramValues.in_features !== undefined && paramValues.out_features !== undefined) {
    parts.push(`${paramValues.in_features}→${paramValues.out_features}`);
  }
  if (paramValues.kernel_size !== undefined) parts.push(`k${paramValues.kernel_size}`);
  if (paramValues.stride !== undefined && paramValues.stride !== 1) parts.push(`s${paramValues.stride}`);
  if (paramValues.dim !== undefined) parts.push(`dim:${paramValues.dim}`);
  if (paramValues.chunks !== undefined) parts.push(`×${paramValues.chunks}`);
  if (paramValues.num_inputs !== undefined) parts.push(`n:${paramValues.num_inputs}`);
  return parts.join('  ');
}

const CustomNode = ({ id, data, isConnectable }: any) => {
  const { setNodes } = useReactFlow();
  const shapeErrorNodeId = useEditorStore((s) => s.shapeErrorNodeId);

  const inputs = data.inputs || [{ id: 'in', name: 'Input' }];
  const outputs = data.outputs || [{ id: 'out', name: 'Output' }];
  const paramValues = data.paramValues || {};
  const paramSummary = getParamSummary(paramValues);
  const accent = getAccentColor(data.label);
  const isError = shapeErrorNodeId === id;

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setNodes((nds) => nds.filter((n) => n.id !== id));
  };

  const handleDuplicate = (e: React.MouseEvent) => {
    e.stopPropagation();
    setNodes((nds) => {
      const src = nds.find((n) => n.id === id);
      if (!src) return nds;
      return [
        ...nds,
        {
          ...src,
          id: `node_${Date.now()}`,
          position: { x: src.position.x + 32, y: src.position.y + 32 },
          selected: false,
        },
      ];
    });
  };

  // Compute the effective variable name shown on the card
  const rawVar = ((data.varName as string) || "").trim();
  const effectiveVar = rawVar
    ? rawVar.toLowerCase().replace(/[^a-z0-9_]/g, "_").replace(/^[^a-z_]/, "x_$&")
    : null;

  return (
    <div
      style={{
        borderColor: isError ? '#e54545' : '#3a3a3a',
        boxShadow: isError ? '0 0 0 1px rgba(229,69,69,0.4), 0 4px 12px rgba(0,0,0,0.5)' : '0 4px 12px rgba(0,0,0,0.5)',
      }}
      className="group relative bg-[#252525] border rounded-[3px] min-w-[164px] flex transition-all duration-150 hover:border-[#505050]"
    >
      {/* Left category accent strip */}
      <div
        className="w-[3px] rounded-l-[3px] flex-shrink-0"
        style={{ background: isError ? '#e54545' : accent }}
      />

      {/* Error badge */}
      {isError && (
        <div className="absolute -top-6 left-0 flex items-center gap-1 bg-[#e54545] text-white text-[10px] font-medium px-2 py-0.5 rounded-sm shadow-md whitespace-nowrap z-20">
          <AlertTriangle className="w-2.5 h-2.5" />
          Shape mismatch
        </div>
      )}

      {/* Quick action toolbar (visible on hover) */}
      <div className="absolute -top-[22px] right-0 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-0.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded-sm px-1 py-0.5 shadow-md z-20">
        <button
          onClick={handleDuplicate}
          title="Duplicate"
          className="p-0.5 text-[#888] hover:text-[#e2e2e2] transition-colors"
        >
          <Copy className="w-3 h-3" />
        </button>
        <button
          onClick={handleDelete}
          title="Delete"
          className="p-0.5 text-[#888] hover:text-[#e54545] transition-colors"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>

      {/* Left input handles */}
      <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-evenly -ml-[5px] z-10">
        {inputs.map((inp: any) => (
          <div key={inp.id} className="relative group/h flex items-center">
            <Handle
              type="target"
              position={Position.Left}
              id={inp.id}
              isConnectable={isConnectable}
              className="!w-2.5 !h-2.5 !bg-[#505050] !border-[1.5px] !border-[#1e1e1e] !relative !transform-none hover:!bg-[#2d8cf0] transition-colors"
            />
            <span className="absolute left-4 pointer-events-none text-[10px] text-[#aaa] bg-[#1e1e1e] border border-[#3a3a3a] px-1.5 py-px rounded-sm whitespace-nowrap opacity-0 group-hover/h:opacity-100 transition-opacity z-30">
              {inp.name}
            </span>
          </div>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-1 px-3 py-2 flex flex-col gap-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[12px] font-semibold text-[#e2e2e2] truncate leading-none">
            {data.label}
          </span>
          <span className="text-[9px] font-mono text-[#666] uppercase tracking-wider shrink-0">
            {data.is_functional ? 'func' : 'layer'}
          </span>
        </div>

        {/* Effective variable name */}
        <span className="text-[10px] font-mono truncate leading-none" style={{ color: effectiveVar ? '#2d8cf0' : '#555' }}>
          {effectiveVar ? `→ ${effectiveVar}` : `→ auto`}
        </span>

        {paramSummary ? (
          <span className="text-[10px] font-mono text-[#888] truncate leading-none">
            {paramSummary}
          </span>
        ) : null}
      </div>

      {/* Right output handles */}
      <div className="absolute right-0 top-0 bottom-0 flex flex-col justify-evenly -mr-[5px] z-10">
        {outputs.map((out: any) => (
          <div key={out.id} className="relative group/h flex items-center">
            <span className="absolute right-4 pointer-events-none text-[10px] text-[#aaa] bg-[#1e1e1e] border border-[#3a3a3a] px-1.5 py-px rounded-sm whitespace-nowrap opacity-0 group-hover/h:opacity-100 transition-opacity z-30">
              {out.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={out.id}
              isConnectable={isConnectable}
              className="!w-2.5 !h-2.5 !bg-[#505050] !border-[1.5px] !border-[#1e1e1e] !relative !transform-none hover:!bg-[#2d8cf0] transition-colors"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default memo(CustomNode);
