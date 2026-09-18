/**
 * src/components/CustomNode.tsx
 *
 * This component defines the visual representation of a PyTorch block (layer/function)
 * on the React Flow canvas. It includes:
 * - Dynamic category accent colors
 * - Input/Output handles (`<Handle>`) for connections
 * - A hover tooltip that displays tensor shape inferences for the node's ports
 * - A quick action toolbar for deleting or duplicating the node
 */
import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, useReactFlow, useUpdateNodeInternals } from '@xyflow/react';
import { Trash2, Copy, AlertTriangle, ArrowUpDown, ArrowRightLeft, FlipHorizontal } from 'lucide-react';
import { useEditorStore } from '../lib/store';

// ─── Category accent colors ───────────────────────────────────────────────────
const CATEGORY_COLORS: Record<string, string> = {
  input:   '#4ade80',
  output:  '#f87171',
  conv:    '#60a5fa',
  linear:  '#818cf8',
  relu:    '#a78bfa',
  softmax: '#c084fc',
  gelu:    '#c084fc',
  tanh:    '#c084fc',
  sigmoid: '#c084fc',
  add:     '#fb923c',
  split:   '#34d399',
  pool:    '#2dd4bf',
  norm:    '#facc15',
  batch:   '#facc15',
  layer:   '#facc15',
  dropout: '#94a3b8',
  reshape: '#f472b6',
  default: '#6b7280',
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

// Format a shape value:
//   number[]   → "1×64×28×28"       (single tensor — used for output / single-input ports)
//   number[][] → "[1×3×224×224] × [1×3×224×224]"  (multi-input port, one bracket per tensor)
function fmtShape(shape: number[] | number[][] | undefined): string {
  if (!shape || (shape as any[]).length === 0) return '';
  // Detect nested array (list-input port stores an array of shapes)
  if (Array.isArray((shape as any[])[0])) {
    return (shape as number[][]).map((s) => `[${s.join('\u00d7')}]`).join(' \u00d7 ');
  }
  return (shape as number[]).join('\u00d7');
}

const CustomNode = ({ id, data, isConnectable }: any) => {
  const { setNodes, setEdges } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const shapeErrorNodeId = useEditorStore((s) => s.shapeErrorNodeId);
  const nodeShapes = useEditorStore((s) => s.nodeShapes);
  const [hovered, setHovered] = useState(false);

  const inputs = Array.isArray(data.inputs) && data.inputs.length > 0
    ? data.inputs
    : (data.block_id === 'input' ? [] : [{ id: 'in', name: 'Input' }]);

  const outputs = Array.isArray(data.outputs) && data.outputs.length > 0
    ? data.outputs
    : (data.block_id === 'output' ? [] : [{ id: 'out', name: 'Output' }]);

  const paramValues = data.paramValues || {};
  const paramSummary = getParamSummary(paramValues);
  const accent = getAccentColor(data.label);
  const isError = shapeErrorNodeId === id;

  const portLayout = data.portLayout || {};
  const isVertical = portLayout.orientation === 'vertical';
  const isFlipped = Boolean(portLayout.flipped);

  // Update react flow internal coordinates when port layout changes
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, isVertical, isFlipped, updateNodeInternals]);

  // Shape data for this node (from last /api/check)
  const myShapes = nodeShapes[id]; // { portId: number[] }

  // Build the hover tooltip lines: one per output port
  const outputShapeLines: { portName: string; shape: string }[] = outputs
    .map((out: any) => ({
      portName: out.name,
      shape: fmtShape(myShapes?.[out.id]),
    }))
    .filter((o: { portName: string; shape: string }) => o.shape !== '');

  // Also collect input shapes for display (useful for multi-input nodes like Add)
  const inputShapeLines: { portName: string; shape: string }[] = inputs
    .map((inp: any) => ({
      portName: inp.name,
      shape: fmtShape(myShapes?.[inp.id]),
    }))
    .filter((o: { portName: string; shape: string }) => o.shape !== '');

  const hasShapeInfo = outputShapeLines.length > 0 || inputShapeLines.length > 0;

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
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

  const handleToggleOrientation = (e: React.MouseEvent) => {
    e.stopPropagation();
    const next = isVertical ? 'horizontal' : 'vertical';
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? { ...n, data: { ...n.data, portLayout: { ...(n.data.portLayout || {}), orientation: next } } }
          : n
      )
    );
  };

  const handleToggleFlip = (e: React.MouseEvent) => {
    e.stopPropagation();
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? { ...n, data: { ...n.data, portLayout: { ...(n.data.portLayout || {}), flipped: !isFlipped } } }
          : n
      )
    );
  };

  // Compute the effective variable name shown on the card
  const rawVar = ((data.varName as string) || "").trim();
  const effectiveVar = rawVar
    ? rawVar.toLowerCase().replace(/[^a-z0-9_]/g, "_").replace(/^[^a-z_]/, "x_$&")
    : null;

  // Handle positions
  const inputPosition = isVertical
    ? (isFlipped ? Position.Bottom : Position.Top)
    : (isFlipped ? Position.Right : Position.Left);

  const outputPosition = isVertical
    ? (isFlipped ? Position.Top : Position.Bottom)
    : (isFlipped ? Position.Left : Position.Right);

  // Handle container styling
  const inputContainerClass = isVertical
    ? (isFlipped
        ? 'absolute bottom-0 left-0 right-0 flex flex-row justify-evenly -mb-[5px] z-10'
        : 'absolute top-0 left-0 right-0 flex flex-row justify-evenly -mt-[5px] z-10')
    : (isFlipped
        ? 'absolute right-0 top-0 bottom-0 flex flex-col justify-evenly -mr-[5px] z-10'
        : 'absolute left-0 top-0 bottom-0 flex flex-col justify-evenly -ml-[5px] z-10');

  const outputContainerClass = isVertical
    ? (isFlipped
        ? 'absolute top-0 left-0 right-0 flex flex-row justify-evenly -mt-[5px] z-10'
        : 'absolute bottom-0 left-0 right-0 flex flex-row justify-evenly -mb-[5px] z-10')
    : (isFlipped
        ? 'absolute left-0 top-0 bottom-0 flex flex-col justify-evenly -ml-[5px] z-10'
        : 'absolute right-0 top-0 bottom-0 flex flex-col justify-evenly -mr-[5px] z-10');

  return (
    <div
      style={{
        borderColor: isError ? '#e54545' : '#3a3a3a',
        boxShadow: isError
          ? '0 0 0 1px rgba(229,69,69,0.4), 0 4px 12px rgba(0,0,0,0.5)'
          : '0 4px 12px rgba(0,0,0,0.5)',
      }}
      className={`group relative bg-[#252525] border rounded-[3px] min-w-[164px] flex ${
        isVertical ? 'flex-col' : 'flex-row'
      } transition-all duration-150 hover:border-[#505050]`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Category accent strip */}
      <div
        className={isVertical ? 'h-[3px] rounded-t-[3px] w-full flex-shrink-0' : 'w-[3px] rounded-l-[3px] flex-shrink-0'}
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
          onClick={handleToggleOrientation}
          title={`Orientation: ${isVertical ? 'Vertical (Top/Bottom)' : 'Horizontal (Left/Right)'} - Click to toggle`}
          className="p-0.5 text-[#888] hover:text-[#38bdf8] transition-colors"
        >
          {isVertical ? <ArrowUpDown className="w-3 h-3 text-[#38bdf8]" /> : <ArrowRightLeft className="w-3 h-3" />}
        </button>
        <button
          onClick={handleToggleFlip}
          title={`Flip Ports: ${isFlipped ? 'Flipped' : 'Normal'} - Click to flip`}
          className={`p-0.5 transition-colors ${isFlipped ? 'text-[#eab308]' : 'text-[#888] hover:text-[#e2e2e2]'}`}
        >
          <FlipHorizontal className="w-3 h-3" />
        </button>
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

      {/* Input handles */}
      <div className={inputContainerClass}>
        {inputs.map((inp: any) => {
          const inShape = fmtShape(myShapes?.[inp.id]);
          return (
            <div key={inp.id} className="relative group/h flex items-center justify-center">
              <Handle
                type="target"
                position={inputPosition}
                id={inp.id}
                isConnectable={isConnectable}
                className="!w-2.5 !h-2.5 !bg-[#505050] !border-[1.5px] !border-[#1e1e1e] !relative !transform-none hover:!bg-[#2d8cf0] transition-colors"
              />
              <span className={`absolute pointer-events-none text-[10px] text-[#aaa] bg-[#1e1e1e] border border-[#3a3a3a] px-1.5 py-px rounded-sm whitespace-nowrap opacity-0 group-hover/h:opacity-100 transition-opacity z-30 ${
                isVertical
                  ? (isFlipped ? 'bottom-4' : 'top-4')
                  : (isFlipped ? 'right-4' : 'left-4')
              }`}>
                {inp.name}{inShape ? ` · ${inShape}` : ''}
              </span>
            </div>
          );
        })}
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

        {/* Output variable badge */}
        <span
          className="text-[10px] font-mono truncate leading-none flex items-center gap-1"
          style={{ color: effectiveVar ? '#2d8cf0' : '#666' }}
          title={effectiveVar ? `Output variable: ${effectiveVar}` : 'Output variable: auto-generated (x_<id>)'}
        >
          <span className="text-[9px] text-[#555]">var:</span>
          <span>{effectiveVar || 'auto'}</span>
        </span>

        {paramSummary ? (
          <span className="text-[10px] font-mono text-[#888] truncate leading-none">
            {paramSummary}
          </span>
        ) : null}
      </div>

      {/* Output handles */}
      <div className={outputContainerClass}>
        {outputs.map((out: any) => {
          const outShape = fmtShape(myShapes?.[out.id]);
          return (
            <div key={out.id} className="relative group/h flex items-center justify-center">
              <span className={`absolute pointer-events-none text-[10px] text-[#aaa] bg-[#1e1e1e] border border-[#3a3a3a] px-1.5 py-px rounded-sm whitespace-nowrap opacity-0 group-hover/h:opacity-100 transition-opacity z-30 ${
                isVertical
                  ? (isFlipped ? 'top-4' : 'bottom-4')
                  : (isFlipped ? 'left-4' : 'right-4')
              }`}>
                {out.name}{outShape ? ` · ${outShape}` : ''}
              </span>
              <Handle
                type="source"
                position={outputPosition}
                id={out.id}
                isConnectable={isConnectable}
                className="!w-2.5 !h-2.5 !bg-[#505050] !border-[1.5px] !border-[#1e1e1e] !relative !transform-none hover:!bg-[#2d8cf0] transition-colors"
              />
            </div>
          );
        })}
      </div>

      {/* ── Shape tooltip — appears below the node on hover ────────────────── */}
      {hovered && hasShapeInfo && (
        <div
          className="absolute left-1/2 -translate-x-1/2 pointer-events-none z-40"
          style={{ top: 'calc(100% + 8px)' }}
        >
          <div className="bg-[#1a1a1a] border border-[#3a3a3a] rounded-[3px] shadow-xl px-3 py-2 min-w-[140px]">
            {/* Output shapes */}
            {outputShapeLines.length > 0 && (
              <div className="mb-1">
                <div className="text-[8px] uppercase tracking-wider text-[#555] mb-1">Output shape</div>
                {outputShapeLines.map(({ portName, shape }) => (
                  <div key={portName} className="flex items-center justify-between gap-3">
                    {outputShapeLines.length > 1 && (
                      <span className="text-[9px] text-[#666]">{portName}</span>
                    )}
                    <span className="text-[11px] font-mono text-[#4ade80] ml-auto">[{shape}]</span>
                  </div>
                ))}
              </div>
            )}

            {/* Input shapes (for multi-input nodes like Add) */}
            {inputShapeLines.length > 0 && (
              <div className={outputShapeLines.length > 0 ? 'border-t border-[#2a2a2a] pt-1 mt-1' : ''}>
                <div className="text-[8px] uppercase tracking-wider text-[#555] mb-1">Input shapes</div>
                {inputShapeLines.map(({ portName, shape }) => (
                  <div key={portName} className="flex items-center justify-between gap-3">
                    <span className="text-[9px] text-[#666]">{portName}</span>
                    {/* Multi-input shapes already carry per-tensor brackets from fmtShape */}
                    <span className="text-[11px] font-mono text-[#60a5fa]">
                      {shape.startsWith('[') ? shape : `[${shape}]`}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Small arrow pointing up */}
            <div
              className="absolute left-1/2 -translate-x-1/2 -top-[5px] w-2.5 h-2.5 bg-[#1a1a1a] border-t border-l border-[#3a3a3a] rotate-45"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(CustomNode);
