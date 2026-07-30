import React, { useState, useCallback } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
} from '@xyflow/react';
import { useEditorStore } from '../lib/store';

// Format shape array → "1×64×28×28"
function fmtShape(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return '';
  return shape.join('×');
}

interface TensorEdgeProps {
  id: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: any;
  targetPosition: any;
  source: string;
  sourceHandleId?: string | null;
  style?: React.CSSProperties;
  markerEnd?: string;
  selected?: boolean;
}

const TensorEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  source,
  sourceHandleId,
  style,
  markerEnd,
  selected,
}: TensorEdgeProps) => {
  const [hovered, setHovered] = useState(false);
  const nodeShapes = useEditorStore((s) => s.nodeShapes);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // The shape flowing through this edge = output shape of the source node at the source handle
  const handleId = sourceHandleId || 'out';
  const shape = nodeShapes[source]?.[handleId];
  const shapeStr = fmtShape(shape);
  const hasShape = shapeStr !== '';

  return (
    <>
      {/* Invisible wider hit area for hover detection */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={16}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{ cursor: 'default' }}
      />

      {/* Actual visible edge */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: hovered
            ? '#2d8cf0'
            : selected
            ? '#505050'
            : '#3a3a3a',
          strokeWidth: hovered || selected ? 1.5 : 1,
          transition: 'stroke 0.15s, stroke-width 0.15s',
        }}
      />

      {/* Shape label — appears at edge midpoint on hover */}
      {hovered && hasShape && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              zIndex: 50,
            }}
            onMouseEnter={() => setHovered(true)}
          >
            <div className="flex items-center gap-1.5 bg-[#1a1a1a] border border-[#2d8cf0]/50 rounded-[3px] px-2.5 py-1 shadow-xl whitespace-nowrap">
              {/* Tensor icon */}
              <svg width="10" height="10" viewBox="0 0 10 10" className="flex-shrink-0">
                <rect x="0" y="0" width="4.5" height="4.5" rx="0.5" fill="#2d8cf0" opacity="0.8" />
                <rect x="5.5" y="0" width="4.5" height="4.5" rx="0.5" fill="#2d8cf0" opacity="0.5" />
                <rect x="0" y="5.5" width="4.5" height="4.5" rx="0.5" fill="#2d8cf0" opacity="0.5" />
                <rect x="5.5" y="5.5" width="4.5" height="4.5" rx="0.5" fill="#2d8cf0" opacity="0.3" />
              </svg>
              <span className="text-[11px] font-mono text-[#7ec8e3]">[{shapeStr}]</span>
            </div>
          </div>
        </EdgeLabelRenderer>
      )}

      {/* Always-visible small dot at midpoint when shape is known (subtle indicator) */}
      {hasShape && !hovered && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              zIndex: 10,
            }}
          >
            <div
              className="w-1.5 h-1.5 rounded-full bg-[#2d8cf0] opacity-40"
            />
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default TensorEdge;
