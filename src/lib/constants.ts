/**
 * src/lib/constants.ts
 *
 * Shared frontend constants:
 * - API_BASE: Python backend URL
 * - initialNodes / initialEdges: empty graph defaults
 * - getId(): monotonically-increasing node ID generator
 * - FALLBACK_BLOCKS: hardcoded block definitions used when the backend is unreachable
 */
import type { Node, Edge } from '@xyflow/react';

/** Backend base URL — the API runs on port 8001 */
export const API_BASE = 'http://localhost:8001';

export const initialNodes: Node[] = [];
export const initialEdges: Edge[] = [];

let nodeCounter = 100;
/** Returns a unique node ID, e.g. "node_101" */
export const getId = () => `node_${nodeCounter++}`;

/** Fallback block registry used when `/api/blocks` is unreachable */
export const FALLBACK_BLOCKS: any[] = [
  { id: 'input',   name: 'Input',    category: 'Core Layers', is_functional: true,  color: '#4ade80', inputs: [],                                              outputs: [{ id: 'out', name: 'Output' }],                      params: [{ name: 'shape', type: 'string', default: '(1,3,224,224)' }] },
  { id: 'output',  name: 'Output',   category: 'Core Layers', is_functional: true,  color: '#f87171', inputs: [{ id: 'in', name: 'Return Value', is_list: true }], outputs: [],                                               params: [] },
  { id: 'linear',  name: 'Linear',   category: 'Core Layers', is_functional: false, color: '#818cf8', inputs: [{ id: 'in', name: 'Input' }],                    outputs: [{ id: 'out', name: 'Output' }],                      params: [{ name: 'in_features', type: 'int', default: 128 }, { name: 'out_features', type: 'int', default: 64 }] },
  { id: 'conv2d',  name: 'Conv2D',   category: 'Core Layers', is_functional: false, color: '#60a5fa', inputs: [{ id: 'in', name: 'Input' }],                    outputs: [{ id: 'out', name: 'Output' }],                      params: [{ name: 'in_channels', type: 'int', default: 3 }, { name: 'out_channels', type: 'int', default: 16 }, { name: 'kernel_size', type: 'int', default: 3 }] },
  { id: 'relu',    name: 'ReLU',     category: 'Activations', is_functional: false, color: '#a78bfa', inputs: [{ id: 'in', name: 'Input' }],                    outputs: [{ id: 'out', name: 'Output' }],                      params: [] },
  { id: 'softmax', name: 'Softmax',  category: 'Activations', is_functional: false, color: '#c084fc', inputs: [{ id: 'in', name: 'Input' }],                    outputs: [{ id: 'out', name: 'Output' }],                      params: [{ name: 'dim', type: 'int', default: 1 }] },
  { id: 'add',     name: 'Add',      category: 'Tensor Ops',  is_functional: true,  color: '#fb923c', inputs: [{ id: 'in', name: 'Inputs', is_list: true }],    outputs: [{ id: 'out', name: 'Out' }],                         params: [] },
  { id: 'mul',     name: 'Multiply', category: 'Tensor Ops',  is_functional: true,  color: '#8b5cf6', inputs: [{ id: 'in', name: 'Inputs', is_list: true }],    outputs: [{ id: 'out', name: 'Out' }],                         params: [] },
  { id: 'split',   name: 'Split',    category: 'Tensor Ops',  is_functional: true,  color: '#34d399', inputs: [{ id: 'in', name: 'Input' }],                    outputs: [{ id: 'out_0', name: 'Chunk 1' }, { id: 'out_1', name: 'Chunk 2' }], params: [{ name: 'chunks', type: 'int', default: 2 }, { name: 'dim', type: 'int', default: 0 }] },
];
