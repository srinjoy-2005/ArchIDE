import { create } from 'zustand';
import { Node, Edge } from '@xyflow/react';
import { SidebarView } from './types';

interface EditorState {
  generatedCode: string;
  setGeneratedCode: (code: string) => void;
  shapeErrorNodeId: string | null;
  setShapeErrorNodeId: (id: string | null) => void;
  nodeShapes: Record<string, any>;
  setNodeShapes: (shapes: Record<string, any>) => void;
  docMenuInfo: { visible: boolean; x: number; y: number; blockId: string; intro: string; details: string; name: string; isLoading?: boolean } | null;
  setDocMenuInfo: (info: any) => void;
  docPanelInfo: { visible: boolean; blockId: string; name: string; intro: string; details: string } | null;
  setDocPanelInfo: (info: any) => void;

  activeSidebarView: SidebarView;
  setActiveSidebarView: (view: SidebarView) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;

  clipboard: { nodes: Node[], edges: Edge[] } | null;
  setClipboard: (data: { nodes: Node[], edges: Edge[] } | null) => void;
  canvasMode: 'pan' | 'select';
  setCanvasMode: (mode: 'pan' | 'select') => void;

  edgeRouting: 'bezier' | 'step';
  setEdgeRouting: (mode: 'bezier' | 'step') => void;
  toggleEdgeRouting: () => void;

  inspectorOpen: boolean;
  setInspectorOpen: (open: boolean) => void;
  toggleInspector: () => void;

  quickInsert: { isOpen: boolean; clientPos: { x: number; y: number } | null };
  setQuickInsert: (val: { isOpen: boolean; clientPos?: { x: number; y: number } | null }) => void;
}

const getInitialEdgeRouting = (): 'bezier' | 'step' => {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('archide_edge_routing');
    if (saved === 'step' || saved === 'bezier') return saved;
  }
  return 'bezier';
};

export const useEditorStore = create<EditorState>((set) => ({
  generatedCode: `import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x`,
  setGeneratedCode: (code) => set({ generatedCode: code }),
  shapeErrorNodeId: null,
  setShapeErrorNodeId: (id) => set({ shapeErrorNodeId: id }),
  nodeShapes: {},
  setNodeShapes: (shapes) => set({ nodeShapes: shapes }),
  docMenuInfo: null,
  setDocMenuInfo: (info) => set({ docMenuInfo: info }),
  docPanelInfo: null,
  setDocPanelInfo: (info) => set({ docPanelInfo: info }),

  activeSidebarView: 'explorer',
  setActiveSidebarView: (view) => set({ activeSidebarView: view, sidebarOpen: true }),
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  clipboard: null,
  setClipboard: (data) => set({ clipboard: data }),
  canvasMode: 'pan',
  setCanvasMode: (mode) => set({ canvasMode: mode }),

  edgeRouting: getInitialEdgeRouting(),
  setEdgeRouting: (mode) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('archide_edge_routing', mode);
    }
    set({ edgeRouting: mode });
  },
  toggleEdgeRouting: () => set((state) => {
    const next = state.edgeRouting === 'step' ? 'bezier' : 'step';
    if (typeof window !== 'undefined') {
      localStorage.setItem('archide_edge_routing', next);
    }
    return { edgeRouting: next };
  }),

  inspectorOpen: true,
  setInspectorOpen: (open) => set({ inspectorOpen: open }),
  toggleInspector: () => set((state) => ({ inspectorOpen: !state.inspectorOpen })),

  quickInsert: { isOpen: false, clientPos: null },
  setQuickInsert: (val) => set({
    quickInsert: {
      isOpen: val.isOpen,
      clientPos: val.clientPos ?? null,
    }
  }),
}));
