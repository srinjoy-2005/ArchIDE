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
}

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
}));
