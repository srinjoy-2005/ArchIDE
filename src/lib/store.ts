import { create } from 'zustand';
import { Node, Edge } from '@xyflow/react';

export interface GraphFile {
  id: string;
  name: string;
  nodes: Node[];
  edges: Edge[];
}

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

  // Multi-file state
  files: GraphFile[];
  activeFileId: string;
  createFile: (name: string) => void;
  switchFile: (id: string) => void;
  updateFileState: (id: string, nodes: Node[], edges: Edge[]) => void;
  deleteFile: (id: string) => void;
}

const generateId = () => Math.random().toString(36).substring(2, 9);

export const useEditorStore = create<EditorState>((set, get) => ({
  generatedCode: `import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Drag and connect blocks to generate code

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

  files: [
    {
      id: 'main',
      name: 'Main',
      nodes: [],
      edges: []
    }
  ],
  activeFileId: 'main',

  createFile: (name) => {
    const newFile: GraphFile = {
      id: generateId(),
      name,
      nodes: [],
      edges: []
    };
    set((state) => ({
      files: [...state.files, newFile],
      activeFileId: newFile.id
    }));
  },

  switchFile: (id) => {
    set({ activeFileId: id });
  },

  updateFileState: (id, nodes, edges) => {
    set((state) => ({
      files: state.files.map(f => f.id === id ? { ...f, nodes, edges } : f)
    }));
  },

  deleteFile: (id) => {
    set((state) => {
      const newFiles = state.files.filter(f => f.id !== id);
      if (newFiles.length === 0) {
        const fallbackId = generateId();
        return {
          files: [{ id: fallbackId, name: 'Main', nodes: [], edges: [] }],
          activeFileId: fallbackId
        };
      }
      return {
        files: newFiles,
        activeFileId: state.activeFileId === id ? newFiles[0].id : state.activeFileId
      };
    });
  }
}));
