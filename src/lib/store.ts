/**
 * src/lib/store.ts
 *
 * Central Zustand state management store for ArchIDE.
 * Manages:
 * - Generated PyTorch code display
 * - Shape validation states (errors and inferred shapes)
 * - Documentation panel and context menu visibility
 * - Virtual File System (VFS): Hierarchical folders and graph files
 * - Multi-Tab session management (opening, switching, closing tabs without deleting files)
 * - Permanent file and folder deletion in VFS
 * - Left sidebar navigation (Activity Bar & view states)
 */
import { create } from 'zustand';
import { Node, Edge } from '@xyflow/react';

export interface Folder {
  id: string;
  name: string;
  parentId: string | null;
  isExpanded: boolean;
}

export interface GraphParamDef {
  name: string;
  type?: string;
  default?: any;
  description?: string;
}

export interface GraphFile {
  id: string;
  name: string;
  parentId?: string | null;
  parameters?: GraphParamDef[];
  nodes: Node[];
  edges: Edge[];
  fileType?: 'graph' | 'code';
  compiledCode?: string;
}

export interface ArchIDEProject {
  name: string;
  version: string;
  entry_point: string;
  folders: Folder[];
  files: GraphFile[];
}

export type SidebarView = 'explorer' | 'library' | 'search';

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

  // Active Center View Mode ('graph' for visual node canvas, 'code' for full Python editor)
  activeViewMode: 'graph' | 'code';
  setActiveViewMode: (mode: 'graph' | 'code') => void;

  // Sidebar & Activity Bar
  activeSidebarView: SidebarView;
  setActiveSidebarView: (view: SidebarView) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;

  // Virtual File System (VFS)
  folders: Folder[];
  files: GraphFile[];
  openTabIds: string[];
  activeFileId: string;

  // Tab Actions (closing tab != deleting file)
  openTab: (id: string) => void;
  closeTab: (id: string) => void;
  switchFile: (id: string) => void;

  // Folder Actions
  createFolder: (name: string, parentId?: string | null) => void;
  renameFolder: (id: string, newName: string) => void;
  deleteFolder: (id: string) => void;
  toggleFolder: (id: string) => void;
  setAllFoldersExpanded: (expanded: boolean) => void;

  // File Actions
  createFile: (name: string, parentId?: string | null, fileType?: 'graph' | 'code') => void;
  renameFile: (id: string, newName: string) => void;
  updateFileState: (id: string, nodes: Node[], edges: Edge[]) => void;
  deleteFile: (id: string) => void;
  moveItem: (id: string, isFolder: boolean, targetParentId: string | null) => void;

  // Project Import / Export Serialization
  exportProjectJson: () => string;
  importProjectJson: (jsonStr: string) => boolean;
}

const generateId = () => Math.random().toString(36).substring(2, 9);

export const useEditorStore = create<EditorState>((set, get) => ({
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

  // Center View Mode
  activeViewMode: 'graph',
  setActiveViewMode: (mode) => set({ activeViewMode: mode }),

  // Sidebar state
  activeSidebarView: 'explorer',
  setActiveSidebarView: (view) => set((state) => ({
    activeSidebarView: view,
    sidebarOpen: true
  })),
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  // Initial Virtual File System with standard clean structure
  folders: [
    {
      id: 'fol_blocks',
      name: 'blocks',
      parentId: null,
      isExpanded: true
    },
    {
      id: 'fol_conv',
      name: 'conv',
      parentId: 'fol_blocks',
      isExpanded: true
    }
  ],
  files: [
    {
      id: 'main',
      name: 'main.json',
      parentId: null,
      nodes: [
        {
          id: 'node_in',
          type: 'custom',
          position: { x: 80, y: 150 },
          data: { block_id: 'input', label: 'x', paramValues: { shape: '(1, 3, 224, 224)' } }
        },
        {
          id: 'node_conv',
          type: 'custom',
          position: { x: 280, y: 150 },
          data: { block_id: 'conv2d', label: 'Conv2D', paramValues: { in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1, stride: 1 } }
        },
        {
          id: 'node_relu',
          type: 'custom',
          position: { x: 480, y: 150 },
          data: { block_id: 'relu', label: 'ReLU', paramValues: {} }
        },
        {
          id: 'node_out',
          type: 'custom',
          position: { x: 680, y: 150 },
          data: { block_id: 'output', label: 'out', paramValues: {} }
        }
      ],
      edges: [
        { id: 'e1', type: 'tensor', source: 'node_in', sourceHandle: 'out', target: 'node_conv', targetHandle: 'in' },
        { id: 'e2', type: 'tensor', source: 'node_conv', sourceHandle: 'out', target: 'node_relu', targetHandle: 'in' },
        { id: 'e3', type: 'tensor', source: 'node_relu', sourceHandle: 'out', target: 'node_out', targetHandle: 'in' }
      ]
    },
    {
      id: 'file_res_block',
      name: 'res_block.json',
      parentId: 'fol_conv',
      parameters: [
        { name: 'in_channels', type: 'int', default: 32 },
        { name: 'out_channels', type: 'int', default: 32 }
      ],
      nodes: [
        {
          id: 'rb_in',
          type: 'custom',
          position: { x: 80, y: 120 },
          data: { block_id: 'input', label: 'x', paramValues: { shape: '(1, 32, 224, 224)' } }
        },
        {
          id: 'rb_conv',
          type: 'custom',
          position: { x: 280, y: 80 },
          data: { block_id: 'conv2d', label: 'Conv 3x3', paramValues: { in_channels: 32, out_channels: 32, kernel_size: 3, padding: 1, stride: 1 } }
        },
        {
          id: 'rb_add',
          type: 'custom',
          position: { x: 480, y: 120 },
          data: { block_id: 'add', label: 'Residual Add', paramValues: {} }
        },
        {
          id: 'rb_out',
          type: 'custom',
          position: { x: 680, y: 120 },
          data: { block_id: 'output', label: 'out', paramValues: {} }
        }
      ],
      edges: [
        { id: 're1', type: 'tensor', source: 'rb_in', sourceHandle: 'out', target: 'rb_conv', targetHandle: 'in' },
        { id: 're2', type: 'tensor', source: 'rb_conv', sourceHandle: 'out', target: 'rb_add', targetHandle: 'in' },
        { id: 're3', type: 'tensor', source: 'rb_in', sourceHandle: 'out', target: 'rb_add', targetHandle: 'in' },
        { id: 're4', type: 'tensor', source: 'rb_add', sourceHandle: 'out', target: 'rb_out', targetHandle: 'in' }
      ]
    }
  ],
  openTabIds: ['main', 'file_res_block'],
  activeFileId: 'main',

  // ─── Tab Actions ─────────────────────────────────────────────────────────────
  openTab: (id) => {
    set((state) => {
      const fileExists = state.files.some(f => f.id === id);
      if (!fileExists) return state;
      const alreadyOpen = state.openTabIds.includes(id);
      return {
        openTabIds: alreadyOpen ? state.openTabIds : [...state.openTabIds, id],
        activeFileId: id
      };
    });
  },

  closeTab: (id) => {
    set((state) => {
      const tabIndex = state.openTabIds.indexOf(id);
      if (tabIndex === -1) return state;

      const newOpenTabs = state.openTabIds.filter(tId => tId !== id);

      // If user closed the only open tab, ensure at least one tab stays open from files
      if (newOpenTabs.length === 0) {
        const remainingFile = state.files.find(f => f.id !== id) || state.files[0];
        if (remainingFile) {
          return {
            openTabIds: [remainingFile.id],
            activeFileId: remainingFile.id
          };
        } else {
          const fallbackId = generateId();
          return {
            files: [{ id: fallbackId, name: 'Main', parentId: null, nodes: [], edges: [] }],
            openTabIds: [fallbackId],
            activeFileId: fallbackId
          };
        }
      }

      // If active tab was closed, switch to adjacent open tab
      let nextActiveId = state.activeFileId;
      if (state.activeFileId === id) {
        const nextIndex = Math.min(tabIndex, newOpenTabs.length - 1);
        nextActiveId = newOpenTabs[nextIndex];
      }

      return {
        openTabIds: newOpenTabs,
        activeFileId: nextActiveId
      };
    });
  },

  switchFile: (id) => {
    set((state) => {
      const targetFile = state.files.find(f => f.id === id);
      if (!targetFile) return state;
      const alreadyOpen = state.openTabIds.includes(id);
      const isCodeFile = targetFile.fileType === 'code' || targetFile.name.endsWith('.py');
      return {
        openTabIds: alreadyOpen ? state.openTabIds : [...state.openTabIds, id],
        activeFileId: id,
        activeViewMode: isCodeFile ? 'code' : state.activeViewMode,
      };
    });
  },

  // ─── Folder Actions ──────────────────────────────────────────────────────────
  createFolder: (name, parentId = null) => {
    const newFolder: Folder = {
      id: generateId(),
      name: name.trim() || 'New Folder',
      parentId: parentId ?? null,
      isExpanded: true
    };
    set((state) => ({
      folders: [...state.folders, newFolder]
    }));
  },

  renameFolder: (id, newName) => {
    const trimmed = newName.trim();
    if (!trimmed) return;
    set((state) => ({
      folders: state.folders.map(f => f.id === id ? { ...f, name: trimmed } : f)
    }));
  },

  deleteFolder: (id) => {
    set((state) => {
      // Find all nested folder IDs recursively
      const getDescendantFolderIds = (folderId: string): string[] => {
        const children = state.folders.filter(f => f.parentId === folderId);
        return [folderId, ...children.flatMap(c => getDescendantFolderIds(c.id))];
      };

      const folderIdsToDelete = new Set(getDescendantFolderIds(id));
      const remainingFolders = state.folders.filter(f => !folderIdsToDelete.has(f.id));
      const remainingFiles = state.files.filter(f => !f.parentId || !folderIdsToDelete.has(f.parentId));
      const remainingDeletedFileIds = new Set(state.files.filter(f => f.parentId && folderIdsToDelete.has(f.parentId)).map(f => f.id));
      const remainingOpenTabs = state.openTabIds.filter(tabId => !remainingDeletedFileIds.has(tabId));

      if (remainingFiles.length === 0) {
        const fallbackId = generateId();
        return {
          folders: remainingFolders,
          files: [{ id: fallbackId, name: 'main.json', parentId: null, nodes: [], edges: [] }],
          openTabIds: [fallbackId],
          activeFileId: fallbackId
        };
      }

      const activeStillExists = remainingFiles.some(f => f.id === state.activeFileId);
      const nextActiveId = activeStillExists
        ? state.activeFileId
        : (remainingOpenTabs[0] || remainingFiles[0].id);

      return {
        folders: remainingFolders,
        files: remainingFiles,
        openTabIds: remainingOpenTabs.length > 0 ? remainingOpenTabs : [nextActiveId],
        activeFileId: nextActiveId
      };
    });
  },

  toggleFolder: (id) => {
    set((state) => ({
      folders: state.folders.map(f => f.id === id ? { ...f, isExpanded: !f.isExpanded } : f)
    }));
  },

  setAllFoldersExpanded: (expanded) => {
    set((state) => ({
      folders: state.folders.map(f => ({ ...f, isExpanded: expanded }))
    }));
  },

  // ─── File Actions ────────────────────────────────────────────────────────────
  createFile: (name, parentId = null, fileType = 'graph') => {
    const trimmed = name.trim() || 'Untitled';
    const isCode = fileType === 'code' || trimmed.endsWith('.py');
    const newFile: GraphFile = {
      id: generateId(),
      name: trimmed,
      parentId: parentId ?? null,
      parameters: [],
      nodes: [],
      edges: [],
      fileType: isCode ? 'code' : 'graph',
      compiledCode: isCode ? '# Python script' : ''
    };
    set((state) => ({
      files: [...state.files, newFile],
      openTabIds: state.openTabIds.includes(newFile.id) ? state.openTabIds : [...state.openTabIds, newFile.id],
      activeFileId: newFile.id,
      activeViewMode: isCode ? 'code' : state.activeViewMode,
    }));
  },

  renameFile: (id, newName) => {
    const trimmed = newName.trim();
    if (!trimmed) return;
    set((state) => ({
      files: state.files.map(f => f.id === id ? { ...f, name: trimmed } : f)
    }));
  },

  updateFileState: (id, nodes, edges) => {
    set((state) => ({
      files: state.files.map(f => f.id === id ? { ...f, nodes, edges } : f)
    }));
  },

  deleteFile: (id) => {
    set((state) => {
      const newFiles = state.files.filter(f => f.id !== id);
      const newOpenTabs = state.openTabIds.filter(tabId => tabId !== id);

      if (newFiles.length === 0) {
        const fallbackId = generateId();
        return {
          files: [{ id: fallbackId, name: 'main.json', parentId: null, nodes: [], edges: [] }],
          openTabIds: [fallbackId],
          activeFileId: fallbackId
        };
      }

      const activeStillExists = newFiles.some(f => f.id === state.activeFileId);
      const nextActiveId = activeStillExists
        ? state.activeFileId
        : (newOpenTabs[0] || newFiles[0].id);

      return {
        files: newFiles,
        openTabIds: newOpenTabs.length > 0 ? newOpenTabs : [nextActiveId],
        activeFileId: nextActiveId
      };
    });
  },

  moveItem: (id, isFolder, targetParentId) => {
    set((state) => {
      if (isFolder) {
        if (id === targetParentId) return state;
        return {
          folders: state.folders.map(f => f.id === id ? { ...f, parentId: targetParentId } : f)
        };
      } else {
        return {
          files: state.files.map(f => f.id === id ? { ...f, parentId: targetParentId } : f)
        };
      }
    });
  },

  // ─── Project Serialization ───────────────────────────────────────────────────
  exportProjectJson: () => {
    const state = get();
    const manifest: ArchIDEProject = {
      name: 'ArchIDE_Project',
      version: '1.0.0',
      entry_point: state.files.find(f => f.id === state.activeFileId)?.name || 'main.json',
      folders: state.folders,
      files: state.files.map(f => ({
        id: f.id,
        name: f.name.endsWith('.json') || f.name.endsWith('.py') ? f.name : `${f.name}.json`,
        parentId: f.parentId || null,
        parameters: f.parameters || [],
        nodes: f.nodes.map(n => ({
          id: n.id,
          position: n.position || { x: 100, y: 100 },
          data: {
            block_id: (n.data as any)?.block_id || 'input',
            label: (n.data as any)?.label || '',
            varName: (n.data as any)?.varName || '',
            custom_module_id: (n.data as any)?.custom_module_id || '',
            paramValues: (n.data as any)?.paramValues || {},
          }
        })) as any,
        edges: f.edges.map(e => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle,
          target: e.target,
          targetHandle: e.targetHandle,
        })) as any,
      })),
    };
    return JSON.stringify(manifest, null, 2);
  },

  importProjectJson: (jsonStr: string) => {
    try {
      const data = JSON.parse(jsonStr);
      if (!data || !Array.isArray(data.files) || data.files.length === 0) {
        console.error('Invalid ArchIDE project format');
        return false;
      }
      const folders: Folder[] = Array.isArray(data.folders) ? data.folders : [];
      const files: GraphFile[] = data.files.map((f: any) => ({
        id: f.id || generateId(),
        name: f.name || 'Untitled',
        parentId: f.parentId ?? null,
        parameters: f.parameters || [],
        nodes: (f.nodes || []).map((n: any) => ({
          id: n.id,
          type: 'custom',
          position: n.position || { x: 100, y: 100 },
          data: {
            block_id: n.data?.block_id || n.block_id || 'input',
            label: n.data?.label || n.label || '',
            varName: n.data?.varName || n.varName || '',
            custom_module_id: n.data?.custom_module_id || n.custom_module_id || '',
            paramValues: n.data?.paramValues || n.paramValues || {},
          }
        })),
        edges: (f.edges || []).map((e: any) => ({
          id: e.id || generateId(),
          type: 'tensor',
          source: e.source,
          sourceHandle: e.sourceHandle,
          target: e.target,
          targetHandle: e.targetHandle,
        })),
      }));

      const entryPoint = data.entry_point || data.entryPointId || files[0].id;
      const validEntryId = files.some(f => f.id === entryPoint || f.name === entryPoint)
        ? (files.find(f => f.id === entryPoint || f.name === entryPoint)?.id || files[0].id)
        : files[0].id;

      set({
        folders,
        files,
        openTabIds: [validEntryId],
        activeFileId: validEntryId,
        shapeErrorNodeId: null,
        nodeShapes: {},
        activeViewMode: 'graph',
      });
      return true;
    } catch (err) {
      console.error('Failed to parse ArchIDE project JSON:', err);
      return false;
    }
  }
}));
