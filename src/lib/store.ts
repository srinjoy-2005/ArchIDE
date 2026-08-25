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

export interface GraphFile {
  id: string;
  name: string;
  parentId?: string | null;
  nodes: Node[];
  edges: Edge[];
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
  createFile: (name: string, parentId?: string | null) => void;
  renameFile: (id: string, newName: string) => void;
  updateFileState: (id: string, nodes: Node[], edges: Edge[]) => void;
  deleteFile: (id: string) => void;
  moveItem: (id: string, isFolder: boolean, targetParentId: string | null) => void;
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

  // Sidebar state
  activeSidebarView: 'explorer',
  setActiveSidebarView: (view) => set((state) => ({
    activeSidebarView: view,
    sidebarOpen: true
  })),
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  // Initial Virtual File System
  folders: [
    {
      id: 'fol_models',
      name: 'models',
      parentId: null,
      isExpanded: true
    },
    {
      id: 'fol_blocks',
      name: 'blocks',
      parentId: null,
      isExpanded: true
    }
  ],
  files: [
    {
      id: 'main',
      name: 'Main',
      parentId: null,
      nodes: [],
      edges: []
    },
    {
      id: 'file_layer',
      name: 'attention',
      parentId: 'fol_blocks',
      nodes: [],
      edges: []
    }
  ],
  openTabIds: ['main', 'file_layer'],
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
      const fileExists = state.files.some(f => f.id === id);
      if (!fileExists) return state;
      const alreadyOpen = state.openTabIds.includes(id);
      return {
        openTabIds: alreadyOpen ? state.openTabIds : [...state.openTabIds, id],
        activeFileId: id
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
          files: [{ id: fallbackId, name: 'Main', parentId: null, nodes: [], edges: [] }],
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
  createFile: (name, parentId = null) => {
    const trimmed = name.trim() || 'Untitled';
    const newFile: GraphFile = {
      id: generateId(),
      name: trimmed,
      parentId: parentId ?? null,
      nodes: [],
      edges: []
    };
    set((state) => ({
      files: [...state.files, newFile],
      openTabIds: state.openTabIds.includes(newFile.id) ? state.openTabIds : [...state.openTabIds, newFile.id],
      activeFileId: newFile.id
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
          files: [{ id: fallbackId, name: 'Main', parentId: null, nodes: [], edges: [] }],
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
  }
}));
