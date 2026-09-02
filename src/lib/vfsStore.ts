import { create } from 'zustand';
import { Node, Edge } from '@xyflow/react';
import { Folder, GraphFile, ArchIDEProject, ArchVariable, ArchVariableType, ArchVariableScope } from './types';

const generateId = () => Math.random().toString(36).substring(2, 9);

function getOrCreateFolderPath(
  folders: Folder[],
  parentId: string | null,
  pathParts: string[]
): string | null {
  if (pathParts.length === 0) return parentId;
  const part = pathParts[0];
  let folder = folders.find((f) => f.parentId === parentId && f.name === part);
  if (!folder) {
    folder = { id: generateId(), name: part, parentId, isExpanded: true };
    folders.push(folder);
  }
  return getOrCreateFolderPath(folders, folder.id, pathParts.slice(1));
}

function migrateParameters(raw: any[]): ArchVariable[] {
  return raw.map((p) => ({
    id: generateId(),
    name: p.name ?? 'unnamed',
    type: (p.type as ArchVariableType) ?? 'int',
    default: p.default ?? 0,
    description: p.description ?? '',
    scope: (p.scope as ArchVariableScope) ?? 'init_param',
  }));
}

interface VFSState {
  folders: Folder[];
  files: GraphFile[];
  openTabIds: string[];
  activeFileId: string;
  entryFileId: string;
  isMirroring: boolean;
  /** Stable UIDs for the root graphs/ and python/ folders — unaffected by user renames */
  graphsFolderId: string;
  pythonFolderId: string;

  setEntryFileId: (id: string) => void;
  setIsMirroring: (val: boolean) => void;
  
  handleCompiledFiles: (compiledData: Record<string, string>) => void;
  overwriteFilesFromVFS: (filesMap: Record<string, any>) => void;

  openTab: (id: string) => void;
  closeTab: (id: string) => void;
  switchFile: (id: string) => void;
  updateFileState: (id: string, nodes: Node[], edges: Edge[]) => void;

  createFolder: (name: string, parentId?: string | null) => void;
  renameFolder: (id: string, newName: string) => void;
  deleteFolder: (id: string) => void;
  toggleFolder: (id: string) => void;
  setAllFoldersExpanded: (expanded: boolean) => void;

  createFile: (name: string, parentId?: string | null, fileType?: 'graph' | 'code') => void;
  renameFile: (id: string, newName: string) => void;
  deleteFile: (id: string) => void;
  moveItem: (id: string, isFolder: boolean, targetParentId: string | null) => void;

  exportProjectJson: () => string;
  importProjectJson: (jsonStr: string) => boolean;

  // Variable management (per active file)
  addVariable: (fileId: string, variable: Omit<ArchVariable, 'id'>) => void;
  updateVariable: (fileId: string, varId: string, updates: Partial<ArchVariable>) => void;
  deleteVariable: (fileId: string, varId: string) => void;
  /** Clears all @var:<varName> bindings from node paramValues in the given file */
  clearVariableBindings: (fileId: string, varName: string) => void;
}

export const useVFSStore = create<VFSState>((set, get) => ({
  folders: [
    { id: 'fol_graphs', name: 'graphs', parentId: null, isExpanded: true },
    { id: 'fol_python', name: 'python', parentId: null, isExpanded: true }
  ],
  files: [
    {
      id: 'archide_toml',
      name: 'archide.toml',
      parentId: null,
      nodes: [], edges: [],
      fileType: 'code',
      compiledCode: '[directories]\\ngraphs_dir = "graphs"\\npython_dir = "python"'
    },
    {
      id: 'main',
      name: 'main.arch',
      parentId: 'fol_graphs',
      nodes: [],
      edges: []
    }
  ],
  openTabIds: ['main'],
  activeFileId: 'main',
  entryFileId: 'main',
  isMirroring: false,
  // Stable UIDs that persist even if the user renames graphs/ or python/
  graphsFolderId: 'fol_graphs',
  pythonFolderId: 'fol_python',

  setEntryFileId: (id) => set({ entryFileId: id }),
  setIsMirroring: (val) => set({ isMirroring: val }),

  handleCompiledFiles: (compiledData) => {
    set((state) => {
      const newFiles = [...state.files];
      const newFolders = [...state.folders];

      // Use stable UID instead of name lookup so renames don't break this
      let pyFolderId = state.pythonFolderId;
      if (!newFolders.find((f) => f.id === pyFolderId)) {
        // python folder was deleted — recreate it
        const newPyFolder = { id: pyFolderId, name: 'python', parentId: null, isExpanded: true };
        newFolders.push(newPyFolder);
      }

      const validCompiledFilePaths = new Set<string>();

      for (const [gid, code] of Object.entries(compiledData)) {
        const pathParts = gid.split('/');
        const baseName = pathParts.pop();
        const pyFileName = baseName + '.py';
        const pyParentId = getOrCreateFolderPath(newFolders, pyFolderId, pathParts);
        
        validCompiledFilePaths.add(`${pyParentId}/${pyFileName}`);

        const existingIdx = newFiles.findIndex(
          (f) => f.parentId === pyParentId && f.name === pyFileName
        );
        if (existingIdx >= 0) {
          newFiles[existingIdx] = { ...newFiles[existingIdx], compiledCode: code as string };
        } else {
          newFiles.push({
            id: generateId(),
            name: pyFileName,
            parentId: pyParentId,
            nodes: [],
            edges: [],
            fileType: 'code',
            compiledCode: code as string
          });
        }
      }
      
      // Cleanup stale files that are inside the python folder but weren't in this compile pass
      const isInsideFolderTree = (fileParentId: string | null, targetRootId: string): boolean => {
        let current = fileParentId;
        while (current) {
          if (current === targetRootId) return true;
          const folder = newFolders.find(f => f.id === current);
          current = folder?.parentId ?? null;
        }
        return false;
      };

      const finalFiles = newFiles.filter(f => {
        if (!isInsideFolderTree(f.parentId ?? null, pyFolderId)) return true;
        return validCompiledFilePaths.has(`${f.parentId}/${f.name}`);
      });

      return { files: finalFiles, folders: newFolders };
    });
  },

  overwriteFilesFromVFS: (filesMap) => set((state) => {
    const newFiles = [...state.files];
    const newFolders = [...state.folders];
    // Use stable UID — not name — so this works even if the user renames the graphs folder
    const graphsFolderId = state.graphsFolderId;
    if (!newFolders.find((f) => f.id === graphsFolderId)) return state;

    for (const [fileId, content] of Object.entries(filesMap)) {
      const parts = fileId.split('/');
      const fileName = parts.pop() + '.arch';
      const targetParentId = getOrCreateFolderPath(newFolders, graphsFolderId, parts);

      const rawVars: any[] = content.variables ?? content.parameters ?? [];
      const variables: ArchVariable[] = migrateParameters(rawVars);

      const existingIdx = newFiles.findIndex(
        (f) => f.name === fileName && f.parentId === targetParentId
      );
      if (existingIdx >= 0) {
        newFiles[existingIdx] = {
          ...newFiles[existingIdx],
          nodes: content.nodes || [],
          edges: content.edges || [],
          variables,
        };
      } else {
        newFiles.push({
          id: fileId === 'main' ? 'main' : generateId(),
          name: fileName,
          parentId: targetParentId,
          nodes: content.nodes || [],
          edges: content.edges || [],
          variables,
          fileType: 'graph',
        });
      }
    }
    return { files: newFiles, folders: newFolders };
  }),

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
            files: [{ id: fallbackId, name: 'main.arch', parentId: null, nodes: [], edges: [] }],
            openTabIds: [fallbackId],
            activeFileId: fallbackId
          };
        }
      }

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
      return {
        openTabIds: alreadyOpen ? state.openTabIds : [...state.openTabIds, id],
        activeFileId: id,
      };
    });
  },

  updateFileState: (id, nodes, edges) => {
    set((state) => ({
      files: state.files.map(f => f.id === id ? { ...f, nodes, edges } : f)
    }));
  },

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
      const getDescendantFolderIds = (folderId: string): string[] => {
        const children = state.folders.filter(f => f.parentId === folderId);
        return [folderId, ...children.flatMap(c => getDescendantFolderIds(c.id))];
      };

      const folderIdsToDelete = new Set(getDescendantFolderIds(id));
      const remainingFolders = state.folders.filter(f => !folderIdsToDelete.has(f.id));
      
      const entryFile = state.files.find(f => f.id === state.entryFileId);
      const isEntryInDeletedFolder = entryFile && entryFile.parentId && folderIdsToDelete.has(entryFile.parentId);

      let remainingFiles = state.files.filter(f => !f.parentId || !folderIdsToDelete.has(f.parentId));
      
      if (isEntryInDeletedFolder && entryFile) {
        const movedEntry = { ...entryFile, parentId: null };
        remainingFiles.push(movedEntry);
      }

      const remainingDeletedFileIds = new Set(
        state.files.filter(f => f.parentId && folderIdsToDelete.has(f.parentId) && f.id !== state.entryFileId).map(f => f.id)
      );
      const remainingOpenTabs = state.openTabIds.filter(tabId => !remainingDeletedFileIds.has(tabId));

      if (remainingFiles.length === 0) {
        const fallbackId = generateId();
        return {
          folders: remainingFolders,
          files: [{ id: fallbackId, name: 'main.arch', parentId: null, nodes: [], edges: [] }],
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

  createFile: (name, parentId = null, fileType = 'graph') => {
    const trimmed = name.trim() || 'Untitled';
    const isCode = fileType === 'code' || trimmed.endsWith('.py');
    const newFile: GraphFile = {
      id: generateId(),
      name: trimmed,
      parentId: parentId ?? null,
      variables: [],
      nodes: [],
      edges: [],
      fileType: isCode ? 'code' : 'graph',
      compiledCode: isCode ? '# Python script' : ''
    };
    set((state) => ({
      files: [...state.files, newFile],
      openTabIds: state.openTabIds.includes(newFile.id) ? state.openTabIds : [...state.openTabIds, newFile.id],
      activeFileId: newFile.id,
    }));
  },

  renameFile: (id, newName) => {
    const trimmed = newName.trim();
    if (!trimmed) return;
    set((state) => ({
      files: state.files.map(f => f.id === id ? { ...f, name: trimmed } : f)
    }));
  },

  deleteFile: (id) => {
    set((state) => {
      if (id === state.entryFileId) return state;

      const newFiles = state.files.filter(f => f.id !== id);
      const newOpenTabs = state.openTabIds.filter(tabId => tabId !== id);

      if (newFiles.length === 0) {
        const fallbackId = generateId();
        return {
          files: [{ id: fallbackId, name: 'main.arch', parentId: null, nodes: [], edges: [] }],
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

  exportProjectJson: () => {
    const state = get();
    const manifest: ArchIDEProject = {
      name: 'ArchIDE_Project',
      version: '1.0.0',
      entry_point: state.entryFileId,
      folders: state.folders,
      files: state.files.map((f) => ({
        id: f.id,
        name: f.name.endsWith('.json') || f.name.endsWith('.py') ? f.name : `${f.name}.json`,
        parentId: f.parentId || null,
        variables: f.variables || [],
        nodes: f.nodes.map((n) => ({
          id: n.id,
          position: n.position || { x: 100, y: 100 },
          data: {
            block_id: (n.data as any)?.block_id || 'input',
            label: (n.data as any)?.label || '',
            varName: (n.data as any)?.varName || '',
            custom_module_id: (n.data as any)?.custom_module_id || '',
            paramValues: (n.data as any)?.paramValues || {},
          },
        })) as any,
        edges: f.edges.map((e) => ({
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
        variables: migrateParameters(f.variables ?? f.parameters ?? []),
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
          },
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
      const validEntryId = files.some((f) => f.id === entryPoint || f.name === entryPoint)
        ? (files.find((f) => f.id === entryPoint || f.name === entryPoint)?.id || files[0].id)
        : files[0].id;

      // Detect stable folder UIDs from imported project (if present) or re-derive by name
      const graphsFolder = folders.find((f) => f.parentId === null && (f.name === 'graphs' || f.id === get().graphsFolderId));
      const pythonFolder = folders.find((f) => f.parentId === null && (f.name === 'python' || f.id === get().pythonFolderId));

      set({
        folders,
        files,
        entryFileId: validEntryId,
        openTabIds: [validEntryId],
        activeFileId: validEntryId,
        graphsFolderId: graphsFolder?.id ?? get().graphsFolderId,
        pythonFolderId: pythonFolder?.id ?? get().pythonFolderId,
      });
      return true;
    } catch (err) {
      console.error('Failed to parse ArchIDE project JSON:', err);
      return false;
    }
  },

  addVariable: (fileId, variable) => set((state) => ({
    files: state.files.map((f) => {
      if (f.id !== fileId) return f;
      const newVar: ArchVariable = { ...variable, id: generateId() };
      return { ...f, variables: [...(f.variables || []), newVar] };
    }),
  })),

  updateVariable: (fileId, varId, updates) => set((state) => ({
    files: state.files.map((f) => {
      if (f.id !== fileId) return f;
      return {
        ...f,
        variables: (f.variables || []).map((v) =>
          v.id === varId ? { ...v, ...updates } : v
        ),
      };
    }),
  })),

  deleteVariable: (fileId, varId) => set((state) => ({
    files: state.files.map((f) => {
      if (f.id !== fileId) return f;
      return { ...f, variables: (f.variables || []).filter((v) => v.id !== varId) };
    }),
  })),

  clearVariableBindings: (fileId, varName) => set((state) => ({
    files: state.files.map((f) => {
      if (f.id !== fileId) return f;
      const binding = `@var:${varName}`;
      return {
        ...f,
        nodes: f.nodes.map((n) => {
          const pv = (n.data?.paramValues as Record<string, any>) || {};
          const hasBinding = Object.values(pv).some((v) => v === binding);
          if (!hasBinding) return n;
          const newPv: Record<string, any> = {};
          for (const [k, v] of Object.entries(pv)) {
            newPv[k] = v === binding ? null : v;
          }
          return { ...n, data: { ...n.data, paramValues: newPv } };
        }),
      };
    }),
  })),
}));
