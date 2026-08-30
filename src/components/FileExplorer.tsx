"use client";

/**
 * src/components/FileExplorer.tsx
 *
 * OneCompiler-inspired File and Folder Explorer for ArchIDE.
 * Features:
 * - Hierarchical tree view (nested folders and files)
 * - Header actions: New File, New Folder, Collapse All
 * - In-place inline creation & renaming for files and folders
 * - Contextual hover actions: Create inside folder, Rename, Delete
 * - Opening tabs on file click without deleting files
 */

import React, { useState, useRef, useEffect } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useEditorStore, type Folder, type GraphFile } from '../lib/store';
import {
  ChevronDown,
  ChevronRight,
  Folder as FolderIcon,
  FolderOpen,
  FolderPlus,
  FilePlus,
  FolderMinus,
  Edit2,
  Trash2,
  FileCode,
  Network,
  Check,
  X,
  Download,
  Upload,
} from 'lucide-react';

interface CreatingState {
  type: 'file' | 'folder';
  parentId: string | null;
}

interface RenamingState {
  id: string;
  isFolder: boolean;
  initialName: string;
}

export function FileExplorer() {
  const {
    folders,
    files,
    activeFileId,
    createFile,
    createFolder,
    renameFile,
    renameFolder,
    deleteFile,
    deleteFolder,
    toggleFolder,
    setAllFoldersExpanded,
    openTab,
    updateFileState,
    exportProjectJson,
    importProjectJson,
    entryFileId,
  } = useEditorStore();

  const { getNodes, getEdges } = useReactFlow();

  const [creating, setCreating] = useState<CreatingState | null>(null);
  const [newItemName, setNewItemName] = useState('');
  const [renaming, setRenaming] = useState<RenamingState | null>(null);
  const [renameValue, setRenameValue] = useState('');

  const inputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExportProject = () => {
    updateFileState(activeFileId, getNodes(), getEdges());
    const jsonStr = exportProjectJson();
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'archide_project.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportProject = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (content) {
        const ok = importProjectJson(content);
        if (!ok) {
          alert('Failed to import project: Invalid format.');
        }
      }
    };
    reader.readAsText(file);
    if (event.target) event.target.value = '';
  };

  useEffect(() => {
    if (creating && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [creating]);

  useEffect(() => {
    if (renaming && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renaming]);

  // Handle opening/switching active file with state snapshot & view mode
  const handleSelectFile = (fileId: string) => {
    updateFileState(activeFileId, getNodes(), getEdges());
    openTab(fileId);
  };

  // Confirm creation of new item
  const handleConfirmCreate = () => {
    if (!creating) return;
    const name = newItemName.trim();
    if (name) {
      updateFileState(activeFileId, getNodes(), getEdges());
      if (creating.type === 'file') {
        const isCode = name.endsWith('.py');
        createFile(name, creating.parentId, isCode ? 'code' : 'graph');
      } else {
        createFolder(name, creating.parentId);
      }
    }
    setCreating(null);
    setNewItemName('');
  };

  // Confirm renaming
  const handleConfirmRename = () => {
    if (!renaming) return;
    const name = renameValue.trim();
    if (name) {
      if (renaming.isFolder) {
        renameFolder(renaming.id, name);
      } else {
        renameFile(renaming.id, name);
      }
    }
    setRenaming(null);
    setRenameValue('');
  };

  // Recursive Tree Rendering
  const renderTree = (parentId: string | null = null, depth = 0) => {
    const currentFolders = folders.filter((f) => f.parentId === parentId);
    const currentFiles = files.filter((f) => (f.parentId ?? null) === parentId);

    return (
      <div className="flex flex-col">
        {/* Folders */}
        {currentFolders.map((folder) => {
          const isRenamingThis = renaming?.id === folder.id && renaming.isFolder;

          return (
            <div key={folder.id} className="flex flex-col">
              <div
                onClick={() => toggleFolder(folder.id)}
                className="group relative flex items-center justify-between py-1 px-2 cursor-pointer hover:bg-[#232323] rounded-[3px] text-[11.5px] text-[#cccccc] transition-colors"
                style={{ paddingLeft: `${depth * 12 + 6}px` }}
              >
                <div className="flex items-center gap-1.5 min-w-0 flex-1">
                  <span className="text-[#777] hover:text-[#e0e0e0] flex-shrink-0">
                    {folder.isExpanded ? (
                      <ChevronDown className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5" />
                    )}
                  </span>
                  <span className="text-[#d4af37] flex-shrink-0">
                    {folder.isExpanded ? (
                      <FolderOpen className="w-3.5 h-3.5" />
                    ) : (
                      <FolderIcon className="w-3.5 h-3.5" />
                    )}
                  </span>

                  {isRenamingThis ? (
                    <input
                      ref={renameInputRef}
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleConfirmRename();
                        if (e.key === 'Escape') setRenaming(null);
                      }}
                      onBlur={handleConfirmRename}
                      onClick={(e) => e.stopPropagation()}
                      className="bg-[#141414] border border-[#2d8cf0] rounded px-1 text-[11px] text-[#e2e2e2] outline-none w-full"
                    />
                  ) : (
                    <span className="truncate font-medium text-[#d8d8d8]">{folder.name}</span>
                  )}
                </div>

                {/* Hover action toolbar on folder */}
                {!isRenamingThis && (
                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 text-[#777] transition-opacity ml-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!folder.isExpanded) toggleFolder(folder.id);
                        setCreating({ type: 'file', parentId: folder.id });
                        setNewItemName('');
                      }}
                      title="New File Inside"
                      className="hover:text-[#e2e2e2] p-0.5 rounded hover:bg-[#333333]"
                    >
                      <FilePlus className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!folder.isExpanded) toggleFolder(folder.id);
                        setCreating({ type: 'folder', parentId: folder.id });
                        setNewItemName('');
                      }}
                      title="New Subfolder"
                      className="hover:text-[#e2e2e2] p-0.5 rounded hover:bg-[#333333]"
                    >
                      <FolderPlus className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenaming({ id: folder.id, isFolder: true, initialName: folder.name });
                        setRenameValue(folder.name);
                      }}
                      title="Rename"
                      className="hover:text-[#e2e2e2] p-0.5 rounded hover:bg-[#333333]"
                    >
                      <Edit2 className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete folder "${folder.name}" and all its contents?`)) {
                          deleteFolder(folder.id);
                        }
                      }}
                      title="Delete Folder"
                      className="hover:text-[#e54545] p-0.5 rounded hover:bg-[#333333]"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>

              {/* Children (if folder is expanded) */}
              {folder.isExpanded && (
                <div className="flex flex-col">
                  {/* Inline creation input inside this folder */}
                  {creating && creating.parentId === folder.id && (
                    <div
                      className="flex items-center gap-1.5 py-1 px-2"
                      style={{ paddingLeft: `${(depth + 1) * 12 + 6}px` }}
                    >
                      {creating.type === 'folder' ? (
                        <FolderIcon className="w-3.5 h-3.5 text-[#d4af37]" />
                      ) : (
                        <FileCode className="w-3.5 h-3.5 text-[#2d8cf0]" />
                      )}
                      <input
                        ref={inputRef}
                        type="text"
                        placeholder={creating.type === 'folder' ? 'folder_name' : 'file_name'}
                        value={newItemName}
                        onChange={(e) => setNewItemName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleConfirmCreate();
                          if (e.key === 'Escape') setCreating(null);
                        }}
                        onBlur={handleConfirmCreate}
                        className="bg-[#141414] border border-[#2d8cf0] rounded px-1.5 py-0.5 text-[11px] text-[#e2e2e2] outline-none flex-1 min-w-0"
                      />
                    </div>
                  )}
                  {renderTree(folder.id, depth + 1)}
                </div>
              )}
            </div>
          );
        })}

        {/* Files */}
        {currentFiles.map((file) => {
          const isActive = file.id === activeFileId;
          const isRenamingThis = renaming?.id === file.id && !renaming.isFolder;
          const isCode = file.fileType === 'code' || file.name.endsWith('.py');

          return (
            <div
              key={file.id}
              onClick={() => handleSelectFile(file.id)}
              className={`group relative flex items-center justify-between py-1 px-2 cursor-pointer rounded-[3px] text-[11.5px] transition-colors ${
                isActive
                  ? 'bg-[#2b2b2b] text-[#ffffff] font-medium'
                  : 'text-[#9c9c9c] hover:bg-[#232323] hover:text-[#e0e0e0]'
              }`}
              style={{ paddingLeft: `${depth * 12 + (parentId ? 18 : 8)}px` }}
            >
              <div className="flex items-center gap-2 min-w-0 flex-1">
                {isCode ? (
                  <FileCode className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-[#eab308]' : 'text-[#888888]'}`} />
                ) : (
                  <Network className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-[#38bdf8]' : 'text-[#666666]'}`} />
                )}
                {isRenamingThis ? (
                  <input
                    ref={renameInputRef}
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleConfirmRename();
                      if (e.key === 'Escape') setRenaming(null);
                    }}
                    onBlur={handleConfirmRename}
                    onClick={(e) => e.stopPropagation()}
                    className="bg-[#141414] border border-[#2d8cf0] rounded px-1 text-[11px] text-[#e2e2e2] outline-none w-full"
                  />
                ) : (
                  <span className="truncate">
                    {file.name}
                    {file.id === entryFileId && (
                      <span className="ml-2 text-[10px] text-[#eab308] border border-[#eab308]/30 bg-[#eab308]/10 px-1 py-0 rounded">
                        Main
                      </span>
                    )}
                  </span>
                )}
              </div>

              {/* Hover actions on file */}
              {!isRenamingThis && (
                <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 text-[#777] transition-opacity ml-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setRenaming({ id: file.id, isFolder: false, initialName: file.name });
                      setRenameValue(file.name);
                    }}
                    title="Rename File"
                    className="hover:text-[#e2e2e2] p-0.5 rounded hover:bg-[#3a3a3a]"
                  >
                    <Edit2 className="w-3 h-3" />
                  </button>
                  {files.length > 1 && file.id !== entryFileId && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Permanently delete file "${file.name}"?`)) {
                          deleteFile(file.id);
                        }
                      }}
                      title="Delete File Permanently"
                      className="hover:text-[#e54545] p-0.5 rounded hover:bg-[#3a3a3a]"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <aside
      className="flex flex-col flex-shrink-0 overflow-hidden select-none"
      style={{
        width: 220,
        background: '#191919',
        borderRight: '1px solid #282828',
      }}
    >
      {/* Explorer Header */}
      <div
        className="flex items-center justify-between px-3 flex-shrink-0"
        style={{ height: 34, borderBottom: '1px solid #282828' }}
      >
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#888888]">
          Explorer
        </span>

        {/* Header Actions (OneCompiler style) */}
        <div className="flex items-center gap-0.5 text-[#777]">
          <button
            onClick={() => {
              setCreating({ type: 'file', parentId: null });
              setNewItemName('');
            }}
            title="New File"
            className="p-1 hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          >
            <FilePlus className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => {
              setCreating({ type: 'folder', parentId: null });
              setNewItemName('');
            }}
            title="New Folder"
            className="p-1 hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          >
            <FolderPlus className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setAllFoldersExpanded(false)}
            title="Collapse All Folders"
            className="p-1 hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          >
            <FolderMinus className="w-3.5 h-3.5" />
          </button>
          <div className="w-[1px] h-3 bg-[#333] mx-0.5" />
          <button
            onClick={handleExportProject}
            title="Export Project (Save)"
            className="p-1 hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Import Project (Open)"
            className="p-1 hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          >
            <Upload className="w-3.5 h-3.5" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImportProject}
            className="hidden"
          />
        </div>
      </div>

      {/* Directory & File Tree View */}
      <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
        {/* Root level creation input */}
        {creating && creating.parentId === null && (
          <div className="flex items-center gap-1.5 py-1 px-2">
            {creating.type === 'folder' ? (
              <FolderIcon className="w-3.5 h-3.5 text-[#d4af37]" />
            ) : (
              <FileCode className="w-3.5 h-3.5 text-[#2d8cf0]" />
            )}
            <input
              ref={inputRef}
              type="text"
              placeholder={creating.type === 'folder' ? 'folder_name' : 'file_name'}
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleConfirmCreate();
                if (e.key === 'Escape') setCreating(null);
              }}
              onBlur={handleConfirmCreate}
              className="bg-[#141414] border border-[#2d8cf0] rounded px-1.5 py-0.5 text-[11px] text-[#e2e2e2] outline-none flex-1 min-w-0"
            />
          </div>
        )}

        {/* Tree Root */}
        {renderTree(null, 0)}
      </div>
    </aside>
  );
}
