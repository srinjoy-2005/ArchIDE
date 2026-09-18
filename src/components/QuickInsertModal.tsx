"use client";

/**
 * src/components/QuickInsertModal.tsx
 *
 * Floating Spotlight-style palette triggered by pressing `Tab` on the canvas.
 * Allows searching and inserting built-in PyTorch layers and custom modules directly
 * at the cursor's canvas coordinates.
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useReactFlow } from '@xyflow/react';
import { Search, Sparkles, Box, ArrowRightLeft, Activity, Brain, CornerDownLeft } from 'lucide-react';
import { useEditorStore, useVFSStore } from '../lib/store';
import { API_BASE, FALLBACK_BLOCKS, getId } from '../lib/constants';

function CategoryIcon({ category }: { category: string }) {
  const cls = 'w-3 h-3 flex-shrink-0';
  const c = category.toLowerCase();
  if (c.includes('core'))       return <Box className={cls} style={{ color: '#60a5fa' }} />;
  if (c.includes('activation')) return <Sparkles className={cls} style={{ color: '#a78bfa' }} />;
  if (c.includes('tensor'))     return <ArrowRightLeft className={cls} style={{ color: '#fb923c' }} />;
  if (c.includes('pool'))       return <Activity className={cls} style={{ color: '#34d399' }} />;
  return <Brain className={cls} style={{ color: '#6b7280' }} />;
}

export function QuickInsertModal() {
  const quickInsert = useEditorStore((s) => s.quickInsert);
  const setQuickInsert = useEditorStore((s) => s.setQuickInsert);

  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [registry, setRegistry] = useState<any[]>(FALLBACK_BLOCKS);

  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const { screenToFlowPosition, setNodes } = useReactFlow();

  const files = useVFSStore((s) => s.files);
  const activeFileId = useVFSStore((s) => s.activeFileId);
  const entryFileId = useVFSStore((s) => s.entryFileId);
  const folders = useVFSStore((s) => s.folders);
  const graphsFolderId = useVFSStore((s) => s.graphsFolderId);

  // Fetch block registry
  useEffect(() => {
    fetch(`${API_BASE}/api/blocks`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data) && data.length) setRegistry(data);
      })
      .catch(() => {});
  }, []);

  // Focus input on open and reset index
  useEffect(() => {
    if (quickInsert.isOpen) {
      setSearch('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [quickInsert.isOpen]);

  // Derive custom blocks from other graph files
  const customBlocks = useMemo(() => {
    const getRootFolderId = (folderId: string | null): string | null => {
      if (!folderId) return null;
      let current = folders.find((f) => f.id === folderId);
      while (current && current.parentId) {
        current = folders.find((f) => f.id === current!.parentId);
      }
      return current?.id ?? null;
    };

    return files
      .filter((f) => {
        if (f.id === activeFileId) return false;
        if (f.id === entryFileId) return false;
        if (f.fileType === 'code') return false;
        const rootId = getRootFolderId(f.parentId ?? null);
        return rootId === graphsFolderId;
      })
      .map((f) => {
        const inputs = f.nodes
          .filter((n) => n.data.block_id === 'input')
          .map((n) => ({ id: n.id, name: n.data.label as string, type: 'tensor' }));
        const outputs = f.nodes
          .filter((n) => n.data.block_id === 'output')
          .map((n) => ({ id: n.id, name: n.data.label as string, type: 'tensor' }));
        const params = (f.variables || [])
          .filter((v) => v.scope === 'init_param')
          .map((v) => ({ name: v.name, type: v.type, default: v.default, section: 'basic' }));
        return {
          id: 'custom_module',
          custom_module_id: f.id,
          name: f.name,
          category: 'Custom Modules',
          color: '#eab308',
          is_functional: false,
          inputs,
          outputs,
          params,
          description: `Custom module from ${f.name}`,
        };
      });
  }, [files, activeFileId, entryFileId, folders, graphsFolderId]);

  const allBlocks = useMemo(() => [...registry, ...customBlocks], [registry, customBlocks]);

  const filteredBlocks = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return allBlocks;
    return allBlocks.filter(
      (b) =>
        b.name.toLowerCase().includes(q) ||
        (b.category && b.category.toLowerCase().includes(q)) ||
        (b.description && b.description.toLowerCase().includes(q))
    );
  }, [allBlocks, search]);

  // Keep selected index within range
  useEffect(() => {
    if (selectedIndex >= filteredBlocks.length) {
      setSelectedIndex(Math.max(0, filteredBlocks.length - 1));
    }
  }, [filteredBlocks.length, selectedIndex]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const activeEl = listRef.current.children[selectedIndex] as HTMLElement;
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [selectedIndex]);

  if (!quickInsert.isOpen) return null;

  const handleInsert = (blockDef: any) => {
    const clientX = quickInsert.clientPos?.x ?? window.innerWidth / 2;
    const clientY = quickInsert.clientPos?.y ?? window.innerHeight / 2;
    const position = screenToFlowPosition({ x: clientX, y: clientY });

    const initialParamValues: any = {};
    if (blockDef.params) {
      blockDef.params.forEach((p: any) => {
        initialParamValues[p.name] = p.default;
      });
    }

    const newNode = {
      id: getId(),
      type: 'custom',
      position,
      data: {
        block_id: blockDef.id,
        label: blockDef.name,
        description: blockDef.description || `A ${blockDef.name} layer`,
        params: blockDef.params || [],
        paramValues: initialParamValues,
        inputs: blockDef.inputs || [],
        outputs: blockDef.outputs || [],
        is_functional: blockDef.is_functional || false,
        varName: '',
        custom_module_id: blockDef.custom_module_id,
      },
    };

    setNodes((nds) => nds.concat(newNode as any));
    setQuickInsert({ isOpen: false });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredBlocks.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredBlocks.length) % Math.max(1, filteredBlocks.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredBlocks[selectedIndex]) {
        handleInsert(filteredBlocks[selectedIndex]);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setQuickInsert({ isOpen: false });
    }
  };

  // Compute position (clamped to screen viewport)
  const modalWidth = 320;
  const modalHeight = 360;
  const x = quickInsert.clientPos
    ? Math.min(Math.max(16, quickInsert.clientPos.x - 40), window.innerWidth - modalWidth - 16)
    : (window.innerWidth - modalWidth) / 2;
  const y = quickInsert.clientPos
    ? Math.min(Math.max(48, quickInsert.clientPos.y - 20), window.innerHeight - modalHeight - 16)
    : (window.innerHeight - modalHeight) / 2;

  return (
    <div
      className="fixed inset-0 z-50 select-none"
      onClick={() => setQuickInsert({ isOpen: false })}
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.25)' }}
    >
      <div
        className="absolute bg-[#1c1c1c] border border-[#383838] rounded-md shadow-2xl overflow-hidden flex flex-col"
        style={{
          left: `${x}px`,
          top: `${y}px`,
          width: `${modalWidth}px`,
          maxHeight: `${modalHeight}px`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search header */}
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-[#2c2c2c] bg-[#222222]">
          <Search className="w-3.5 h-3.5 text-[#2d8cf0] flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Search layer (Conv2D, Linear, ReLU...)..."
            className="w-full bg-transparent text-[12px] text-[#e2e2e2] placeholder-[#666] outline-none font-sans"
          />
        </div>

        {/* Results List */}
        <div ref={listRef} className="flex-1 overflow-y-auto p-1 space-y-0.5 max-h-[260px]">
          {filteredBlocks.length === 0 ? (
            <div className="p-4 text-center text-[11px] text-[#666] italic">
              No matching layers found.
            </div>
          ) : (
            filteredBlocks.map((block, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <div
                  key={`${block.id}_${block.custom_module_id || block.name}_${idx}`}
                  onClick={() => handleInsert(block)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-[#2b2b2b] text-[#ffffff]'
                      : 'text-[#aaa] hover:bg-[#232323] hover:text-[#ddd]'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <div
                      className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                      style={{ background: block.color || '#4a4a4a' }}
                    />
                    <div className="flex flex-col min-w-0">
                      <span className="text-[11.5px] font-medium truncate">{block.name}</span>
                      {block.description && (
                        <span className="text-[9.5px] text-[#666] truncate">
                          {block.description}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5 ml-2 flex-shrink-0">
                    <span className="text-[9px] font-mono text-[#555] bg-[#171717] px-1 py-0.5 rounded border border-[#2a2a2a]">
                      {block.category || 'General'}
                    </span>
                    {isSelected && (
                      <CornerDownLeft className="w-2.5 h-2.5 text-[#2d8cf0]" />
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-[#171717] border-t border-[#262626] text-[9.5px] text-[#666]">
          <span>↑↓ to navigate</span>
          <span>↵ to insert</span>
          <span>esc to close</span>
        </div>
      </div>
    </div>
  );
}
