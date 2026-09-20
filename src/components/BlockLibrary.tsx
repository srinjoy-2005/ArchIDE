"use client";

/**
 * src/components/BlockLibrary.tsx
 *
 * The left sidebar of the IDE. Contains:
 *
 * - CategoryIcon: maps a category string to a matching lucide icon.
 * - BlockItem: a single draggable block entry. Sets up HTML5 drag data so the
 *   DnDCanvas can reconstruct the full block definition on drop. Also handles
 *   right-click to load docs from /api/blocks/:id/docs into the DocContextMenu.
 * - BlockLibrary: the full sidebar component. Fetches the live block registry from
 *   the backend on mount (with FALLBACK_BLOCKS as the default). Derives custom module
 *   blocks from other open files in the Zustand store. Groups all blocks by category.
 */

import React, { useState, useEffect } from 'react';
import { useEditorStore } from '../lib/store';
import { Search, Box, Sparkles, ArrowRightLeft, Activity, Brain } from 'lucide-react';
import { API_BASE, FALLBACK_BLOCKS } from '../lib/constants';

// ─── CategoryIcon ─────────────────────────────────────────────────────────────

function CategoryIcon({ category }: { category: string }) {
  const cls = 'w-3 h-3 flex-shrink-0';
  const c = category.toLowerCase();
  if (c.includes('core'))       return <Box className={cls} style={{ color: '#60a5fa' }} />;
  if (c.includes('activation')) return <Sparkles className={cls} style={{ color: '#a78bfa' }} />;
  if (c.includes('tensor'))     return <ArrowRightLeft className={cls} style={{ color: '#fb923c' }} />;
  if (c.includes('pool'))       return <Activity className={cls} style={{ color: '#34d399' }} />;
  return <Brain className={cls} style={{ color: '#6b7280' }} />;
}

// ─── BlockItem ────────────────────────────────────────────────────────────────

function BlockItem({ blockDef }: { blockDef: any }) {
  const setDocMenuInfo = useEditorStore((s) => s.setDocMenuInfo);

  const onDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/reactflow', 'custom');
    event.dataTransfer.setData('application/label', blockDef.name);
    event.dataTransfer.setData('application/blockDef', JSON.stringify(blockDef));
    event.dataTransfer.effectAllowed = 'move';
  };

  const onContextMenu = async (event: React.MouseEvent) => {
    event.preventDefault();
    // Show loading state immediately so the menu appears without delay
    setDocMenuInfo({ visible: true, x: event.clientX, y: event.clientY, blockId: blockDef.id, name: blockDef.name, intro: '', details: '', isLoading: true });

    try {
      const res = await fetch(`${API_BASE}/api/blocks/${blockDef.id}/docs`);
      if (res.ok) {
        const docs = await res.json();
        setDocMenuInfo({ visible: true, x: event.clientX, y: event.clientY, blockId: blockDef.id, name: blockDef.name, intro: docs.intro, details: docs.details, isLoading: false });
      } else {
        setDocMenuInfo({ visible: true, x: event.clientX, y: event.clientY, blockId: blockDef.id, name: blockDef.name, intro: 'Failed to load documentation.', details: '', isLoading: false });
      }
    } catch (err) {
      console.error('Failed to fetch block docs:', err);
      setDocMenuInfo({ visible: true, x: event.clientX, y: event.clientY, blockId: blockDef.id, name: blockDef.name, intro: 'Error loading documentation.', details: '', isLoading: false });
    }
  };

  return (
    <div
      onDragStart={onDragStart}
      onContextMenu={onContextMenu}
      draggable
      className="flex items-center gap-2.5 bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#3a3a3a] hover:border-[#505050] rounded-[3px] px-3 py-2 cursor-grab active:cursor-grabbing transition-colors group"
    >
      <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: blockDef.color || '#4a4a4a' }} />
      <span className="text-[12px] text-[#c8c8c8] group-hover:text-[#e2e2e2] transition-colors truncate">{blockDef.name}</span>
    </div>
  );
}

// ─── BlockLibrary ─────────────────────────────────────────────────────────────

export function BlockLibrary() {
  const [registry, setRegistry] = useState<any[]>(FALLBACK_BLOCKS);
  const [search, setSearch] = useState('');
  const files = useEditorStore((s) => s.files);
  const activeFileId = useEditorStore((s) => s.activeFileId);

  // Fetch the live block registry from the backend; silently fall back on error
  useEffect(() => {
    fetch(`${API_BASE}/api/blocks`)
      .then((r) => r.json())
      .then((data) => { if (Array.isArray(data) && data.length) setRegistry(data); })
      .catch(() => {});
  }, []);

  // Derive custom module entries from other open files (non-active tabs)
  const customBlocks = files
    .filter((f) => f.id !== activeFileId)
    .map((f) => {
      const inputs  = f.nodes.filter((n) => n.data.block_id === 'input').map((n) => ({ id: n.id, name: n.data.label as string, type: 'tensor' }));
      const outputs = f.nodes.filter((n) => n.data.block_id === 'output').map((n) => ({ id: n.id, name: n.data.label as string, type: 'tensor' }));
      return { id: 'custom_module', custom_module_id: f.id, name: f.name, category: 'Custom Modules', color: '#eab308', is_functional: false, inputs, outputs, params: [] };
    });

  // Group filtered blocks by category
  const categories: Record<string, any[]> = {};
  [...registry, ...customBlocks]
    .filter((b) => b.name.toLowerCase().includes(search.toLowerCase()))
    .forEach((block) => {
      if (!categories[block.category]) categories[block.category] = [];
      categories[block.category].push(block);
    });

  return (
    <aside
      className="flex flex-col flex-shrink-0 overflow-hidden"
      style={{ width: 220, background: '#1e1e1e', borderRight: '1px solid #363636' }}
    >
      {/* Sidebar header */}
      <div className="flex items-center justify-between px-3 flex-shrink-0" style={{ height: 32, borderBottom: '1px solid #363636' }}>
        <span className="text-[9px] uppercase tracking-wider text-[#555]">Layer Library</span>
      </div>

      {/* Search */}
      <div className="px-2 pt-2 pb-1 flex-shrink-0">
        <div className="flex items-center gap-1.5 bg-[#252525] border border-[#3a3a3a] rounded-[3px] px-2 py-1">
          <Search className="w-3 h-3 text-[#555]" />
          <input
            type="text"
            placeholder="Search layers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent text-[11px] text-[#c8c8c8] placeholder-[#444] outline-none flex-1 min-w-0"
          />
        </div>
      </div>

      {/* Block list grouped by category */}
      <div className="flex-1 overflow-y-auto px-2 py-1 space-y-3">
        {Object.keys(categories).length === 0 ? (
          <div className="text-[11px] text-[#555] italic px-1 pt-2">
            {search ? 'No matching layers.' : 'Loading layers…'}
          </div>
        ) : (
          Object.entries(categories).map(([category, blocks]) => (
            <div key={category}>
              <div className="flex items-center gap-1.5 px-1 py-1">
                <CategoryIcon category={category} />
                <span className="text-[9px] uppercase tracking-wider text-[#555]">{category}</span>
              </div>
              <div className="flex flex-col gap-1">
                {blocks.map((block: any) => (
                  <BlockItem key={block.custom_module_id || block.id} blockDef={block} />
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
