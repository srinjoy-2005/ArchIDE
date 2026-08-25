"use client";

/**
 * src/components/DocPanels.tsx
 *
 * Two floating documentation UI components:
 * - DocContextMenu: a small popover that appears on right-click over a block in the sidebar,
 *   showing a quick intro fetched from /api/blocks/:id/docs.
 * - DocDetailsPanel: a fuller slide-in panel with detailed docs, shown when the user
 *   clicks the expand button inside the context menu.
 */

import { useEditorStore } from '../lib/store';
import { Plus, X } from 'lucide-react';

export function DocContextMenu() {
  const info = useEditorStore((s) => s.docMenuInfo);
  const setInfo = useEditorStore((s) => s.setDocMenuInfo);
  const setPanelInfo = useEditorStore((s) => s.setDocPanelInfo);

  if (!info || !info.visible) return null;

  return (
    <>
      {/* Invisible overlay to close the menu on click-away */}
      <div
        className="fixed inset-0 z-[100]"
        onClick={() => setInfo(null)}
        onContextMenu={(e) => { e.preventDefault(); setInfo(null); }}
      />
      <div
        className="fixed z-[101] w-64 bg-[#1e1e1e] border border-[#3a3a3a] rounded-[5px] shadow-2xl p-3 flex flex-col gap-2"
        style={{ top: info.y, left: info.x }}
      >
        <div className="flex justify-between items-start">
          <span className="text-[13px] font-semibold text-[#e2e2e2]">{info.name}</span>
          {!info.isLoading && (
            <button
              className="text-[#888] hover:text-[#d4d4d4] transition-colors bg-[#2a2a2a] p-1 rounded"
              onClick={() => {
                setPanelInfo({ visible: true, blockId: info.blockId, name: info.name, intro: info.intro, details: info.details });
                setInfo(null);
              }}
              title="Show full details"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <div className="text-[11px] text-[#aaa] leading-relaxed">
          {info.isLoading ? (
            <span className="animate-pulse flex items-center gap-2">
              <div className="w-3 h-3 rounded-full border-2 border-[#555] border-t-[#888] animate-spin" />
              Loading docs...
            </span>
          ) : (
            info.intro
          )}
        </div>
      </div>
    </>
  );
}

export function DocDetailsPanel() {
  const info = useEditorStore((s) => s.docPanelInfo);
  const setInfo = useEditorStore((s) => s.setDocPanelInfo);

  if (!info || !info.visible) return null;

  return (
    <div className="w-80 border-r border-[#3a3a3a] bg-[#1a1a1a] flex flex-col flex-shrink-0 z-50">
      <div className="flex items-center justify-between p-3 border-b border-[#3a3a3a] flex-shrink-0">
        <span className="text-[13px] font-semibold text-[#e2e2e2]">{info.name} Docs</span>
        <button onClick={() => setInfo(null)} className="text-[#888] hover:text-[#d4d4d4] transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 text-[12px] text-[#ccc] flex flex-col gap-3">
        <div>
          <h4 className="text-[13px] font-medium text-[#fff] mb-1">Intro</h4>
          <p className="leading-relaxed">{info.intro}</p>
        </div>
        <div>
          <h4 className="text-[13px] font-medium text-[#fff] mb-1">Details</h4>
          <pre className="whitespace-pre-wrap font-sans leading-relaxed text-[#aaa]">{info.details}</pre>
        </div>
      </div>
    </div>
  );
}
