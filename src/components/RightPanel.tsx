"use client";

/**
 * src/components/RightPanel.tsx
 *
 * The collapsible right panel dedicated exclusively to block inspection & configuration.
 * Renders the <PropertiesPanel /> with node parameters, shapes, and layer docs.
 */

import { useState } from 'react';
import { Settings2, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { PropertiesPanel } from './PropertiesPanel';

export function RightPanel() {
  const [rightOpen, setRightOpen] = useState(true);

  if (!rightOpen) {
    return (
      <button
        onClick={() => setRightOpen(true)}
        className="flex-shrink-0 flex items-center justify-center text-[#555] hover:text-[#aaa] transition-colors"
        style={{ width: 24, background: '#191919', borderLeft: '1px solid #282828' }}
        title="Open Inspector"
      >
        <PanelRightOpen className="w-3.5 h-3.5" />
      </button>
    );
  }

  return (
    <aside
      className="flex flex-col flex-shrink-0 overflow-hidden select-none"
      style={{ width: 280, background: '#191919', borderLeft: '1px solid #282828' }}
    >
      {/* Inspector Header */}
      <div
        className="flex items-center justify-between px-3 flex-shrink-0"
        style={{ height: 34, borderBottom: '1px solid #282828' }}
      >
        <div className="flex items-center gap-1.5 text-[#888888]">
          <Settings2 className="w-3.5 h-3.5 text-[#2d8cf0]" />
          <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#888888]">
            Inspector
          </span>
        </div>
        <button
          onClick={() => setRightOpen(false)}
          className="p-1 text-[#666] hover:text-[#e2e2e2] hover:bg-[#252525] rounded transition-colors"
          title="Collapse Inspector"
        >
          <PanelRightClose className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Inspector Content */}
      <div className="flex-1 overflow-y-auto p-3">
        <PropertiesPanel />
      </div>
    </aside>
  );
}
