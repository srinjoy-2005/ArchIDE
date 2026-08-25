"use client";

/**
 * src/components/RightPanel.tsx
 *
 * The collapsible right panel shell. Owns the `rightTab` and `rightOpen` local state.
 * Renders a two-tab bar (Inspector / Code) and delegates content rendering to:
 * - <PropertiesPanel /> for the Inspector tab
 * - <CodePanel /> for the Code tab
 *
 * When collapsed, renders a thin vertical button strip to reopen the panel.
 */

import { useState } from 'react';
import { Settings2, Code2, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { PropertiesPanel } from './PropertiesPanel';
import { CodePanel } from './CodePanel';

export function RightPanel() {
  const [rightTab,  setRightTab]  = useState<'inspector' | 'code'>('inspector');
  const [rightOpen, setRightOpen] = useState(true);

  if (!rightOpen) {
    return (
      <button
        onClick={() => setRightOpen(true)}
        className="flex-shrink-0 flex items-center justify-center text-[#555] hover:text-[#aaa] transition-colors"
        style={{ width: 24, background: '#1e1e1e', borderLeft: '1px solid #363636' }}
        title="Open panel"
      >
        <PanelRightOpen className="w-3.5 h-3.5" />
      </button>
    );
  }

  return (
    <aside
      className="flex flex-col flex-shrink-0 overflow-hidden"
      style={{ width: 280, background: '#1e1e1e', borderLeft: '1px solid #363636' }}
    >
      {/* Tab bar */}
      <div
        className="flex items-center flex-shrink-0"
        style={{ height: 32, borderBottom: '1px solid #363636' }}
      >
        <button
          onClick={() => setRightTab('inspector')}
          className={`flex items-center gap-1.5 px-3 h-full text-[11px] border-b-2 transition-colors ${
            rightTab === 'inspector' ? 'text-[#d4d4d4] border-[#2d8cf0]' : 'text-[#666] border-transparent hover:text-[#aaa]'
          }`}
        >
          <Settings2 className="w-3 h-3" />
          Inspector
        </button>
        <button
          onClick={() => setRightTab('code')}
          className={`flex items-center gap-1.5 px-3 h-full text-[11px] border-b-2 transition-colors ${
            rightTab === 'code' ? 'text-[#d4d4d4] border-[#2d8cf0]' : 'text-[#666] border-transparent hover:text-[#aaa]'
          }`}
        >
          <Code2 className="w-3 h-3" />
          Code
        </button>
        <div className="flex-1" />
        <button
          onClick={() => setRightOpen(false)}
          className="p-1 mr-1 text-[#555] hover:text-[#aaa] transition-colors"
          title="Close panel"
        >
          <PanelRightClose className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Inspector tab content */}
      {rightTab === 'inspector' && (
        <div className="flex-1 overflow-y-auto p-3">
          <PropertiesPanel />
        </div>
      )}

      {/* Code tab content */}
      {rightTab === 'code' && <CodePanel />}
    </aside>
  );
}
