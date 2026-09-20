"use client";

/**
 * src/components/ActivityBar.tsx
 *
 * Far-left vertical icon bar inspired by modern IDEs / OneCompiler.
 * Provides quick switching between:
 * - Explorer (File & Folder tree)
 * - Block Library (Neural network layer palette)
 * - Search (Graph & block search)
 * - Settings (Preferences)
 */

import React from 'react';
import { useEditorStore, type SidebarView } from '../lib/store';
import { FolderTree, Layers, Variable, Settings } from 'lucide-react';

interface ActivityTab {
  id: SidebarView;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const TOP_TABS: ActivityTab[] = [
  { id: 'explorer',  label: 'Explorer (Files & Folders)',   icon: FolderTree },
  { id: 'library',   label: 'Block Library (Layers & Ops)', icon: Layers },
  { id: 'variables', label: 'Variables',                    icon: Variable },
];

export function ActivityBar() {
  const activeSidebarView = useEditorStore((s) => s.activeSidebarView);
  const setActiveSidebarView = useEditorStore((s) => s.setActiveSidebarView);
  const sidebarOpen = useEditorStore((s) => s.sidebarOpen);
  const toggleSidebar = useEditorStore((s) => s.toggleSidebar);

  const handleTabClick = (tabId: SidebarView) => {
    if (activeSidebarView === tabId && sidebarOpen) {
      toggleSidebar();
    } else {
      setActiveSidebarView(tabId);
    }
  };

  return (
    <div
      className="flex flex-col items-center justify-between py-2 flex-shrink-0 z-20 select-none"
      style={{
        width: 44,
        background: '#141414',
        borderRight: '1px solid #282828',
      }}
    >
      {/* Top action icons */}
      <div className="flex flex-col items-center gap-1 w-full">
        {TOP_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = sidebarOpen && activeSidebarView === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => handleTabClick(tab.id)}
              title={tab.label}
              className={`group relative flex items-center justify-center w-9 h-9 rounded-[4px] transition-colors ${
                isActive
                  ? 'text-[#2d8cf0] bg-[#222222]'
                  : 'text-[#777] hover:text-[#e2e2e2] hover:bg-[#1c1c1c]'
              }`}
            >
              {/* Active left indicator bar */}
              {isActive && (
                <span
                  className="absolute left-0 top-1.5 bottom-1.5 w-[2px] bg-[#2d8cf0] rounded-r"
                />
              )}
              <Icon className="w-4 h-4 transition-transform group-hover:scale-105" />
            </button>
          );
        })}
      </div>

      {/* Bottom icons */}
      <div className="flex flex-col items-center gap-1 w-full">
        <button
          title="Settings"
          className="flex items-center justify-center w-9 h-9 rounded-[4px] text-[#666] hover:text-[#bbb] hover:bg-[#1c1c1c] transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
