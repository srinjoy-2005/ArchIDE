"use client";

/**
 * src/app/page.tsx
 *
 * Root page — a thin layout shell that composes the IDE's major panels:
 *
 *   Header         → src/components/Header.tsx
 *   ActivityBar    → src/components/ActivityBar.tsx (Far-left view switcher)
 *   FileExplorer   → src/components/FileExplorer.tsx (OneCompiler-style VFS tree)
 *   BlockLibrary   → src/components/BlockLibrary.tsx (Neural network layer palette)
 *   DnDCanvas      → src/components/DnDCanvas.tsx (Multi-tab canvas)
 *   RightPanel     → src/components/RightPanel.tsx (Inspector + Code preview)
 *   DocPanels      → src/components/DocPanels.tsx (Documentation overlays)
 */

import { ReactFlowProvider } from '@xyflow/react';
import { Header }                        from '../components/Header';
import { ActivityBar }                   from '../components/ActivityBar';
import { FileExplorer }                  from '../components/FileExplorer';
import { BlockLibrary }                  from '../components/BlockLibrary';
import { DnDCanvas }                     from '../components/DnDCanvas';
import { RightPanel }                    from '../components/RightPanel';
import { DocContextMenu, DocDetailsPanel } from '../components/DocPanels';
import { useEditorStore }                from '../lib/store';

function LeftSidebar() {
  const activeSidebarView = useEditorStore((s) => s.activeSidebarView);
  const sidebarOpen       = useEditorStore((s) => s.sidebarOpen);

  if (!sidebarOpen) return null;

  if (activeSidebarView === 'library') {
    return <BlockLibrary />;
  }

  // Default to File/Folder Explorer
  return <FileExplorer />;
}

export default function Home() {
  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col select-none" style={{ background: '#181818', color: '#d4d4d4' }}>
        <Header />

        <div className="flex flex-1 overflow-hidden">
          {/* Far-left Activity Bar (OneCompiler / IDE style) */}
          <ActivityBar />

          {/* Left Primary Sidebar Drawer (Explorer or Block Library) */}
          <LeftSidebar />

          {/* Doc details slide-in panel (appears left of canvas when open) */}
          <DocDetailsPanel />

          {/* Main multi-tab canvas */}
          <DnDCanvas />

          {/* Right panel: Inspector + Code tabs */}
          <RightPanel />
        </div>
      </div>

      {/* Floating context menu overlay for block docs */}
      <DocContextMenu />
    </ReactFlowProvider>
  );
}
