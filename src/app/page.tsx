"use client";

/**
 * src/app/page.tsx
 *
 * Root page — a thin layout shell that composes the IDE's major panels.
 * All business logic has been extracted to focused component files:
 *
 *   Header         → src/components/Header.tsx
 *   BlockLibrary   → src/components/BlockLibrary.tsx
 *   DnDCanvas      → src/components/DnDCanvas.tsx
 *   RightPanel     → src/components/RightPanel.tsx
 *   DocPanels      → src/components/DocPanels.tsx
 *
 * The ReactFlowProvider wraps everything so that useReactFlow(), useNodes(),
 * and useEdges() hooks in any descendant component share the same internal store.
 */

import { ReactFlowProvider } from '@xyflow/react';
import { Header }                        from '../components/Header';
import { BlockLibrary }                  from '../components/BlockLibrary';
import { DnDCanvas }                     from '../components/DnDCanvas';
import { RightPanel }                    from '../components/RightPanel';
import { DocContextMenu, DocDetailsPanel } from '../components/DocPanels';

export default function Home() {
  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-full flex-col" style={{ background: '#181818', color: '#d4d4d4' }}>
        <Header />

        <div className="flex flex-1 overflow-hidden">
          {/* Left sidebar: draggable block library */}
          <BlockLibrary />

          {/* Doc details slide-in panel (appears left of canvas when open) */}
          <DocDetailsPanel />

          {/* Main canvas */}
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
