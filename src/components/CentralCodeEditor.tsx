"use client";

/**
 * src/components/CentralCodeEditor.tsx
 *
 * Full-workspace Python code viewer and editor.
 * Displays compiled PyTorch module code or raw Python script files with:
 * - Line numbering gutter
 * - Syntax color styling
 * - Quick copy & download actions
 */

import React, { useState } from 'react';
import { Copy, Check, Download, FileCode, Play, Sparkles } from 'lucide-react';
import { useEditorStore } from '../lib/store';

export function CentralCodeEditor() {
  const generatedCode = useEditorStore((s) => s.generatedCode);
  const activeFileId = useEditorStore((s) => s.activeFileId);
  const files = useEditorStore((s) => s.files);
  const activeFile = files.find((f) => f.id === activeFileId);

  const [copied, setCopied] = useState(false);

  const codeToShow = activeFile?.compiledCode || generatedCode || `# import torch\n# import torch.nn as nn\n\n# Your compiled PyTorch code will appear here.`;
  const lines = codeToShow.split('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(codeToShow);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const fileName = activeFile?.name ? (activeFile.name.endsWith('.py') ? activeFile.name : activeFile.name.replace(/\.[^/.]+$/, "") + ".py") : "model.py";
    const blob = new Blob([codeToShow], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#181818] overflow-hidden select-text">
      {/* Code Editor Toolbar */}
      <div
        className="flex items-center justify-between px-4 flex-shrink-0 bg-[#1e1e1e] border-b border-[#282828]"
        style={{ height: 36 }}
      >
        <div className="flex items-center gap-2">
          <FileCode className="w-4 h-4 text-[#eab308]" />
          <span className="text-[12px] font-mono text-[#d4d4d4] font-medium">
            {activeFile?.name ? (activeFile.name.endsWith('.py') ? activeFile.name : activeFile.name.replace(/\.[^/.]+$/, "") + ".py") : "model.py"}
          </span>
          <span className="text-[10px] text-[#666666] bg-[#262626] px-1.5 py-0.5 rounded ml-1.5">
            PyTorch 2.x
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={handleCopy}
            title="Copy code to clipboard"
            className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-[#999] hover:text-[#f0f0f0] bg-[#252525] hover:bg-[#303030] rounded transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-[#4ade80]" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            onClick={handleDownload}
            title="Download .py file"
            className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-[#999] hover:text-[#f0f0f0] bg-[#252525] hover:bg-[#303030] rounded transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download
          </button>
        </div>
      </div>

      {/* Code Area with Line Numbers */}
      <div className="flex-1 flex overflow-auto font-mono text-[12px] leading-[20px] bg-[#141414]">
        {/* Line Numbers Gutter */}
        <div className="py-3 px-3 text-right text-[#4a4a4a] select-none bg-[#161616] border-r border-[#222222] min-w-[44px]">
          {lines.map((_, idx) => (
            <div key={idx} className="h-[20px]">
              {idx + 1}
            </div>
          ))}
        </div>

        {/* Code Content */}
        <div className="flex-1 p-3 overflow-x-auto">
          <pre
            className="font-mono text-[12.5px] leading-[20px]"
            style={{
              color: codeToShow.startsWith('# ❌') ? '#f87171' : '#9cdcfe',
            }}
          >
            {codeToShow}
          </pre>
        </div>
      </div>
    </div>
  );
}
