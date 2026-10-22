"use client";

/**
 * src/components/CentralCodeEditor.tsx
 *
 * Full-workspace Python code viewer and bi-directional editor.
 * Features:
 * - Editable code surface with line numbers and tab indentation support
 * - "Sync to Graph" action powered by the backend PyTorch AST Decompiler
 * - Explicit tooltips and status feedback indicating AST decompiler usage
 * - Quick copy, download, and switch to visual canvas
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Copy,
  Check,
  Download,
  FileCode,
  Sparkles,
  RefreshCw,
  AlertTriangle,
  ArrowRight,
  Info,
} from 'lucide-react';
import { useEditorStore, useVFSStore } from '../lib/store';
import { API_BASE } from '../lib/constants';

export function CentralCodeEditor() {
  const generatedCode = useEditorStore((s) => s.generatedCode);
  const activeFileId = useVFSStore((s) => s.activeFileId);
  const files = useVFSStore((s) => s.files);
  const folders = useVFSStore((s) => s.folders);
  const graphsFolderId = useVFSStore((s) => s.graphsFolderId);
  const updateFileCode = useVFSStore((s) => s.updateFileCode);
  const overwriteFilesFromVFS = useVFSStore((s) => s.overwriteFilesFromVFS);
  const switchFile = useVFSStore((s) => s.switchFile);

  const activeFile = files.find((f) => f.id === activeFileId);

  const [copied, setCopied] = useState(false);
  const [isDecompiling, setIsDecompiling] = useState(false);
  const [decompileStatus, setDecompileStatus] = useState<{
    type: 'idle' | 'success' | 'error';
    message?: string;
    targetGraphId?: string;
  }>({ type: 'idle' });

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);

  // Default fallback code if empty
  const initialCode =
    activeFile?.compiledCode ??
    generatedCode ??
    `import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self, in_features: int = 64, out_features: int = 10):\n        super().__init__()\n        self.fc = nn.Linear(in_features, out_features)\n\n    def forward(self, x):\n        return self.fc(x)\n`;

  const [code, setCode] = useState(initialCode);

  // Keep local code in sync when activeFile changes
  useEffect(() => {
    const updated = activeFile?.compiledCode ?? generatedCode ?? '';
    if (updated !== undefined) {
      setCode(updated);
    }
    setDecompileStatus({ type: 'idle' });
  }, [activeFileId, activeFile?.compiledCode, generatedCode]);

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newCode = e.target.value;
    setCode(newCode);
    if (activeFileId) {
      updateFileCode(activeFileId, newCode);
    }
    if (decompileStatus.type !== 'idle') {
      setDecompileStatus({ type: 'idle' });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Tab key inserts 4 spaces
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const val = target.value;
      const updated = val.substring(0, start) + '    ' + val.substring(end);
      setCode(updated);
      if (activeFileId) {
        updateFileCode(activeFileId, updated);
      }
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      }, 0);
    }
    // Ctrl+Enter or Cmd+Enter triggers decompile to graph
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleDecompile();
    }
  };

  const handleScroll = () => {
    if (textareaRef.current && gutterRef.current) {
      gutterRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const fileName = activeFile?.name || 'model.py';
    const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  };

  /**
   * Decompiles current Python code into visual graph representation (.arch)
   * using the backend PyTorch AST Decompiler.
   */
  const handleDecompile = async () => {
    if (!code.trim() || isDecompiling) return;

    setIsDecompiling(true);
    setDecompileStatus({ type: 'idle' });

    try {
      const fileName = activeFile?.name || 'main.py';
      const fileStem = fileName.replace(/\.py$/, '');

      const response = await fetch(`${API_BASE}/api/decompile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          file_name: fileName,
          validate_graph: true,
        }),
      });

      const data = await response.json();

      if (data.success && data.graphs) {
        // Sync decompiled graphs into VFS store
        overwriteFilesFromVFS(data.graphs);

        const graphKeys = Object.keys(data.graphs);
        const targetGraphKey = data.main_graph_id || fileStem;

        // Locate corresponding graph file in VFS to allow quick jump
        const targetGraphFile = files.find(
          (f) =>
            (f.name === `${targetGraphKey}.arch` || f.name === `${fileStem}.arch` || f.id === targetGraphKey) &&
            f.fileType !== 'code'
        );

        setDecompileStatus({
          type: 'success',
          message: `Decompiled ${graphKeys.length} model class${graphKeys.length > 1 ? 'es' : ''} using PyTorch AST Decompiler!`,
          targetGraphId: targetGraphFile?.id,
        });
      } else {
        setDecompileStatus({
          type: 'error',
          message: data.error || 'Decompiler could not parse Python code into an nn.Module graph.',
        });
      }
    } catch (err: any) {
      setDecompileStatus({
        type: 'error',
        message: err.message || 'Failed to connect to backend decompiler API.',
      });
    } finally {
      setIsDecompiling(false);
    }
  };

  const lines = code.split('\n');
  const isJson = activeFile?.name?.endsWith('.json');
  const isToml = activeFile?.name?.endsWith('.toml');
  const isPython = !isJson && !isToml;
  const badgeLabel = isJson ? 'JSON IR' : isToml ? 'TOML Config' : 'PyTorch 2.x';

  return (
    <div className="flex-1 flex flex-col h-full bg-[#181818] overflow-hidden select-text">
      {/* Code Editor Toolbar */}
      <div
        className="flex items-center justify-between px-4 flex-shrink-0 bg-[#1e1e1e] border-b border-[#282828]"
        style={{ height: 38 }}
      >
        <div className="flex items-center gap-2">
          <FileCode className="w-4 h-4 text-[#eab308]" />
          <span className="text-[12px] font-mono text-[#d4d4d4] font-medium">
            {activeFile?.name || 'model.py'}
          </span>
          <span className="text-[10px] text-[#888888] bg-[#262626] px-1.5 py-0.5 rounded ml-1">
            {badgeLabel}
          </span>

          {/* Decompiler Badge & Tooltip */}
          {isPython && (
            <div
              className="group relative flex items-center gap-1 ml-2 px-2 py-0.5 bg-[#14231b] border border-[#1b432a] text-[#4ade80] rounded text-[11px] cursor-help"
              title="PyTorch AST Decompiler: Parses Python code into visual graph blocks & connections via AST and Agentic IR."
            >
              <Sparkles className="w-3 h-3 text-[#4ade80]" />
              <span className="font-sans font-medium text-[10.5px]">Decompiler Active</span>

              {/* Informative hover tooltip */}
              <div className="absolute left-0 top-full mt-1.5 hidden group-hover:block z-50 w-72 p-2.5 bg-[#1f1f1f] border border-[#333333] rounded-md shadow-xl text-[11px] leading-[15px] text-[#cccccc] font-sans">
                <div className="font-semibold text-[#4ade80] mb-1 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  PyTorch AST Decompiler
                </div>
                Converts your Python <code className="text-[#9cdcfe]">nn.Module</code> code back into visual canvas blocks and connections. Edit code here and click{' '}
                <strong className="text-white">Sync to Graph</strong> to update your canvas.
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Decompile to Graph Action Button */}
          {isPython && (
            <div className="group relative">
              <button
                onClick={handleDecompile}
                disabled={isDecompiling}
                title="Decompile Python code to Graph (Ctrl+Enter)"
                className={`flex items-center gap-1.5 px-3 py-1 text-[11px] font-medium rounded transition-all shadow-sm ${
                  isDecompiling
                    ? 'bg-[#2a2a2a] text-[#777777] cursor-wait'
                    : 'bg-[#1e3a2b] hover:bg-[#254b37] text-[#4ade80] border border-[#2d5d42] active:scale-[0.98]'
                }`}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isDecompiling ? 'animate-spin' : ''}`} />
                <span>{isDecompiling ? 'Decompiling AST...' : 'Sync to Graph'}</span>
              </button>

              {/* Action Tooltip */}
              <div className="absolute right-0 top-full mt-1.5 hidden group-hover:block z-50 w-64 p-2 bg-[#202020] border border-[#3a3a3a] rounded shadow-lg text-[11px] text-[#aaaaaa] font-sans">
                Runs <strong className="text-[#4ade80]">PyTorchASTDecompiler</strong> to parse classes, layers, and functional tensor ops into graph nodes & edges.
              </div>
            </div>
          )}

          <div className="h-4 w-[1px] bg-[#333333] mx-0.5" />

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

      {/* Decompiler Feedback Notification Banner */}
      {decompileStatus.type === 'success' && (
        <div className="flex items-center justify-between px-4 py-2 bg-[#122b1e] border-b border-[#1d4a32] text-[#4ade80] text-[12px]">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4 text-[#4ade80]" />
            <span>{decompileStatus.message}</span>
          </div>
          {decompileStatus.targetGraphId && (
            <button
              onClick={() => switchFile(decompileStatus.targetGraphId!)}
              className="flex items-center gap-1 text-[11px] bg-[#1a402b] hover:bg-[#235338] px-2.5 py-0.5 rounded text-[#e0e0e0] font-medium transition-colors"
            >
              Open Graph Tab <ArrowRight className="w-3 h-3" />
            </button>
          )}
        </div>
      )}

      {decompileStatus.type === 'error' && (
        <div className="flex items-start gap-2 px-4 py-2 bg-[#311515] border-b border-[#5a2424] text-[#f87171] text-[12px]">
          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0 text-[#f87171]" />
          <div className="flex-1">
            <span className="font-semibold">Decompiler Error: </span>
            <span className="font-mono text-[11.5px]">{decompileStatus.message}</span>
          </div>
        </div>
      )}

      {/* Code Editor Surface with Synchronized Line Numbers */}
      <div className="flex-1 flex overflow-hidden font-mono text-[12px] leading-[20px] bg-[#141414] relative">
        {/* Line Numbers Gutter */}
        <div
          ref={gutterRef}
          className="py-3 px-3 text-right text-[#4a4a4a] select-none bg-[#161616] border-r border-[#222222] min-w-[48px] overflow-hidden"
        >
          {lines.map((_, idx) => (
            <div key={idx} className="h-[20px]">
              {idx + 1}
            </div>
          ))}
        </div>

        {/* Editable Textarea */}
        <div className="flex-1 relative h-full">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={handleCodeChange}
            onKeyDown={handleKeyDown}
            onScroll={handleScroll}
            spellCheck={false}
            className="w-full h-full p-3 font-mono text-[12.5px] leading-[20px] text-[#9cdcfe] bg-[#141414] resize-none outline-none overflow-auto border-none focus:ring-0 whitespace-pre"
            style={{ tabSize: 4 }}
          />
        </div>
      </div>
    </div>
  );
}
