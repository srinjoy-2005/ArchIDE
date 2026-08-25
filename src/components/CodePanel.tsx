"use client";

/**
 * src/components/CodePanel.tsx
 *
 * The "Code" tab in the right panel. Reads `generatedCode` from the Zustand store
 * and displays it as syntax-highlighted monospace text. Provides Copy and Download
 * actions for the generated PyTorch code.
 */

import { useState } from 'react';
import { Copy, Check, Download } from 'lucide-react';
import { useEditorStore } from '../lib/store';

export function CodePanel() {
  const generatedCode = useEditorStore((s) => s.generatedCode);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Code actions toolbar */}
      <div
        className="flex items-center gap-1.5 px-3 flex-shrink-0"
        style={{ height: 32, borderBottom: '1px solid #363636' }}
      >
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#e2e2e2] transition-colors"
        >
          {copied ? <Check className="w-3 h-3 text-[#4ade80]" /> : <Copy className="w-3 h-3" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        <div className="w-px h-3 bg-[#363636]" />
        <button
          onClick={handleDownload}
          className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#e2e2e2] transition-colors"
        >
          <Download className="w-3 h-3" />
          Download
        </button>
      </div>

      {/* Code display */}
      <div className="flex-1 overflow-auto p-3">
        <pre
          className="text-[11px] font-mono leading-relaxed"
          style={{ color: generatedCode.startsWith('# ❌') ? '#e54545' : '#7ec8e3' }}
        >
          {generatedCode || '# Export PyTorch code will appear here after compiling.'}
        </pre>
      </div>
    </div>
  );
}
