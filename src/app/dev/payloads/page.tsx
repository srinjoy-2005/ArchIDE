"use client";

import React, { useEffect, useState } from "react";
import { Bug, Trash2, ArrowRightLeft } from "lucide-react";

interface PayloadLog {
  id: string;
  endpoint: string;
  request: any;
  response: any;
  timestamp: number;
}

export default function DevPayloadsPage() {
  const [logs, setLogs] = useState<PayloadLog[]>([]);
  const [selectedLogId, setSelectedLogId] = useState<string | null>(null);

  useEffect(() => {
    // Check if we are actually in development mode (as an extra safeguard)
    if (process.env.NODE_ENV !== "development") {
      return;
    }

    const channel = new BroadcastChannel("archide_payloads");
    
    channel.onmessage = (event) => {
      const { endpoint, request, response, timestamp } = event.data;
      const newLog: PayloadLog = {
        id: `log_${timestamp}_${Math.random().toString(36).substr(2, 9)}`,
        endpoint,
        request,
        response,
        timestamp,
      };
      
      setLogs((prev) => [newLog, ...prev]);
    };

    return () => {
      channel.close();
    };
  }, []);

  if (process.env.NODE_ENV !== "development") {
    return (
      <div className="min-h-screen bg-[#181818] flex items-center justify-center text-[#d4d4d4] font-mono text-sm">
        404 Not Found (Development Mode Only)
      </div>
    );
  }

  const selectedLog = logs.find((l) => l.id === selectedLogId) || logs[0];

  return (
    <div className="h-screen bg-[#181818] text-[#d4d4d4] font-sans flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#363636] bg-[#1e1e1e] flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-[#2d8cf0]/20 flex items-center justify-center border border-[#2d8cf0]/50">
            <Bug className="w-4 h-4 text-[#2d8cf0]" />
          </div>
          <div>
            <h1 className="text-[14px] font-semibold text-[#e2e2e2]">Payload Inspector</h1>
            <p className="text-[11px] text-[#888]">Monitor cross-boundary JSON payloads</p>
          </div>
        </div>
        <button
          onClick={() => {
            setLogs([]);
            setSelectedLogId(null);
          }}
          className="flex items-center gap-1.5 text-[12px] px-3 py-1.5 rounded bg-[#2a2a2a] hover:bg-[#363636] transition-colors border border-[#3a3a3a] text-[#aaa] hover:text-[#e54545]"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Clear Logs
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Sidebar: Log History */}
        <aside className="w-72 border-r border-[#363636] bg-[#1a1a1a] flex flex-col flex-shrink-0">
          <div className="p-3 text-[11px] uppercase tracking-wider text-[#666] font-semibold border-b border-[#2a2a2a]">
            Recent Requests ({logs.length})
          </div>
          <div className="flex-1 overflow-y-auto">
            {logs.length === 0 ? (
              <div className="p-6 text-center text-[#666] text-[12px] font-mono mt-10">
                Waiting for payloads...
                <br />
                <br />
                Trigger "Check Shapes" or "Export" in the main IDE.
              </div>
            ) : (
              logs.map((log) => (
                <button
                  key={log.id}
                  onClick={() => setSelectedLogId(log.id)}
                  className={`w-full text-left p-3 border-b border-[#2a2a2a] transition-colors ${
                    (selectedLog?.id === log.id) ? "bg-[#2d8cf0]/10 border-l-2 border-l-[#2d8cf0]" : "hover:bg-[#252525] border-l-2 border-l-transparent"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-[12px] font-mono font-bold ${log.response?.ok || log.response?.code ? "text-[#4ade80]" : "text-[#e54545]"}`}>
                      {log.endpoint}
                    </span>
                    <span className="text-[10px] text-[#666]">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-[11px] text-[#888] truncate">
                    Nodes: {log.request?.nodes?.length || 0} | Edges: {log.request?.edges?.length || 0}
                  </div>
                </button>
              ))
            )}
          </div>

          {/* Architecture Reference Cheat Sheet */}
          <div className="p-3 border-t border-[#2a2a2a] bg-[#151515] flex-shrink-0">
            <div className="text-[10px] uppercase tracking-wider text-[#666] font-semibold mb-2 flex items-center justify-between">
              <span>Architecture Map</span>
            </div>
            <div className="space-y-3 text-[10px] font-mono">
              <div>
                <span className="text-[#4ade80]">/api/check</span><br/>
                <span className="text-[#888]">→ </span><span className="text-[#d4d4d4]">shape_inference_pass()</span><br/>
                <span className="text-[#555] ml-3">backend/compiler.py</span>
              </div>
              <div>
                <span className="text-[#4ade80]">/api/compile</span><br/>
                <span className="text-[#888]">→ </span><span className="text-[#d4d4d4]">generate_pytorch_code()</span><br/>
                <span className="text-[#555] ml-3">backend/compiler.py</span>
              </div>
              <div>
                <span className="text-[#4ade80]">/api/blocks</span><br/>
                <span className="text-[#888]">→ </span><span className="text-[#d4d4d4]">get_all_block_defs()</span><br/>
                <span className="text-[#555] ml-3">backend/blocks/__init__.py</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Main: Payload Details */}
        <main className="flex-1 flex flex-col bg-[#1e1e1e] overflow-hidden">
          {selectedLog ? (
            <div className="flex-1 flex flex-col min-h-0">
              {/* Toolbar */}
              <div className="flex items-center gap-2 p-2 border-b border-[#363636] bg-[#252525]">
                <div className="px-3 py-1 rounded bg-[#1a1a1a] border border-[#3a3a3a] text-[12px] font-mono text-[#d4d4d4]">
                  {selectedLog.endpoint}
                </div>
                <ArrowRightLeft className="w-3.5 h-3.5 text-[#666]" />
                <div className="text-[11px] text-[#888]">
                  Captured at {new Date(selectedLog.timestamp).toLocaleString()}
                </div>
              </div>
              
              {/* Split View: Request & Response */}
              <div className="flex-1 flex overflow-hidden">
                {/* Request */}
                <div className="flex-1 flex flex-col border-r border-[#363636] min-h-0 min-w-0">
                  <div className="p-2 border-b border-[#363636] bg-[#2a2a2a] text-[11px] font-semibold text-[#aaa] tracking-wider uppercase">
                    Frontend Payload (Request)
                  </div>
                  <div className="flex-1 overflow-auto p-4">
                    <pre className="text-[11px] font-mono text-[#7ec8e3]">
                      {JSON.stringify(selectedLog.request, null, 2)}
                    </pre>
                  </div>
                </div>

                {/* Response */}
                <div className="flex-1 flex flex-col min-h-0 min-w-0">
                  <div className="p-2 border-b border-[#363636] bg-[#2a2a2a] text-[11px] font-semibold text-[#aaa] tracking-wider uppercase flex items-center justify-between">
                    <span>Backend Payload (Response)</span>
                    <span className={selectedLog.response?.ok || selectedLog.response?.code ? "text-[#4ade80]" : "text-[#e54545]"}>
                      {selectedLog.response?.ok || selectedLog.response?.code ? "SUCCESS" : "ERROR"}
                    </span>
                  </div>
                  <div className="flex-1 overflow-auto p-4">
                    <pre className={`text-[11px] font-mono ${selectedLog.response?.ok || selectedLog.response?.code ? "text-[#4ade80]" : "text-[#e54545]"}`}>
                      {JSON.stringify(selectedLog.response, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-[#555] font-mono text-sm">
              Select a request from the sidebar to inspect.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
