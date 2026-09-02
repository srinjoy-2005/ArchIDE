"use client";

/**
 * src/components/VariablesPanel.tsx
 *
 * Panel for managing per-file ArchVariables. Variables can be dragged from
 * this panel and dropped onto parameter inputs in PropertiesPanel to bind a
 * param to the variable — the compiler emits it as a constructor argument or
 * local constant instead of a hard-coded literal.
 */

import React, { useState } from "react";
import { Plus, Trash2, GripVertical, Variable } from "lucide-react";
import { useVFSStore } from "../lib/store";
import type { ArchVariable, ArchVariableType, ArchVariableScope } from "../lib/types";

const TYPE_OPTIONS: ArchVariableType[] = ["int", "float", "bool", "string"];
const SCOPE_OPTIONS: { value: ArchVariableScope; label: string }[] = [
  { value: "init_param", label: "Init Param" },
  { value: "local_const", label: "Constant" },
];
const DEFAULT_VALUES: Record<ArchVariableType, any> = {
  int: 64, float: 0.1, bool: true, string: "",
};

export function VariablesPanel() {
  const activeFileId   = useVFSStore((s) => s.activeFileId);
  const files          = useVFSStore((s) => s.files);
  const addVariable    = useVFSStore((s) => s.addVariable);
  const updateVariable = useVFSStore((s) => s.updateVariable);
  const deleteVariable = useVFSStore((s) => s.deleteVariable);

  const activeFile  = files.find((f) => f.id === activeFileId);
  const variables   = activeFile?.variables || [];
  const isGraphFile = activeFile?.fileType !== "code" && !activeFile?.name.endsWith(".py");

  const [editingId, setEditingId]     = useState<string | null>(null);
  const [showAdd, setShowAdd]         = useState(false);
  const [newVarName, setNewVarName]   = useState("");
  const [newVarType, setNewVarType]   = useState<ArchVariableType>("int");
  const [newVarScope, setNewVarScope] = useState<ArchVariableScope>("init_param");

  if (!isGraphFile) return null;

  const handleAdd = () => {
    if (!newVarName.trim()) return;
    addVariable(activeFileId, {
      name: newVarName.trim(), type: newVarType, scope: newVarScope,
      default: DEFAULT_VALUES[newVarType],
    });
    setNewVarName(""); setShowAdd(false);
  };

  const handleDragStart = (e: React.DragEvent, variable: ArchVariable) => {
    e.dataTransfer.setData("application/archide-variable", JSON.stringify(variable));
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <div className="flex flex-col gap-2 border-t border-[#282828] pt-3 mt-1">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[#888]">
          <Variable className="w-3 h-3 text-[#a855f7]" />
          <span className="text-[10px] font-semibold uppercase tracking-wider">Variables</span>
        </div>
        <button onClick={() => setShowAdd((v) => !v)}
          className="flex items-center gap-1 text-[9px] text-[#666] hover:text-[#d4d4d4] transition-colors px-1.5 py-0.5 rounded hover:bg-[#252525]">
          <Plus className="w-3 h-3" /> Add
        </button>
      </div>

      {/* Add form */}
      {showAdd && (
        <div className="flex flex-col gap-2 bg-[#1a1a1a] border border-[#333] rounded-[3px] p-2">
          <input autoFocus type="text" value={newVarName} onChange={(e) => setNewVarName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleAdd(); if (e.key === "Escape") setShowAdd(false); }}
            placeholder="variable_name"
            className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#a855f7] rounded-[3px] px-2 py-1 text-[11px] text-[#e2e2e2] font-mono outline-none" />
          <div className="flex gap-2">
            <select value={newVarType} onChange={(e) => setNewVarType(e.target.value as ArchVariableType)}
              className="flex-1 bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-1.5 py-1 text-[10px] text-[#aaa] outline-none">
              {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <select value={newVarScope} onChange={(e) => setNewVarScope(e.target.value as ArchVariableScope)}
              className="flex-1 bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-1.5 py-1 text-[10px] text-[#aaa] outline-none">
              {SCOPE_OPTIONS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>
          <div className="flex gap-1.5">
            <button onClick={handleAdd}
              className="flex-1 bg-[#a855f7]/20 hover:bg-[#a855f7]/30 border border-[#a855f7]/40 rounded-[3px] py-1 text-[10px] text-[#c084fc] transition-colors">
              Create
            </button>
            <button onClick={() => setShowAdd(false)}
              className="flex-1 bg-[#252525] border border-[#363636] rounded-[3px] py-1 text-[10px] text-[#666] transition-colors">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {variables.length === 0 && !showAdd && (
        <div className="text-[10px] text-[#444] italic text-center py-2">No variables yet. Add one to bind params.</div>
      )}

      {/* Variable list */}
      {variables.map((v) => (
        <div key={v.id} draggable onDragStart={(e) => handleDragStart(e, v)}
          className="group flex items-center gap-2 bg-[#1e1e1e] border border-[#333] hover:border-[#a855f7]/40 rounded-[3px] px-2 py-1.5 cursor-grab active:cursor-grabbing transition-colors">
          <GripVertical className="w-2.5 h-2.5 text-[#444] flex-shrink-0" />
          <div className="flex-1 min-w-0">
            {editingId === v.id ? (
              <input autoFocus type="text" defaultValue={v.name}
                className="w-full bg-transparent text-[11px] font-mono text-[#e2e2e2] outline-none border-b border-[#a855f7]"
                onBlur={(e) => { updateVariable(activeFileId, v.id, { name: e.target.value.trim() || v.name }); setEditingId(null); }}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === "Escape") { updateVariable(activeFileId, v.id, { name: (e.target as HTMLInputElement).value.trim() || v.name }); setEditingId(null); } }} />
            ) : (
              <span className="text-[11px] font-mono text-[#e2e2e2] truncate block cursor-text"
                onDoubleClick={() => setEditingId(v.id)}>{v.name}</span>
            )}
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="text-[9px] font-mono text-[#a855f7] bg-[#a855f7]/10 px-1 rounded">{v.type}</span>
              <span className="text-[9px] text-[#555]">{v.scope === "init_param" ? "param" : "const"}</span>
            </div>
          </div>
          <input
            type={v.type === "int" || v.type === "float" ? "number" : "text"}
            value={v.default as any ?? ""}
            onChange={(e) => {
              const raw = e.target.value;
              const val = v.type === "int" ? parseInt(raw) : v.type === "float" ? parseFloat(raw) : raw;
              updateVariable(activeFileId, v.id, { default: val });
            }}
            className="w-14 bg-[#151515] border border-[#2a2a2a] rounded-[3px] px-1.5 py-0.5 text-[10px] font-mono text-[#aaa] outline-none text-center" />
          <button onClick={() => deleteVariable(activeFileId, v.id)}
            className="opacity-0 group-hover:opacity-100 p-0.5 text-[#555] hover:text-[#e54545] transition-all">
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      ))}

      {variables.length > 0 && (
        <p className="text-[9px] text-[#444] italic">Drag a variable onto a parameter field to bind it.</p>
      )}
    </div>
  );
}
