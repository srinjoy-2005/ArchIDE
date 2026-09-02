"use client";

/**
 * src/components/VariablesPanel.tsx
 *
 * Left-sidebar Variables panel (opened via the Variable icon in the Activity Bar).
 * Features:
 * - Lists all ArchVariables declared on the active graph file
 * - Create / rename (double-click) / delete with in-use safety warning
 * - Drag variables to PropertiesPanel param inputs to bind them (@var:<name>)
 * - Inline default value editing and type / scope selectors
 */

import React, { useState } from "react";
import { Plus, Trash2, GripVertical, Variable, AlertTriangle, X } from "lucide-react";
import { useVFSStore } from "../lib/store";
import type { ArchVariable, ArchVariableType, ArchVariableScope } from "../lib/types";

// ─── Constants ────────────────────────────────────────────────────────────────

const TYPE_OPTIONS: ArchVariableType[] = ["int", "float", "bool", "string"];
const SCOPE_OPTIONS: { value: ArchVariableScope; label: string; hint: string }[] = [
  { value: "init_param", label: "Init Param",  hint: "Emitted as __init__ argument (self.x = x)" },
  { value: "local_const", label: "Constant",   hint: "Emitted as module-level constant (X = ...)" },
];
const DEFAULT_VALUES: Record<ArchVariableType, any> = {
  int: 64, float: 0.1, bool: true, string: "",
};

const VAR_PREFIX = "@var:";

// ─── Deletion Warning Dialog ───────────────────────────────────────────────────

interface DeleteWarningProps {
  variable: ArchVariable;
  usedByNodes: { id: string; label: string; params: string[] }[];
  onConfirm: () => void;
  onCancel: () => void;
}

function DeleteWarningDialog({ variable, usedByNodes, onConfirm, onCancel }: DeleteWarningProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div
        className="flex flex-col gap-4 rounded-[4px] border border-[#e54545]/40 bg-[#1a1212] p-5 shadow-2xl"
        style={{ width: 360 }}
      >
        {/* Header */}
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-[#e54545]" />
          <div>
            <div className="text-[13px] font-semibold text-[#e2e2e2]">
              Variable in Use
            </div>
            <div className="mt-0.5 text-[11px] text-[#888]">
              <span className="font-mono text-[#c084fc]">{variable.name}</span> is bound
              to {usedByNodes.length} node{usedByNodes.length !== 1 ? "s" : ""}.
            </div>
          </div>
        </div>

        {/* Affected nodes */}
        <div className="flex flex-col gap-1.5 rounded-[3px] border border-[#333] bg-[#141414] p-2.5">
          <div className="text-[9px] uppercase tracking-wider text-[#555] mb-1">Affected Parameters</div>
          {usedByNodes.map((n) => (
            <div key={n.id} className="flex items-start gap-2">
              <span className="text-[10px] font-mono text-[#888] w-24 truncate flex-shrink-0">{n.label}</span>
              <div className="flex flex-wrap gap-1">
                {n.params.map((p) => (
                  <span key={p} className="text-[9px] font-mono text-[#e54545] bg-[#e54545]/10 px-1 rounded">{p}</span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Message */}
        <p className="text-[11px] text-[#aaa]">
          Deleting this variable will set all bound parameters to{" "}
          <span className="font-mono text-[#e2e2e2]">null</span>. You will need to
          reassign them before you can compile.
        </p>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={onConfirm}
            className="flex-1 rounded-[3px] border border-[#e54545]/50 bg-[#e54545]/20 py-1.5 text-[11px] text-[#e54545] transition-colors hover:bg-[#e54545]/30"
          >
            Delete & Clear Bindings
          </button>
          <button
            onClick={onCancel}
            className="flex-1 rounded-[3px] border border-[#363636] bg-[#252525] py-1.5 text-[11px] text-[#888] transition-colors hover:bg-[#2a2a2a]"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── VariablesPanel ───────────────────────────────────────────────────────────

export function VariablesPanel() {
  const activeFileId      = useVFSStore((s) => s.activeFileId);
  const files             = useVFSStore((s) => s.files);
  const addVariable       = useVFSStore((s) => s.addVariable);
  const updateVariable    = useVFSStore((s) => s.updateVariable);
  const deleteVariable    = useVFSStore((s) => s.deleteVariable);
  const clearVariableBindings = useVFSStore((s) => s.clearVariableBindings);

  const activeFile  = files.find((f) => f.id === activeFileId);
  const variables   = activeFile?.variables || [];
  const nodes       = activeFile?.nodes || [];
  const isGraphFile = activeFile && activeFile.fileType !== "code" && !activeFile.name.endsWith(".py");

  const [editingId, setEditingId]     = useState<string | null>(null);
  const [showAdd, setShowAdd]         = useState(false);
  const [newVarName, setNewVarName]   = useState("");
  const [newVarType, setNewVarType]   = useState<ArchVariableType>("int");
  const [newVarScope, setNewVarScope] = useState<ArchVariableScope>("init_param");
  const [deleteTarget, setDeleteTarget] = useState<ArchVariable | null>(null);

  // Find all nodes that bind a given variable name
  const getUsages = (varName: string) => {
    const binding = `${VAR_PREFIX}${varName}`;
    const result: { id: string; label: string; params: string[] }[] = [];
    for (const node of nodes) {
      const pv = (node.data?.paramValues as Record<string, any>) || {};
      const bound = Object.entries(pv)
        .filter(([, v]) => v === binding)
        .map(([k]) => k);
      if (bound.length > 0) {
        result.push({ id: node.id, label: (node.data?.label as string) || node.id, params: bound });
      }
    }
    return result;
  };

  const handleAdd = () => {
    if (!newVarName.trim()) return;
    addVariable(activeFileId, {
      name: newVarName.trim(), type: newVarType, scope: newVarScope,
      default: DEFAULT_VALUES[newVarType],
    });
    setNewVarName(""); setShowAdd(false);
  };

  const handleDeleteRequest = (variable: ArchVariable) => {
    const usages = getUsages(variable.name);
    if (usages.length > 0) {
      // Show warning dialog with affected nodes
      setDeleteTarget(variable);
    } else {
      // No usages — delete immediately
      deleteVariable(activeFileId, variable.id);
    }
  };

  const handleConfirmDelete = () => {
    if (!deleteTarget) return;
    clearVariableBindings(activeFileId, deleteTarget.name);
    deleteVariable(activeFileId, deleteTarget.id);
    setDeleteTarget(null);
  };

  const handleDragStart = (e: React.DragEvent, variable: ArchVariable) => {
    e.dataTransfer.setData("application/archide-variable", JSON.stringify(variable));
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <>
      {/* Deletion warning modal */}
      {deleteTarget && (
        <DeleteWarningDialog
          variable={deleteTarget}
          usedByNodes={getUsages(deleteTarget.name)}
          onConfirm={handleConfirmDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      <aside
        className="flex flex-col flex-shrink-0 overflow-hidden select-none"
        style={{ width: 220, background: "#191919", borderRight: "1px solid #282828" }}
      >
        {/* Panel Header */}
        <div
          className="flex items-center justify-between px-3 flex-shrink-0"
          style={{ height: 34, borderBottom: "1px solid #282828" }}
        >
          <div className="flex items-center gap-1.5">
            <Variable className="w-3.5 h-3.5 text-[#a855f7]" />
            <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#888]">
              Variables
            </span>
          </div>
          {isGraphFile && (
            <button
              onClick={() => setShowAdd((v) => !v)}
              className="flex items-center gap-1 text-[9px] text-[#666] hover:text-[#d4d4d4] transition-colors px-1.5 py-0.5 rounded hover:bg-[#252525]"
              title="Add Variable"
            >
              <Plus className="w-3 h-3" />
              Add
            </button>
          )}
        </div>

        {/* Panel Body */}
        <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
          {!isGraphFile && (
            <div className="text-[10px] text-[#444] italic text-center py-4">
              Switch to a graph file to manage variables.
            </div>
          )}

          {isGraphFile && (
            <>
              {/* Scope legend */}
              <div className="flex flex-col gap-1 mb-1">
                <div className="text-[9px] text-[#444] flex items-center gap-1">
                  <span className="font-mono text-[#a855f7]">init_param</span> → <code className="text-[#666]">__init__</code> arg
                </div>
                <div className="text-[9px] text-[#444] flex items-center gap-1">
                  <span className="font-mono text-[#a855f7]">constant</span> → module-level <code className="text-[#666]">CONST = …</code>
                </div>
              </div>

              {/* Add variable form */}
              {showAdd && (
                <div className="flex flex-col gap-2 bg-[#1a1a1a] border border-[#333] rounded-[3px] p-2">
                  <input
                    autoFocus
                    type="text"
                    value={newVarName}
                    onChange={(e) => setNewVarName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleAdd();
                      if (e.key === "Escape") setShowAdd(false);
                    }}
                    placeholder="variable_name"
                    className="w-full bg-[#1e1e1e] border border-[#3a3a3a] focus:border-[#a855f7] rounded-[3px] px-2 py-1 text-[11px] text-[#e2e2e2] font-mono outline-none"
                  />
                  <div className="flex gap-1.5">
                    <select
                      value={newVarType}
                      onChange={(e) => setNewVarType(e.target.value as ArchVariableType)}
                      className="flex-1 bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-1 py-1 text-[10px] text-[#aaa] outline-none"
                    >
                      {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <select
                      value={newVarScope}
                      onChange={(e) => setNewVarScope(e.target.value as ArchVariableScope)}
                      className="flex-1 bg-[#1e1e1e] border border-[#3a3a3a] rounded-[3px] px-1 py-1 text-[10px] text-[#aaa] outline-none"
                    >
                      {SCOPE_OPTIONS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
                    </select>
                  </div>
                  <div className="flex gap-1.5">
                    <button
                      onClick={handleAdd}
                      className="flex-1 bg-[#a855f7]/20 hover:bg-[#a855f7]/30 border border-[#a855f7]/40 rounded-[3px] py-1 text-[10px] text-[#c084fc] transition-colors"
                    >
                      Create
                    </button>
                    <button
                      onClick={() => setShowAdd(false)}
                      className="flex-1 bg-[#252525] border border-[#363636] rounded-[3px] py-1 text-[10px] text-[#666] transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Empty state */}
              {variables.length === 0 && !showAdd && (
                <div className="text-[10px] text-[#444] italic text-center py-4">
                  No variables. Click <span className="text-[#a855f7]">Add</span> to create one.
                </div>
              )}

              {/* Variables list */}
              {variables.map((v) => {
                const usages = getUsages(v.name);
                const inUse  = usages.length > 0;

                return (
                  <div
                    key={v.id}
                    draggable
                    onDragStart={(e) => handleDragStart(e, v)}
                    className="group flex flex-col gap-1.5 bg-[#1e1e1e] border border-[#2a2a2a] hover:border-[#a855f7]/30 rounded-[3px] px-2 pt-2 pb-1.5 cursor-grab active:cursor-grabbing transition-colors"
                  >
                    {/* Name row */}
                    <div className="flex items-center gap-1.5">
                      <GripVertical className="w-2.5 h-2.5 text-[#3a3a3a] flex-shrink-0" />
                      {editingId === v.id ? (
                        <input
                          autoFocus
                          type="text"
                          defaultValue={v.name}
                          className="flex-1 bg-transparent text-[11px] font-mono text-[#e2e2e2] outline-none border-b border-[#a855f7]"
                          onBlur={(e) => {
                            updateVariable(activeFileId, v.id, { name: e.target.value.trim() || v.name });
                            setEditingId(null);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === "Escape") {
                              updateVariable(activeFileId, v.id, { name: (e.target as HTMLInputElement).value.trim() || v.name });
                              setEditingId(null);
                            }
                          }}
                        />
                      ) : (
                        <span
                          className="flex-1 text-[11px] font-mono text-[#e2e2e2] truncate cursor-text"
                          onDoubleClick={() => setEditingId(v.id)}
                          title="Double-click to rename"
                        >
                          {v.name}
                        </span>
                      )}
                      {inUse && (
                        <span
                          className="text-[8px] font-mono text-[#a855f7] bg-[#a855f7]/10 px-1 rounded flex-shrink-0"
                          title={`Used by ${usages.length} node(s)`}
                        >
                          ×{usages.length}
                        </span>
                      )}
                      <button
                        onClick={() => handleDeleteRequest(v)}
                        className="opacity-0 group-hover:opacity-100 p-0.5 text-[#444] hover:text-[#e54545] transition-all flex-shrink-0"
                        title="Delete variable"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Type / scope badges + default value */}
                    <div className="flex items-center gap-1.5 pl-4">
                      <select
                        value={v.type}
                        onChange={(e) => updateVariable(activeFileId, v.id, { type: e.target.value as ArchVariableType })}
                        className="bg-transparent text-[9px] font-mono text-[#a855f7] outline-none cursor-pointer border-none"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
                      </select>
                      <span className="text-[#333]">·</span>
                      <select
                        value={v.scope}
                        onChange={(e) => updateVariable(activeFileId, v.id, { scope: e.target.value as ArchVariableScope })}
                        className="bg-transparent text-[9px] text-[#555] outline-none cursor-pointer border-none"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {SCOPE_OPTIONS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
                      </select>
                      <span className="flex-1" />
                      <input
                        type={v.type === "int" || v.type === "float" ? "number" : "text"}
                        value={v.default as any ?? ""}
                        onChange={(e) => {
                          const raw = e.target.value;
                          const val = v.type === "int" ? parseInt(raw) : v.type === "float" ? parseFloat(raw) : raw;
                          updateVariable(activeFileId, v.id, { default: val });
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="w-12 bg-[#151515] border border-[#2a2a2a] rounded-[3px] px-1 py-0.5 text-[9px] font-mono text-[#888] outline-none text-center"
                        title="Default value"
                      />
                    </div>

                    {/* In-use hint */}
                    {inUse && (
                      <div className="pl-4 text-[9px] text-[#555] italic">
                        Bound to: {usages.map((u) => u.label).join(", ")}
                      </div>
                    )}
                  </div>
                );
              })}

              {variables.length > 0 && (
                <p className="text-[9px] text-[#3a3a3a] italic mt-1">
                  Drag a variable onto a param field to bind it.
                </p>
              )}
            </>
          )}
        </div>
      </aside>
    </>
  );
}
