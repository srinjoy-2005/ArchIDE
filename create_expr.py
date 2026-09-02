content = """\"use client\";

import React, { useState, useRef, useEffect } from 'react';
import { useVFSStore } from '../lib/vfsStore';

export function ExpressionInput({
  value,
  onChange,
  expectedType,
  isReadOnly,
  inputClass,
  title,
  onDragOver,
  onDragLeave,
  onDrop
}: {
  value: any;
  onChange: (val: string | null) => void;
  expectedType: string;
  isReadOnly: boolean;
  inputClass: string;
  title: string;
  onDragOver: any;
  onDragLeave: any;
  onDrop: any;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [localValue, setLocalValue] = useState(String(value ?? ''));
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  
  const activeFileId = useVFSStore(s => s.activeFileId);
  const files = useVFSStore(s => s.files);
  const activeFile = files.find(f => f.id === activeFileId);
  const variables = activeFile?.variables || [];
  
  const inputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (!isEditing) {
      setLocalValue(value === null || value === undefined ? '' : String(value));
    }
  }, [value, isEditing]);

  const handleBlur = () => {
    setTimeout(() => {
      setIsEditing(false);
      setShowAutocomplete(false);
      
      const trimmed = localValue.trim();
      if (trimmed === '') {
        onChange(null); // \"Not Set\" concept
      } else {
        onChange(trimmed);
      }
    }, 200);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setLocalValue(val);
    
    const cursor = e.target.selectionStart || 0;
    const textBeforeCursor = val.slice(0, cursor);
    const match = textBeforeCursor.match(/@var:[a-zA-Z0-9_]*$/);
    
    if (match) {
      setShowAutocomplete(true);
    } else {
      setShowAutocomplete(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      inputRef.current?.blur();
    }
  };

  const insertVariable = (varName: string) => {
    const cursor = inputRef.current?.selectionStart || 0;
    const textBeforeCursor = localValue.slice(0, cursor);
    const textAfterCursor = localValue.slice(cursor);
    
    const match = textBeforeCursor.match(/@var:[a-zA-Z0-9_]*$/);
    if (match) {
      const newText = localValue.slice(0, match.index) + `@var:${varName} ` + textAfterCursor;
      setLocalValue(newText);
      setShowAutocomplete(false);
      inputRef.current?.focus();
    }
  };

  let validationError = '';
  if (localValue.trim() !== '') {
    let parsed = localValue;
    variables.forEach(v => {
      parsed = parsed.replace(new RegExp(`@var:${v.name}`, 'g'), String(v.default));
    });
    if (/^[0-9+\\-*/().\s]+$/.test(parsed)) {
      try {
        const result = new Function(`return ${parsed}`)();
        if (expectedType === 'int' && !Number.isInteger(result)) {
          validationError = 'Must evaluate to an integer';
        }
      } catch {
        validationError = 'Invalid expression';
      }
    }
  }

  const renderReadMode = () => {
    const parts = localValue.split(/(@var:[a-zA-Z0-9_]+)/g);
    return (
      <div 
        className={`${inputClass} overflow-hidden whitespace-nowrap cursor-text flex items-center min-h-[28px] ${localValue === '' || validationError ? 'border-red-500/50' : ''}`}
        onClick={() => !isReadOnly && setIsEditing(true)}
        title={validationError || title}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {localValue === '' ? (
          <span className="text-red-400/50 italic text-[10px]">Not Set</span>
        ) : (
          parts.map((p, i) => {
            if (p.startsWith('@var:')) {
              const vName = p.replace('@var:', '');
              return (
                <span key={i} className={`inline-flex items-center bg-[#6b21a8] text-[#e9d5ff] px-1 rounded-[2px] font-mono leading-none mx-0.5 text-[10px] ${validationError ? 'bg-red-900' : ''}`}>
                  {vName}
                </span>
              );
            }
            return <span key={i} className={validationError ? 'text-red-400' : ''}>{p}</span>;
          })
        )}
      </div>
    );
  };

  const compatibleVars = variables.filter(v => {
    if (expectedType === 'float') return v.type === 'float' || v.type === 'int';
    return v.type === expectedType;
  });

  return (
    <div className="relative w-full">
      {!isEditing ? renderReadMode() : (
        <>
          <input
            ref={inputRef}
            type="text"
            className={`${inputClass} ${validationError ? 'border-red-500 text-red-400' : ''}`}
            value={localValue}
            onChange={handleChange}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            autoFocus
          />
          {showAutocomplete && compatibleVars.length > 0 && (
            <div className="absolute top-full left-0 mt-1 w-full bg-[#252525] border border-[#3a3a3a] rounded-[3px] shadow-lg z-50 max-h-32 overflow-y-auto">
              <div className="text-[9px] text-[#888] px-2 py-1 uppercase tracking-wider border-b border-[#3a3a3a]">Variables</div>
              {compatibleVars.map(v => (
                <div 
                  key={v.id}
                  className="px-2 py-1.5 text-[11px] text-[#e2e2e2] font-mono hover:bg-[#3a3a3a] cursor-pointer flex justify-between"
                  onMouseDown={(e) => { e.preventDefault(); insertVariable(v.name); }}
                >
                  <span>{v.name}</span>
                  <span className="text-[#888]">{v.default}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
"""
with open('src/components/ExpressionInput.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
