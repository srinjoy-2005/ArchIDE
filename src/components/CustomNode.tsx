import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const CustomNode = ({ data, isConnectable }: any) => {
  const inputs = data.inputs || [{ id: 'in', name: 'Input' }];
  const outputs = data.outputs || [{ id: 'out', name: 'Output' }];

  return (
    <div className="shadow-lg rounded-md bg-slate-800/90 backdrop-blur-md border border-slate-600 min-w-[150px] flex flex-col relative">
      {/* Left side inputs */}
      <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-evenly -ml-1.5 z-10">
        {inputs.map((input: any) => (
          <div key={input.id} className="relative group flex items-center">
            <Handle
              type="target"
              position={Position.Left}
              id={input.id}
              isConnectable={isConnectable}
              className="w-3 h-3 bg-blue-400 border-2 border-slate-900 !relative !transform-none"
            />
            <span className="absolute left-4 text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap bg-slate-900 px-1 py-0.5 rounded">
              {input.name}
            </span>
          </div>
        ))}
      </div>

      <div className="px-4 py-3 flex-1 flex flex-col justify-center">
        <div className="flex items-center justify-center">
          {data.icon && <div className="mr-2 text-slate-300">{data.icon}</div>}
          <div className="text-sm font-bold text-slate-100 text-center">{data.label}</div>
        </div>
      </div>

      {/* Right side outputs */}
      <div className="absolute right-0 top-0 bottom-0 flex flex-col justify-evenly -mr-1.5 z-10">
        {outputs.map((output: any) => (
          <div key={output.id} className="relative group flex items-center">
            <span className="absolute right-4 text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap bg-slate-900 px-1 py-0.5 rounded">
              {output.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={output.id}
              isConnectable={isConnectable}
              className="w-3 h-3 bg-indigo-400 border-2 border-slate-900 !relative !transform-none"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default memo(CustomNode);
