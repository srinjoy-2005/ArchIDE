import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const CustomNode = ({ data, isConnectable }: any) => {
  return (
    <div className="px-4 py-2 shadow-lg rounded-md bg-slate-800/80 backdrop-blur-md border border-slate-600 min-w-[120px]">
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-blue-500 border-2 border-slate-900"
      />
      
      <div className="flex items-center">
        {data.icon && <div className="mr-2 text-slate-300">{data.icon}</div>}
        <div className="text-sm font-bold text-slate-100">{data.label}</div>
      </div>
      <div className="text-xs text-slate-400 mt-1">
        {data.description || 'Block layer'}
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-blue-500 border-2 border-slate-900"
      />
    </div>
  );
};

export default memo(CustomNode);
