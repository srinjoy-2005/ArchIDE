import { Node, Edge } from '@xyflow/react';

export function generatePyTorchCode(nodes: Node[], edges: Edge[]): string {
  if (nodes.length === 0) {
    return "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return x";
  }

  // 1. Find the input node
  const inputNode = nodes.find((n) => n.data.label === 'Input');
  
  let initLines: string[] = [];
  let forwardLines: string[] = [];
  
  // Create layer definitions for all non-input nodes
  nodes.forEach((node, index) => {
    if (node.data.label === 'Input') return;
    
    const layerName = `${node.data.label?.toString().toLowerCase()}_${index}`;
    node.data.layerName = layerName; // Save for forward pass
    
    switch (node.data.label) {
      case 'Linear':
        initLines.push(`        self.${layerName} = nn.Linear(128, 64) # Default shapes`);
        break;
      case 'Conv2D':
        initLines.push(`        self.${layerName} = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)`);
        break;
      case 'ReLU':
        initLines.push(`        self.${layerName} = nn.ReLU()`);
        break;
      case 'Sigmoid':
        initLines.push(`        self.${layerName} = nn.Sigmoid()`);
        break;
      case 'Tanh':
        initLines.push(`        self.${layerName} = nn.Tanh()`);
        break;
      case 'Softmax':
        initLines.push(`        self.${layerName} = nn.Softmax(dim=1)`);
        break;
      case 'Dropout':
        initLines.push(`        self.${layerName} = nn.Dropout(p=0.5)`);
        break;
      case 'BatchNorm2d':
        initLines.push(`        self.${layerName} = nn.BatchNorm2d(16)`);
        break;
      case 'MaxPool2d':
        initLines.push(`        self.${layerName} = nn.MaxPool2d(kernel_size=2)`);
        break;
      case 'AvgPool2d':
        initLines.push(`        self.${layerName} = nn.AvgPool2d(kernel_size=2)`);
        break;
      default:
        initLines.push(`        self.${layerName} = nn.Identity() # Unknown layer: ${node.data.label}`);
    }
  });

  // 2. Build forward pass using simple topological traversal
  // For PoC, we just follow the edges from the input node
  let currentVar = 'x';
  let currentVarMap = new Map<string, string>();
  
  if (inputNode) {
    currentVarMap.set(inputNode.id, 'x');
  }

  // Very naive edge traversal for sequential models
  let processedNodes = new Set<string>();
  if (inputNode) processedNodes.add(inputNode.id);

  let forwardQueue = edges.filter(e => e.source === inputNode?.id);
  
  while (forwardQueue.length > 0) {
    const edge = forwardQueue.shift()!;
    const targetNode = nodes.find(n => n.id === edge.target);
    
    if (targetNode && !processedNodes.has(targetNode.id)) {
      const sourceVar = currentVarMap.get(edge.source) || 'x';
      const layerName = targetNode.data.layerName as string;
      const outVar = `${layerName}_out`;
      
      forwardLines.push(`        ${outVar} = self.${layerName}(${sourceVar})`);
      currentVarMap.set(targetNode.id, outVar);
      currentVar = outVar;
      processedNodes.add(targetNode.id);
      
      // Add next edges to queue
      forwardQueue.push(...edges.filter(e => e.source === targetNode.id));
    }
  }

  return `import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
${initLines.length > 0 ? initLines.join('\n') : '        pass'}

    def forward(self, x):
${forwardLines.length > 0 ? forwardLines.join('\n') : '        return x'}
${forwardLines.length > 0 ? `        return ${currentVar}` : ''}
`;
}
