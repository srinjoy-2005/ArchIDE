# ArchIDE: PyTorch Block Implementation Status

> [!NOTE]
> **Status: 🟢 Up-to-date**
> This document accurately tracks the implementation status of core blocks in the backend registry.

## ✅ Implemented Blocks

### Core Layers
- `input`, `output`
- `linear` (nn.Linear)
- `conv2d` (nn.Conv2d)

### Activations
- `relu` (nn.ReLU)
- `softmax` (nn.Softmax)
- `sigmoid` (nn.Sigmoid)
- `tanh` (nn.Tanh)

### Pooling
- `maxpool2d` (nn.MaxPool2d)
- `avgpool2d` (nn.AvgPool2d)
- `adaptiveavgpool2d` (nn.AdaptiveAvgPool2d)

### Normalization
- `batchnorm2d` (nn.BatchNorm2d)
- `layernorm` (nn.LayerNorm)
- `dropout` (nn.Dropout)

### Shape Operations
- `flatten` (torch.flatten)
- `reshape` (tensor.reshape)
- `unsqueeze` (torch.unsqueeze)
- `split` (torch.chunk)
- `cat` (torch.cat)

### Tensor Operations (Math)
- `add`, `sub`, `mul`, `div`, `pow` (element-wise ops)
- `matmul` (torch.matmul)
- `sin`, `cos` (torch.sin, torch.cos)
- `arange` (torch.arange)

---

## 🔲 Pending Implementation (For Future Phases)

### Priority 2: Remaining Activations
- `gelu` (nn.GELU)
- `leakyrelu` (nn.LeakyReLU)
- `elu` (nn.ELU)
- `silu` (nn.SiLU)

### Priority 3: Remaining Normalization & Regularization
- `batchnorm1d` (nn.BatchNorm1d)
- `instancenorm2d` (nn.InstanceNorm2d)
- `groupnorm` (nn.GroupNorm)
- `dropout2d` (nn.Dropout2d)

### Priority 4: Remaining Shape Manipulation
- `transpose` (torch.transpose)
- `permute` (torch.permute)
- `squeeze` (torch.squeeze)
- `stack` (torch.stack)
- `view` (tensor.view)
- `expand` (tensor.expand)
- `repeat` (tensor.repeat)

### Priority 5: Reductions & Math
- `mean` (torch.mean)
- `sum` (torch.sum)
- `max` (torch.max)
- `min` (torch.min)
- `abs` (torch.abs)
- `exp` (torch.exp)
- `log` (torch.log)
- `sqrt` (torch.sqrt)
- `clamp` (torch.clamp)
- `norm` (torch.norm)

### Priority 6: Transformer / Advanced
- `multiheadattention` (nn.MultiheadAttention)
- `embedding` (nn.Embedding)
- `lstm` (nn.LSTM)
- `gru` (nn.GRU)
