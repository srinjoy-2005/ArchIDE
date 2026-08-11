# ArchiDE — Block Registry Specification

This document provides the formal specification for all blocks available in ArchiDE. The Block Registry defines every draggable node's identity, parameters, input/output ports, shape propagation rules, and PyTorch code generation templates.

> [!NOTE]
> **Status: ✅ Implemented**
> The block registry specifications detailed here reflect the active backend implementation in `backend/blocks/`.

---

## 1. Block Schema & Architecture Specification

The Block Registry defines every draggable node's identity, parameters, and input/output ports. 
In the current architecture, the registry is defined natively in Python.

### Backend Implementation
- **Data Models**: Core structural definitions are in `backend/models.py` using Pydantic (e.g., `BlockDef`, `PortDef`, `ParamDef`).
- **OOP Architecture**: Blocks are implemented as Python classes inheriting from `BaseBlock`. They are categorized into modules within the `backend/blocks/` directory (e.g., `core.py`, `tensor_ops.py`). Each block class implements `infer_shapes`, `emit_init`, and `emit_forward` methods.
- **Registry**: `backend/registry.py` imports all block classes and exposes them via `get_all_block_defs()` and `get_block_by_id()`.

### Frontend Integration
The frontend (Next.js) dynamically fetches the block palette from the FastAPI backend.
1. `src/app/page.tsx` makes a `GET` request to `/api/blocks`.
2. The UI groups the blocks by their `category` and populates the sidebar.
3. Upon clicking "Export PyTorch", the frontend packages the nodes and edges into a `CompileRequest` payload and sends it via `POST` to `/api/compile`.

### Pydantic Schema Reference (`backend/models.py`)
```python
class PortDef(BaseModel):
    id: str
    name: str # User-facing name
    type: str = "tensor"
    is_list: bool = False  
    
class ParamDef(BaseModel):
    name: str
    type: str
    default: Any

class BlockDef(BaseModel):
    id: str
    name: str
    category: str
    color: str
    is_functional: bool
    inputs: List[PortDef]
    outputs: List[PortDef]
    params: List[ParamDef]
```
---

## 2. Block Catalog & Categories

### Category A: Core Layers (`Layers`)

#### 1. Linear (`nn.Linear`)
- **Params**: `in_features` (int, default: 128), `out_features` (int, default: 64), `bias` (bool, default: true)
- **Inputs**: `x` `[B, ..., in_features]`
- **Outputs**: `out` `[B, ..., out_features]`
- **Shape Propagation**: `out_shape = [...input_shape.slice(0, -1), out_features]`
- **PyTorch Init**: `nn.Linear({in_features}, {out_features}, bias={bias})`
- **PyTorch Forward**: `{out} = self.{attr}({x})`

#### 2. Conv2D (`nn.Conv2d`)
- **Params**: `in_channels` (int, default: 3), `out_channels` (int, default: 16), `kernel_size` (int, default: 3), `stride` (int, default: 1), `padding` (int, default: 1)
- **Inputs**: `x` `[B, C_in, H, W]`
- **Outputs**: `out` `[B, C_out, H_out, W_out]`
- **Shape Propagation**:
  - $H_{out} = \lfloor \frac{H + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1 \rfloor$
  - $W_{out} = \lfloor \frac{W + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1 \rfloor$
- **PyTorch Init**: `nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})`

#### 3. ConvTranspose2D (`nn.ConvTranspose2d`)
- **Params**: `in_channels` (int), `out_channels` (int), `kernel_size` (int, default: 3), `stride` (int, default: 2), `padding` (int, default: 1)
- **Inputs**: `x` `[B, C_in, H, W]`
- **Outputs**: `out` `[B, C_out, H_out, W_out]`
- **PyTorch Init**: `nn.ConvTranspose2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})`

#### 4. Flatten (`nn.Flatten`)
- **Params**: `start_dim` (int, default: 1), `end_dim` (int, default: -1)
- **Inputs**: `x` `[B, C, H, W]`
- **Outputs**: `out` `[B, C * H * W]`
- **Shape Propagation**: Preserves dimensions before `start_dim`, flattens target dimensions into product.
- **PyTorch Init**: `nn.Flatten(start_dim={start_dim}, end_dim={end_dim})`

---

### Category B: Activations (`Activations`)

Activations are stateless functional operations or lightweight modules.

| Block Name | PyTorch Class / Function | Parameters | Forward Snippet |
|---|---|---|---|
| **ReLU** | `nn.ReLU` | `inplace=False` | `self.relu(x)` |
| **GELU** | `nn.GELU` | - | `self.gelu(x)` |
| **Sigmoid** | `nn.Sigmoid` | - | `self.sigmoid(x)` |
| **Tanh** | `nn.Tanh` | - | `self.tanh(x)` |
| **LeakyReLU** | `nn.LeakyReLU` | `negative_slope=0.01` | `self.leaky_relu(x)` |
| **Softmax** | `nn.Softmax` | `dim=1` | `self.softmax(x)` |

---

### Category C: Normalization & Regularization (`Normalization`)

#### 1. BatchNorm2D (`nn.BatchNorm2d`)
- **Params**: `num_features` (int, default: 16), `eps` (float, default: 1e-5)
- **PyTorch Init**: `nn.BatchNorm2d({num_features}, eps={eps})`

#### 2. LayerNorm (`nn.LayerNorm`)
- **Params**: `normalized_shape` (int, default: 64), `eps` (float, default: 1e-5)
- **PyTorch Init**: `nn.LayerNorm({normalized_shape}, eps={eps})`

#### 3. Dropout (`nn.Dropout`)
- **Params**: `p` (float, default: 0.5)
- **PyTorch Init**: `nn.Dropout(p={p})`

---

### Category D: Pooling (`Pooling`)

#### 1. MaxPool2D (`nn.MaxPool2d`)
- **Params**: `kernel_size` (int, default: 2), `stride` (int, default: 2), `padding` (int, default: 0)
- **Shape Propagation**: Halves spatial dimensions when `kernel_size=2, stride=2`.

#### 2. AdaptiveAvgPool2D (`nn.AdaptiveAvgPool2d`)
- **Params**: `output_size` (tuple/int, default: "(1, 1)")
- **Shape Propagation**: Output spatial dimensions fixed to `output_size`.

---

### Category E: Multi-Port Tensor Operations (`TensorOps`)

These multi-input operations enable complex DAG topologies (skip connections, ResNets, Transformers):

#### 1. Add / Residual Connection (`torch.add`)
- **Inputs**: `a` (tensor), `b` (tensor)
- **Outputs**: `out` (tensor)
- **Shape Propagation**: Asserts `shape(a) == shape(b)`
- **PyTorch Forward**: `{out} = {a} + {b}`

#### 2. Concatenate (`torch.cat`)
- **Inputs**: `a` (tensor), `b` (tensor)
- **Params**: `dim` (int, default: 1)
- **Outputs**: `out` (tensor)
- **Shape Propagation**: Concatenates along specified `dim`.
- **PyTorch Forward**: `{out} = torch.cat([{a}, {b}], dim={dim})`

#### 3. MultiHeadAttention (`nn.MultiheadAttention`)
- **Inputs**: `query`, `key`, `value`
- **Params**: `embed_dim` (int, default: 256), `num_heads` (int, default: 8), `dropout` (float, default: 0.1)
- **Outputs**: `out`
- **PyTorch Init**: `nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}, batch_first=True)`
- **PyTorch Forward**: `{out}, _ = self.{attr}({query}, {key}, {value})`

---

### Category F: Graph I/O (`I/O`)

#### 1. Input Node
- **Params**: `shape` (string, default: "(1, 3, 224, 224)")
- **Outputs**: `out`
- **Description**: Defines entry tensor into the model `forward(self, x)`.

#### 2. Output Node
- **Inputs**: `in`
- **Description**: Defines exit tensor returned by `return x`.
