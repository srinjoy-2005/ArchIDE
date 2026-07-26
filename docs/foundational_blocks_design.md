# Designing Foundational Blocks

As discussed, before we worry about the wiring logic (the compiler), we need to solidify the design of our **Foundational Primitives**. These are the lowest-level math and tensor operations that users will drag-and-drop to build extremely complex modules (like RoPE or Attention).

## 1. Goal
Identify the core set of operations required to build any complex Neural Network architecture visually, define their exact Inputs/Outputs, and specify what properties the user can edit.

## 2. Pydantic Registry Design (Backend Implementation Proposal)

To solve the UI challenges you mentioned (like an `Add` block taking up to 4 tensors, or a `Split` block returning multiple tensors), we need to explicitly define **Inputs** and **Outputs** (Ports) in the `BlockDef`. 

Here is how I suggest you structure the Pydantic models in `backend/main.py`:

```python
from pydantic import BaseModel
from typing import List, Any, Optional

class PortDef(BaseModel):
    id: str             # Internal port ID (e.g., 'in_1')
    name: str           # User-facing name (e.g., 'Tensor A')
    type: str = "tensor" # 'tensor' or 'scalar'
    is_list: bool = False # If True, the frontend lets the user connect multiple edges to this single port (like a variadic Add/Concat)

class ParamDef(BaseModel):
    name: str
    type: str
    default: Any

class BlockDef(BaseModel):
    id: str
    name: str
    category: str
    color: str
    is_functional: bool       # Tells the backend: if True, put in forward() only. If False, instantiate in __init__().
    inputs: List[PortDef]     # Defines the "dots" on the left/top of the block
    outputs: List[PortDef]    # Defines the "dots" on the right/bottom of the block
    params: List[ParamDef]
```

### Example 1: The `Add` Block
Instead of hardcoding 4 inputs, we can use `is_list=True` so the frontend knows it can accept infinite connections, or we can hardcode 4 distinct ports. 

```python
BlockDef(
    id="add",
    name="Add",
    category="Math",
    color="#8b5cf6",
    is_functional=True,  # No trainable weights, just `a + b + c`
    inputs=[
        PortDef(id="in_a", name="A"),
        PortDef(id="in_b", name="B"),
        PortDef(id="in_c", name="C"),
        PortDef(id="in_d", name="D")
    ],
    outputs=[
        PortDef(id="out", name="Out")
    ],
    params=[]
)
```

### Example 2: The `Split` Block (Multiple Outputs)
```python
BlockDef(
    id="split",
    name="Split (Chunk)",
    category="Tensor Ops",
    color="#ef4444",
    is_functional=True,
    inputs=[
        PortDef(id="in", name="Input Tensor")
    ],
    outputs=[
        # If the user sets 'chunks=2' in properties, the frontend can dynamically render these
        PortDef(id="out_1", name="Chunk 1"),
        PortDef(id="out_2", name="Chunk 2")
    ],
    params=[
        ParamDef(name="chunks", type="int", default=2),
        ParamDef(name="dim", type="int", default=-1)
    ]
)
```

## 3. How the Frontend Uses This
If you send me this JSON:
1. I will read `inputs` and generate exact dots on the left side of the visual block.
2. I will read `outputs` and generate exact dots on the right side.
3. When the user draws a line, React Flow saves an edge: `{ source: 'split_node', sourceHandle: 'out_1', target: 'add_node', targetHandle: 'in_a' }`.
4. When you receive that edge in the compiler, you know *exactly* which tensor flows where!
