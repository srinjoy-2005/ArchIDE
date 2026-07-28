# ArchIDE: Compiler Design & Extensibility Plan

This document outlines the architectural plan for decoupling block functionalities and implementing static shape validation in the ArchIDE Python backend. It addresses the requirements from the roadmap to use established Compiler Design (CD) principles to make the system scalable for custom blocks.

---

## 1. Decoupling Block Functionalities

Currently, `backend/registry.py` uses a purely data-driven approach (`BlockDef` Pydantic models) to define blocks. However, to support shape checking and code generation scalably, a block's definition must couple its **data schema** (parameters/ports) with its **behavior** (shape inference and code generation).

### Proposed Architecture: Object-Oriented Intermediate Representation (IR) Node Classes

Instead of a single monolithic compiler file mapping IDs to strings, we will adopt a standard Compiler Design pattern using polymorphic AST (Abstract Syntax Tree) nodes.

Each block will be represented by a class inheriting from a core `BaseBlock` interface.

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from .models import BlockDef

class BaseBlock(ABC):
    # Data Definitions (Schema)
    @property
    @abstractmethod
    def definition(self) -> BlockDef:
        """Returns the Pydantic schema for the frontend registry."""
        pass

    # Shape Engine Logic
    @abstractmethod
    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        """Calculates output shapes given input port shapes and block parameters."""
        pass

    # Code Generator Logic
    @abstractmethod
    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        """Generates the nn.Module instantiation code (if stateful)."""
        pass

    @abstractmethod
    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str]) -> str:
        """Generates the functional execution code."""
        pass
```

#### Where is code written?
- Core blocks (e.g., `Conv2D`, `Linear`, `ReLU`) will be defined in a `backend/blocks/` directory, each in their respective module (e.g., `backend/blocks/core.py`, `backend/blocks/activations.py`).
- Adding a custom block simply requires defining a new class that implements `BaseBlock` and registering it in an `__init__.py` registry loader. 
- The frontend will automatically fetch the schema via the `/api/blocks` endpoint by iterating over the registered classes' `definition` properties.

---

## 2. Static Check for Shape Compatibility

To implement robust static shape checking when blocks are connected in the UI, the compiler will introduce a **Semantic Analysis Pass** (Shape Inference) immediately after the topological sort.

### The Shape Propagation Pipeline in `compiler.py`

1. **Topological Sort**: Kahn's algorithm resolves dependencies and detects cycles (already implemented).
2. **Shape Inference Pass**:
    - The compiler initializes a `tensor_shapes` dictionary mapping `edge_id` (or `node_port_id`) to a concrete tensor shape tuple (e.g., `(B, C, H, W)`).
    - Starting with the `Input` node, which dictates the initial shape.
    - Traverse the sorted nodes. For each node:
        - Gather the shapes of all incoming edges.
        - Look up the node's class instance (e.g., `Conv2DBlock`).
        - Invoke `Conv2DBlock.infer_shapes(incoming_shapes, node.params)`.
        - If the shape inference logic detects a mismatch (e.g., trying to pool a 1D tensor with a 2D pooler, or mismatched matrix multiplication dimensions), it raises a `ShapeMismatchError`.
        - Assign the returned output shapes to the node's outgoing edges.
3. **Code Generation Pass**: If shape inference succeeds, proceed to emit PyTorch strings.

### Example: Conv2D Shape Inference Implementation

```python
import math

class Conv2DBlock(BaseBlock):
    ...
    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if in_shape is None or len(in_shape) != 4:
            raise ValueError("Conv2D requires a 4D input tensor (B, C, H, W).")
            
        B, C, H, W = in_shape
        if C != params["in_channels"]:
            raise ValueError(f"Expected in_channels={params['in_channels']}, but got tensor with {C} channels.")
            
        kernel = params.get("kernel_size", 3)
        padding = params.get("padding", 0)
        stride = params.get("stride", 1)
        
        out_h = math.floor((H + 2*padding - kernel) / stride + 1)
        out_w = math.floor((W + 2*padding - kernel) / stride + 1)
        
        return {"out": (B, params["out_channels"], out_h, out_w)}
```

---

## 3. Options for Custom Block Parameters

When a user defines a "Custom Block" through the UI later on, they cannot easily write Python class methods on the fly. 

To make the compiler scalable to **UI-defined custom blocks**, we will need a dynamic shape inference engine for those specific blocks:
- **DSL (Domain Specific Language)**: For simple custom blocks, we can allow users to provide mathematical formulas strings in the UI (e.g. `out_h = (in_h - kernel) / 1 + 1`).
- **Any-Shape Propagation**: For completely black-box custom blocks, the shape engine will fall back to `"ANY"` and disable strict checking for downstream nodes, delegating the crash to PyTorch runtime.

### Summary
This object-oriented approach centralizes a block's data, shape rules, and code generation into a single file, preventing the `compiler.py` and `registry.py` from becoming massively bloated `switch/case` statements.
