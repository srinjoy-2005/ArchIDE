# Compiler Engine Design

This document outlines how we built the `/api/compile` endpoint in Python. The goal of this compiler is to take the raw JSON graph from the frontend and translate it into a valid, executable PyTorch `nn.Module`.

> [!NOTE]
> **Status: ✅ Implemented**
> The compiler engine, Kahn's topological sort, and the string-based AST generation described here are fully operational in the backend.

## 1. The Input Payload (CompileRequest & CheckRequest)
The frontend will send a JSON payload containing `nodes` and `edges`. We define this strictly using Pydantic:

```python
class Edge(BaseModel):
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str

class NodeData(BaseModel):
    block_id: str = ""
    label: str
    is_functional: bool = False
    paramValues: dict = {}
    varName: str = ""

class Node(BaseModel):
    id: str
    data: NodeData

class CompileRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
```

## 2. Topological Sort (The Execution Order)
You cannot execute a `Linear` layer before the `Input` layer. We must order the nodes mathematically.

**The Algorithm (Kahn's Algorithm):**
1. Map every edge to an adjacency list: `adjacency[source_node].append(target_node)`
2. Count the `in_degree` (number of incoming edges) for every node.
3. Find all nodes with `in_degree == 0` (these are our `Input` blocks) and put them in a Queue.
4. Pop a node from the Queue, add it to our `sorted_nodes` list, and decrement the `in_degree` of all its neighbors.
5. If a neighbor's `in_degree` becomes `0`, push it to the Queue.
6. Continue until the Queue is empty.

## 3. Code Generation (AST / String Building)
Once we have `sorted_nodes`, we iterate through them exactly twice:
- **Pass 1 (`__init__`)**: We find all nodes where `is_functional == False`. We look up their block class via `get_block_by_id(node.data.block_id)` and delegate generation by calling `block.emit_init()`.
- **Pass 2 (`forward`)**: We iterate through *all* nodes to generate the execution path. We look up their incoming edge variables, resolve their output variables, and call `block.emit_forward()`.

### Variable Tracking (The tricky part)
To write the `forward` pass, we need to know the names of the variables passed between blocks. We maintain a dictionary: `port_to_var: dict[str, str]`.
- Key: The `NodeID + PortID` (e.g., `node_1_out`)
- Value: The generated Python variable name (e.g., `x_1` or a user-provided `varName`)

**OOP Generation Loop:**
Instead of a giant switch/case statement, the compiler relies on polymorphic blocks defined in `backend/blocks/`:
1. It resolves incoming ports by finding all edges that point to the current node.
2. It fetches the variables from `port_to_var` for those incoming edges.
3. It passes `input_vars` and `output_vars` to `block.emit_forward(node_id, input_vars, output_vars, params)`.
4. The block handles generating its own specific PyTorch string (like `return f"{out_var} = self.layer_{node_id}({in_var})"`) and the compiler simply appends it to the code body.
