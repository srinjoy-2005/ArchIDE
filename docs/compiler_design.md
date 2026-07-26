# Compiler Engine Design

This document outlines how we will build the `/api/compile` endpoint in Python. The goal of this compiler is to take the raw JSON graph from the frontend and translate it into a valid, executable PyTorch `nn.Module`.

## 1. The Input Payload (CompileRequest)
The frontend will send a JSON payload containing `nodes` and `edges`. We will define this strictly using Pydantic:

```python
class Edge(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str

class NodeData(BaseModel):
    label: str
    is_functional: bool = False
    paramValues: dict = {}

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
- **Pass 1 (`__init__`)**: We find all nodes where `is_functional == False`. We generate their instantiation strings (e.g., `self.layer_{id} = nn.Linear(in_features=128, out_features=64)`).
- **Pass 2 (`forward`)**: We iterate through *all* nodes to generate the execution path. 

### Variable Tracking (The tricky part)
To write the `forward` pass, we need to know the names of the variables passed between blocks. We will maintain a dictionary: `port_to_var: dict[str, str]`.
- Key: The `NodeID + PortID` (e.g., `node_1_out`)
- Value: The generated Python variable name (e.g., `x_1`)

**Example Generation Loop:**
- **If Node is `Input`**: 
  - Generates: `x_{id} = input_tensor`
  - Stores `x_{id}` in the dictionary so the next block can use it.
- **If Node is `Add`**: 
  - Looks up the variables connected to `in_0` and `in_1` using the `edges` data.
  - Generates: `x_{id} = var_a + var_b`
- **If Node is `Output`**:
  - Looks up the variable connected to `in`.
  - Generates: `return var_in`

## 4. Implementation Steps
1. Define the Pydantic request models in `main.py`.
2. Write a helper function `topological_sort(nodes, edges)` that returns a list of sorted node IDs.
3. Write a helper function `generate_pytorch_code(sorted_nodes, nodes_dict, edges)` that implements the two-pass string building.
4. Expose the `@app.post("/api/compile")` endpoint.
