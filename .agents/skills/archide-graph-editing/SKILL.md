---
name: archide-graph-editing
description: Instructs agents on how to scalably create or edit complex architectures using the Agentic Graph IR and compiler.
---

# Scalable Architecture Generation Guidelines

When tasked with creating or modifying an ArchIDE architecture graph (`.arch` file), **DO NOT** attempt to write raw React Flow JSON manually. Raw JSON requires exact UUIDs, (x, y) spatial coordinates, and heavy UI boilerplate which leads to broken graphs.

Instead, you must use the **Agentic Graph IR** and compile it using the backend tooling.

## 1. The Agentic Graph IR
The IR is a simplified JSON file where nodes use semantic aliases (instead of UUIDs), coordinates are omitted, and edges use a simple `source.port -> target.port` string syntax.

Create a `.ir.json` file like this:
```json
{
  "name": "My Architecture",
  "hyperparameters": [
    {"name": "dim", "type": "int", "default": 128}
  ],
  "nodes": {
    "in": {"block": "input", "params": {"shape": "(1, dim, 32, 32)"}},
    "conv1": {"block": "conv2d", "params": {"in_channels": "dim", "out_channels": "dim * 2"}},
    "relu": {"block": "relu"},
    "out": {"block": "output"}
  },
  "edges": [
    "in.out -> conv1.in",
    "conv1.out -> relu.in",
    "relu.out -> out.in"
  ]
}
```

## 2. Dynamic Schema Discovery
Before writing the IR, you need to know what blocks, ports, and parameters are available.
Run the schema dumper tool:
```bash
source backend/.venv/bin/activate && python backend/dump_block_schema.py
cat backend/block_schema.json
```
This will give you the exact `block` IDs, valid `params`, and `inputs`/`outputs` handles you can use.

## 3. Storage Locations
- **IR files**: Stored in `workspace/ir/*.ir.json`
- **Visual graph files**: Stored in `workspace/graphs/*.arch`

## 4. Compiling and Decompiling Graphs

### Compiling IR to Visual Graph (`.arch`)
Once you have written or updated your `workspace/ir/my_graph.ir.json` file:
```bash
./scripts/compile_all_to_arch.sh workspace/ir/my_graph.ir.json
```
To batch compile and validate all IR files:
```bash
./scripts/compile_all_to_arch.sh
```
This will automatically:
1. Generate strict React Flow UUIDs.
2. Hydrate all missing `inputs`, `outputs`, and UI boilerplate from the schema.
3. Calculate beautiful topological `x, y` coordinates using a DAG layout algorithm.
4. Perform shape inference and PyTorch code validation.
5. Output `workspace/graphs/my_graph.arch` (which ArchIDE loads via Live-Sync).

### Decompiling Visual Graph (`.arch`) to IR (`.ir.json`)
To inspect or decompile visual `.arch` files to compact IR:
```bash
./scripts/decompile_all_to_ir.sh workspace/graphs/my_graph.arch
# Or batch decompile all:
./scripts/decompile_all_to_ir.sh
```

**Always use `agent_compiler.py` or the helper scripts. Never edit `.arch` files directly.**

