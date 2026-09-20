# Project: ArchIDE AST Decompiler & Roundtrip Hardening

## Architecture
ArchIDE converts between PyTorch Python code, visual Node/Edge Graph representations, and intermediate AST definitions:
- **Block Registry**: `backend/blocks/` defines block schemas (`BlockDef`), shape inference, code generation (`emit_init`, `emit_forward`).
- **AST Decompiler**: `backend/python_decompiler.py` parses Python source code into AST, traverses statements, extracts layers, variables, and dataflow connections into Agentic IR (`.ir.json`).
- **Compiler Pipeline**: `backend/agent_compiler.py` and `backend/compiler.py` validate IR, load module ports, perform Kahn's topological sort, generate clean modular PyTorch code, and emit to disk/SSE.
- **Verification Engine**: `backend/tests/test_real_world_models.py` validates roundtrip equivalence via `torch.allclose(atol=1e-4)` across canonical vision and transformer architectures.

---

## Feature Inventory
Every surveyed requirement and architectural enhancement is cataloged below with its assigned milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `GELUBlock` | Implement `GELUBlock` with `approximate` param ('none'/'tanh'), shape inference, code emission | M1 | Survey R1 |
| 2 | `SiLUBlock` | Implement `SiLUBlock` with `inplace` param, shape inference, code emission | M1 | Survey R1 |
| 3 | `Conv1DBlock` | Implement `Conv1DBlock` with in/out channels, kernel, stride, pad, dil, groups, bias, 1D shape math | M1 | Survey R1 |
| 4 | `EmbeddingBlock` | Implement `EmbeddingBlock` with num_embeddings, embedding_dim, padding_idx, max_norm, shape inference | M1 | Survey R1 |
| 5 | Block Registry & Schema Dump | Register 4 blocks in `_BLOCK_INSTANCES` and dump updated schema to `block_schema.json` | M1 | Survey R1 |
| 6 | Scalar Binary Ops Preservation | Update `Add`, `Sub`, `Mul`, `Div` in `tensor_ops.py` to preserve `scalar_a` and `scalar_b` during `emit_forward` | M1 | Survey R1 |
| 7 | Compiler Math Import | Add `import math` to compiler module imports header for `math.sqrt(d)` expressions | M1 | Survey R1 |
| 8 | Multi-Class Discovery & DAG Sort | Extract all `ast.ClassDef` in a file, compute constructor dependency DAG, topological sort | M2 | Survey R2 |
| 9 | Multi-Class Interconnected IRs | Generate distinct `.ir.json` per class, linking parent to child via `custom_module` | M2 | Survey R2 |
| 10 | Recursive Import Resolution | Resolve relative/absolute module imports with alias support and visited-set cycle guard | M2 | Survey R2 |
| 11 | Deep Object Attribute Chains | Extract arbitrary self-attribute/subscript chains (`self.backbone.layer1(x)`) into connected dataflow | M2 | Survey R2 |
| 12 | Agent Compiler IR Port Loading | Update `_load_custom_module_ports` in `agent_compiler.py` to support dictionary-based `.ir.json` | M2 | Survey R2 |
| 13 | Intermediate Node ID Collision Fix | Fix `_next_node_id` to never allocate reserved `"out"` or `"in"` to intermediate tensor variables | M3 | Survey R3 |
| 14 | ModuleList For-Loop Unrolling | Unroll `ast.For` over `nn.ModuleList` (supporting `ast.ListComp`), executing body and chaining tensor state | M3 | Survey R3 |
| 15 | Structural Conditionals (`ast.If`) | Statically evaluate `ast.If` (`self.attr is not None`, `self.attr`, etc.) against `layer_instances`/`attributes` | M3 | Survey R3 |
| 16 | Dynamic Shape Extraction | Map `x.shape` / `x.size()` tuple unpackings to `ShapeExtractorBlock`, preserving variable names | M3 | Survey R3 |
| 17 | Reshape/View Multi-Arg & Alpha Fix | Fix `ReshapeBlock` character stripping to preserve variable expressions; pack varargs in `view`/`reshape` | M3 | Survey R3 |
| 18 | Canonical Benchmark Suite | Implement `test_real_world_models.py` testing ResNet, ConvNeXt, ViT MHA/MLP, UNet, and multi-class files | M4 / E2E | Survey R4 |
| 19 | Numerical Equivalence Verification | Verify `torch.allclose(out_orig, out_gen, atol=1e-4)` across canonical architectures | M4 / E2E | Survey R4 |
| 20 | Dual Track Test Infra & Hardening | Formalize opaque-box test runner (Tiers 1-4) and Phase 2 white-box adversarial stress testing (Tier 5) | E2E Track | Survey R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Block Extensions & Scalar Binary Operations | Features 1–7: GELU, SiLU, Conv1d, Embedding blocks, scalar binary ops, compiler math import, schema dump | none | DONE |
| M2 | Multilevel Files & Multi-Class Decompilation | Features 8–12: Multi-class extraction, submodule IR generation, recursive imports, deep attribute chains | none | IN_PROGRESS |
| M3 | AST Control Flow & Dynamic Shape Extraction | Features 13–17: Fix "out" collision, ModuleList loop unrolling, ast.If conditionals, ShapeExtractor mapping, ReshapeBlock fix | M1 | PLANNED |
| E2E | E2E Test Suite Infrastructure & Test Cases | Features 18–20: Test infra runner, Tier 1–4 requirement-driven test cases, TEST_READY.md | none | IN_PROGRESS |
| M4 | Final Milestone: Canonical Benchmark & Roundtrip Equivalence | Phase 1: 100% E2E test pass (Tiers 1–4). Phase 2: Adversarial coverage hardening (Tier 5) | M1, M2, M3, E2E | PLANNED |

---

## Interface Contracts

### M1 ↔ M2 / M3 (Block Registry & Decompiler)
- `get_block_by_id(block_id)` returns valid `BaseBlock` for `"gelu"`, `"silu"`, `"conv1d"`, `"embedding"`. [VERIFIED]
- `AddBlock`, `SubBlock`, `MulBlock`, `DivBlock` accept `params["scalar_a"]` and `params["scalar_b"]` and emit scalar expressions when incoming edges are missing. [VERIFIED]
- `backend/compiler.py` includes `"import math"` in generated headers. [VERIFIED]

### M2 ↔ M3 / M4 (Multi-Class IR & Custom Modules)
- `decompile_source` or `decompile_all_classes` produces:
  - Root model IR: `{ "name": "<model>", "nodes": {...}, "edges": [...] }`
  - Submodule IRs: `{ "<submodule>": { "name": "<submodule>", "nodes": {...}, "edges": [...] } }`
- Calling nested attributes `self.backbone.layer1(x)` produces a connected node in IR with valid `(node_id, "out")` returned to `self.env`.
- `agent_compiler.py:_load_custom_module_ports` loads ports from both `.arch` (list nodes) and `.ir.json` (dict nodes).

### M3 ↔ M4 (AST Control Flow & Dynamic Shapes)
- Intermediate variables named `out` are assigned `out_1`, `out_2`, etc., avoiding collision with terminal `out` node.
- `ShapeExtractorBlock` output ports `dim_0`..`dim_k` map directly to target unpacked variables (`B`, `N`, `C`).
- `ReshapeBlock` preserves string expressions like `(b, c, -1)` without stripping non-digits.

---

## Code Layout
- `backend/blocks/activations.py` — `GELUBlock`, `SiLUBlock` [M1 DONE]
- `backend/blocks/core.py` — `Conv1DBlock`, `EmbeddingBlock` [M1 DONE]
- `backend/blocks/tensor_ops.py` — `AddBlock`, `SubBlock`, `MulBlock`, `DivBlock` [M1 DONE]
- `backend/blocks/__init__.py` — `_BLOCK_INSTANCES` registration [M1 DONE]
- `backend/compiler.py` — PyTorch code generator, imports, and naming [M1 DONE]
- `backend/agent_compiler.py` — Agent graph compiler & port loader [M2]
- `backend/python_decompiler.py` — PyTorch AST decompiler [M2, M3]
- `backend/dump_block_schema.py` & `backend/block_schema.json` — Schema exporter [M1 DONE]
- `backend/tests/` — Existing test suite
- `backend/tests/test_r1_blocks_and_scalars.py` — M1 unit verification [M1 DONE]
- `backend/tests/test_real_world_models.py` — M4 benchmark & roundtrip test suite
