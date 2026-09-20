# Test Ready Report: ArchIDE E2E Canonical Benchmark Test Suite

## Test Runner Invocations
- **Benchmark Suite**: `pytest backend/tests/test_real_world_models.py -v`
- **Full Backend Suite**: `pytest backend/tests/`
- **Frontend Typecheck**: `npx tsc --noEmit`

---

## Coverage Summary Across Tiers 1–4

| Tier | Category | Test Count | Passing | Pending (XFail) | Key Focus Areas |
|---|---|:---:|:---:|:---:|---|
| **Tier 1** | Feature Unit Tests | 5 | 5 | 0 | Robust numerical equivalence helper, weight & buffer sync, perturbation detection, eval determinism, multi-input handling |
| **Tier 2** | Boundary & Edge Conditions | 5 | 5 | 0 | Batch size 1, odd batch size (7), odd spatial shapes (15x15), strict tolerance (`atol=1e-4`, `rtol=1e-4`), scalar ops preservation |
| **Tier 3** | Cross-Feature Interactions | 5 | 5 | 0 | Conv2d+BatchNorm2d+ReLU residual, Linear+GELU+scalar scaling, multi-tensor cat+conv, chained scalar ops, LayerNorm+Linear residual |
| **Tier 4** | Canonical Real-World Benchmarks | 8 | 2 | 6 | ResNet BasicBlock (with & without downsample), ConvNeXt Block, Transformer MLP, Multi-Head Attention, ViT Block ModuleList, UNet DoubleConv, Multi-Class File |
| **Total** | **All Tiers** | **23** | **17** | **6** | **100% test suite pass rate (exit code 0 across 85 collected backend tests)** |

---

## Feature Checklist & Milestone Alignment

| # | Canonical Architecture / Feature | Origin Repo | Status | Milestone | Key Structural & Numerical Aspects |
|---|---|---|:---:|:---:|---|
| 1 | Transformer MLP | `timm` / `transformers` | **PASS** | M1 (Completed) | `Linear -> GELU -> Linear`, scalar ops preservation (`h * 1.0`), init param bindings (`@var:in_features`). Exact numerical match (`diff = 0.0`). |
| 2 | UNet DoubleConv with Skip Cat | Canonical UNet | **PASS** | M1 (Completed) | Multi-input forward `forward(self, x, skip)`, `torch.cat([skip, x], dim=1)` via `CatBlock`, Double `Conv2d -> BatchNorm2d -> ReLU`. Exact numerical match (`diff = 0.0`). |
| 3 | ResNet BasicBlock with Downsample | `torchvision.models.resnet` | **XFAIL** | M3 | `ast.If self.downsample is not None:` True branch, `nn.Sequential` downsample, intermediate `"out"` variable re-assignment, residual addition. |
| 4 | ResNet BasicBlock without Downsample | `torchvision.models.resnet` | **XFAIL** | M3 | `self.downsample = None`, `ast.If self.downsample is not None:` False branch, direct identity pass-through, intermediate `"out"` variable re-assignment. |
| 5 | ConvNeXt Block | `torchvision.models.convnext` | **XFAIL** | M3 | 7x7 Depthwise `Conv2d (groups=dim)`, `LayerNorm`/`BatchNorm2d`, `GELU`, 1x1 `Conv2d`/`Linear` expansion, residual addition, intermediate `"out"` variable. |
| 6 | Multi-Head Attention (MHA) | `timm` / HuggingFace | **XFAIL** | M3 | Dynamic shape extraction `B, N, C = x.shape` tuple unpacking to `ShapeExtractorBlock`, dynamic `reshape(B, N, ...)`, scaled dot-product attention with `import math`. |
| 7 | ViT Block with ModuleList | `timm` | **XFAIL** | M3 | `nn.ModuleList([Block(...) for _ in range(3)])` list comprehension, `for blk in self.blocks: x = blk(x)` loop unrolling, iteration tensor state chaining. |
| 8 | Multi-Class File Roundtrip | Multi-class pattern | **XFAIL** | M2 | Multiple `nn.Module` classes in a single file (`FeedForward` + `TransformerEncoderBlock`), cross-module dependency resolution, interconnected IRs. |

---

## Progressive Testability & Self-Healing Architecture
The test suite implements dynamic capability probes (`_check_feature`):
- Rather than static test skips, probes directly evaluate backend AST capabilities.
- As workers complete **M2 (Multi-Class Decompilation)** and **M3 (AST Control Flow & Dynamic Shapes)**, the tests will automatically transition from `XFAIL` to live active tests.
- In **Milestone 4 (Integration & Canonical Benchmark)**, 100% of Tier 4 benchmark tests are expected to achieve live `PASS` with `torch.allclose(atol=1e-4, rtol=1e-4)`.

---

## Escalated Defects & Architectural Discoveries
1. **Intermediate Node ID Collision (`_next_node_id`)**:
   - Intermediate variables named `out` in real-world models (e.g. `out = self.conv(x); out = out + residual; return out`) cause `_next_node_id("out")` to return `"out"`. Terminal `_parse_return` overwrites the node with output block and creates `out.out -> out.in`, causing Kahn's topological sort to fail with `ValueError: Cycle detected in graph! Cannot compile.` (Assigned to M3).
2. **Kahn's Sort Queue UUID Ordering in Multi-Input Models**:
   - `AgentGraphCompiler` assigns random UUIDs (`generate_id("input")`). `topological_sort` sorts the zero-in-degree queue lexicographically with `queue.sort()`. For multi-input graphs (e.g. `x, skip` or `x1, x2`), this randomly flips the forward argument signature (`x_input, x_input_2` vs `x_input_2, x_input`).
3. **LayerNorm Positional Parameter Mapping in Decompiler**:
   - `nn.LayerNorm(512)` positional argument `512` is not mapped because `LayerNorm` is absent from `BLOCK_PARAM_MAP` in `python_decompiler.py`, causing `normalized_shape` to fall back to default `512`. Passing variables or non-512 values defaults to 512.
