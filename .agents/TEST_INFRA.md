# E2E Test Infra: ArchIDE AST Decompiler & Roundtrip Hardening

## Test Philosophy
- Opaque-box, requirement-driven. Derived from user requirements in `ORIGINAL_REQUEST.md`.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise + Real-World Workload Testing.
- Numerical Equivalence: `torch.allclose(out_orig, out_gen, atol=1e-4, rtol=1e-4)` on dummy input tensors.

## Feature Inventory & Test Matrix
| # | Feature | Requirement | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Real-World) |
|---|---------|-------------|:----------------:|:-----------------:|:----------------------:|:-------------------:|
| 1 | `GELUBlock` | R1 | 5 | 5 | ✓ | ✓ (ConvNeXt, ViT) |
| 2 | `SiLUBlock` | R1 | 5 | 5 | ✓ | ✓ |
| 3 | `Conv1DBlock` | R1 | 5 | 5 | ✓ | ✓ |
| 4 | `EmbeddingBlock` | R1 | 5 | 5 | ✓ | ✓ |
| 5 | Scalar Binary Ops (`Add`, `Sub`, `Mul`, `Div`) | R1 | 5 | 5 | ✓ | ✓ (Attention, Scaling) |
| 6 | Multi-Class File Extraction | R2 | 5 | 5 | ✓ | ✓ (ResNet + Block) |
| 7 | Deep Attribute Chains | R2 | 5 | 5 | ✓ | ✓ (Backbone + Layer) |
| 8 | Recursive Module Imports | R2 | 5 | 5 | ✓ | ✓ |
| 9 | `nn.ModuleList` Loop Unrolling | R3 | 5 | 5 | ✓ | ✓ (ViT Blocks) |
| 10 | Structural Conditionals (`ast.If`) | R3 | 5 | 5 | ✓ | ✓ (Downsample / Residual) |
| 11 | Dynamic Shape Extraction | R3 | 5 | 5 | ✓ | ✓ (MHA B, N, C = x.shape) |
| 12 | Roundtrip Numerical Equivalence | R4 | 5 | 5 | ✓ | ✓ (All Canonical Models) |

## Test Architecture
- Test runner: `pytest backend/tests/test_real_world_models.py`
- Invocations:
  - `pytest backend/tests/test_r1_blocks_and_scalars.py`
  - `pytest backend/tests/test_real_world_models.py`
  - `pytest backend/tests/` (full suite)
  - `npx tsc --noEmit` (frontend contract check)
- Pass/fail semantics: Exit code 0, 100% tests passing, all `assert torch.allclose` passing with `atol=1e-4`.

## Real-World Canonical Scenarios (Tier 4)
| # | Scenario | Source Repo / Model | Key Structural & Numerical Aspects |
|---|----------|---------------------|-----------------------------------|
| 1 | ResNet BasicBlock with downsample | `torchvision.models.resnet` | `ast.If` structural check, residual add, intermediate "out" var |
| 2 | ResNet BasicBlock without downsample | `torchvision.models.resnet` | `ast.If` False branch, direct residual identity add |
| 3 | ConvNeXt Block | `torchvision.models.convnext` | 7x7 Depthwise Conv2d, LayerNorm, GELU, 1x1 Linear/Conv, residual |
| 4 | Transformer MLP | `timm` / `transformers` | Linear -> GELU -> Linear, init param binding, scalar ops |
| 5 | Multi-Head Attention (MHA) | `timm` / HuggingFace | `B, N, C = x.shape` tuple unpacking, `ShapeExtractorBlock`, scaled dot-product |
| 6 | ViT Block with ModuleList | `timm` | `nn.ModuleList` unrolling, `for blk in self.blocks: x = blk(x)` dataflow chaining |
| 7 | UNet DoubleConv with Skip Cat | Canonical UNet | Conv2d -> BatchNorm2d -> ReLU (x2), skip concatenation `torch.cat` |
| 8 | Multi-Class File Roundtrip | Canonical multi-class | Two or more `nn.Module` classes in single file, cross-module instantiations |

## Coverage Thresholds
- Tier 1: ≥5 per feature
- Tier 2: ≥5 per feature (where boundaries exist)
- Tier 3: pairwise coverage of major feature interactions
- Tier 4: ≥8 realistic canonical models
