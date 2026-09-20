# Original User Request

## 2026-09-20T07:41:12Z

Extend and harden the ArchIDE PyTorch AST decompiler and roundtrip pipeline to support multilevel files, multi-class modules, deep object attribute chains, and missing layer blocks, validated through an automated benchmark across canonical architectures from popular PyTorch repositories.

Working directory: d:\ML\ArchIDE
Integrity mode: development

## Requirements

### R1. Core Block Extensions & Scalar Binary Operations
Implement missing activation and layer blocks (`GELU`, `SiLU`, `Conv1d`, `Embedding`) in the ArchIDE backend, and enable scalar operands for binary tensor operations (`Add`, `Sub`, `Mul`, `Div`) so expressions like `x + 1` or `x / math.sqrt(d)` compile and execute without dropping scalar values.

### R2. Multilevel Files & Multi-Class Decompilation
Extend `python_decompiler.py` to extract all `nn.Module` classes within a file (rather than only the last class), resolve cross-file and local module imports recursively, and map deep attribute chains (`self.backbone.layer1(x)`) into valid submodule dataflows.

### R3. AST Control Flow & Dynamic Shape Extraction
Support `ast.For` loop unrolling over `nn.ModuleList` by tracing tensor mutations across iterations, handle structural conditional checks (`if self.downsample is not None:`), and map `x.shape` / `x.size()` tuple unpackings to `ShapeExtractorBlock`.

### R4. Real-World Multi-Repo Benchmark & Roundtrip Equivalence
Create a comprehensive automated benchmark test suite covering canonical models (ResNet/ConvNeXt from `torchvision`, ViT/MHA and Transformer MLP from `timm`/HuggingFace, UNet, and multi-class files) verifying that decompiled and recompiled models produce mathematically identical outputs (`torch.allclose`) on dummy inputs.

## Acceptance Criteria

### Engine & Registry Completeness
- [ ] `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, and `EmbeddingBlock` are registered in `_BLOCK_INSTANCES` and pass shape inference and code generation.
- [ ] Binary ops (`Add`, `Sub`, `Mul`, `Div`) preserve scalar parameters (`scalar_a`, `scalar_b`) during forward code generation.

### Multilevel & Structural Decompilation
- [ ] A Python file containing multiple `nn.Module` classes decompiles into distinct, interconnected IRs (`.ir.json`).
- [ ] Nested attribute calls (`self.submodule.layer(x)`) resolve to valid dataflow nodes and edges.
- [ ] Unrolled `nn.ModuleList` loops correctly propagate tensor dataflow through each step.

### Roundtrip Equivalence & Test Suite
- [ ] All existing tests (`pytest backend/tests/`) and new benchmark tests (`pytest backend/tests/test_real_world_models.py`) pass.
- [ ] `npx tsc --noEmit` passes with 0 errors.
- [ ] PyTorch numerical forward-pass equivalence is verified on dummy inputs with `atol=1e-4`.
