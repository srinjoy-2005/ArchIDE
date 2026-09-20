# Dispatch Log

## 2026-09-20T07:41:50Z

From: parent (efdde2d6-2f53-4768-80fc-85750ed12f6b)

You are the Project Orchestrator for ArchIDE.

Your working directory is: d:\ML\ArchIDE\.agents\orchestrator_1
Project root: d:\ML\ArchIDE
User Request Reference: d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md

## Mission & Requirements
Extend and harden the ArchIDE PyTorch AST decompiler and roundtrip pipeline to support multilevel files, multi-class modules, deep object attribute chains, and missing layer blocks, validated through an automated benchmark across canonical architectures from popular PyTorch repositories.

Integrity mode: development

### R1. Core Block Extensions & Scalar Binary Operations
Implement missing activation and layer blocks (`GELU`, `SiLU`, `Conv1d`, `Embedding`) in the ArchIDE backend, and enable scalar operands for binary tensor operations (`Add`, `Sub`, `Mul`, `Div`) so expressions like `x + 1` or `x / math.sqrt(d)` compile and execute without dropping scalar values.

### R2. Multilevel Files & Multi-Class Decompilation
Extend `python_decompiler.py` to extract all `nn.Module` classes within a file (rather than only the last class), resolve cross-file and local module imports recursively, and map deep attribute chains (`self.backbone.layer1(x)`) into valid submodule dataflows.

### R3. AST Control Flow & Dynamic Shape Extraction
Support `ast.For` loop unrolling over `nn.ModuleList` by tracing tensor mutations across iterations, handle structural conditional checks (`if self.downsample is not None:`), and map `x.shape` / `x.size()` tuple unpackings to `ShapeExtractorBlock`.

### R4. Real-World Multi-Repo Benchmark & Roundtrip Equivalence
Create a comprehensive automated benchmark test suite covering canonical models (ResNet/ConvNeXt from `torchvision`, ViT/MHA and Transformer MLP from `timm`/HuggingFace, UNet, and multi-class files) verifying that decompiled and recompiled models produce mathematically identical outputs (`torch.allclose`) on dummy inputs.

## Acceptance Criteria
- [ ] `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, and `EmbeddingBlock` are registered in `_BLOCK_INSTANCES` and pass shape inference and code generation.
- [ ] Binary ops (`Add`, `Sub`, `Mul`, `Div`) preserve scalar parameters (`scalar_a`, `scalar_b`) during forward code generation.
- [ ] A Python file containing multiple `nn.Module` classes decompiles into distinct, interconnected IRs (`.ir.json`).
- [ ] Nested attribute calls (`self.submodule.layer(x)`) resolve to valid dataflow nodes and edges.
- [ ] Unrolled `nn.ModuleList` loops correctly propagate tensor dataflow through each step.
- [ ] All existing tests (`pytest backend/tests/`) and new benchmark tests (`pytest backend/tests/test_real_world_models.py`) pass.
- [ ] `npx tsc --noEmit` passes with 0 errors.
- [ ] PyTorch numerical forward-pass equivalence is verified on dummy inputs with `atol=1e-4`.

## Critical Project Guardrails & Rules
1. Frontend State: `DnDCanvas.tsx` is strictly UNCONTROLLED. State is synced to `vfsStore` via debounced effect. To mutate nodes programmatically, MUST use `useReactFlow().setNodes()`.
2. Backend Authority: The Python backend is the source of truth for block schemas. Frontend fetches from `/api/blocks`.
3. Data Pipeline: React Flow JSON -> `/api/compile` -> Kahn's Sort -> AST Generation -> Disk -> SSE -> VFS Store.
4. DO NOT manually document new blocks or maintain block lists (`blocks_status.md` is deprecated).
5. DO NOT update documentation for bug fixes or local refactors.
6. Verification commands: Run `npx tsc --noEmit` and `pytest backend/tests/`.
7. Maintain `BRIEFING.md` and `progress.md` in your working directory (`d:\ML\ArchIDE\.agents\orchestrator_1`).
8. When complete, notify the Sentinel with a full summary of results.
