## 2026-09-20T07:50:44Z
You are the E2E Test Writer for ArchIDE.
Your working directory is: d:\ML\ArchIDE\.agents\test_writer_e2e_1
Original user request reference: d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md
Project specification: d:\ML\ArchIDE\.agents\PROJECT.md
Test Infra specification: d:\ML\ArchIDE\.agents\TEST_INFRA.md
Explorer Survey 3 Findings & Design: d:\ML\ArchIDE\.agents\explorer_survey_3\handoff.md

Mission:
Build the E2E Canonical Benchmark Test Suite in `backend/tests/test_real_world_models.py` and publish `d:\ML\ArchIDE\.agents\TEST_READY.md`.

Scope:
1. Write `backend/tests/test_real_world_models.py` covering canonical models from torchvision, timm, transformers:
   - ResNet BasicBlock with downsample (`ast.If`, residual addition, intermediate "out" variable)
   - ResNet BasicBlock without downsample (`ast.If` False branch)
   - ConvNeXt Block (7x7 Depthwise Conv2d, LayerNorm, GELU, 1x1 Linear/Conv, residual)
   - Transformer MLP (Linear -> GELU -> Linear, scalar ops, init param binding)
   - Multi-Head Attention with dynamic shape extraction (`B, N, C = x.shape` tuple unpacking, ShapeExtractorBlock, Q/K/V projections, scaled dot-product)
   - ViT Block with `nn.ModuleList` loop unrolling (`for blk in self.blocks: x = blk(x)`)
   - UNet DoubleConv with skip concatenation (`torch.cat([skip, x], dim=1)` via CatBlock)
   - Multi-class file roundtrip (e.g. `BasicBlock` + `ResNet` or `Mlp` + `TransformerBlock`)
2. Numerical Equivalence Helper:
   - Create a robust roundtrip helper function in `test_real_world_models.py`:
     - Given an original PyTorch module instance and its source code:
     - Decompile source code to IR (`decompile_source` or `decompile_all_classes` from `backend.python_decompiler`).
     - Compile IR to code (`compile_ir` from `backend.agent_compiler` or `generate_pytorch_code` from `backend.compiler`).
     - Execute the generated code in an isolated `exec()` namespace to instantiate the recompiled module.
     - Copy weights (`orig.named_parameters()` -> `gen.named_parameters()`) and buffers (`orig.named_buffers()` -> `gen.named_buffers()`).
     - Switch both to `.eval()` mode (to ensure deterministic inference, no dropout).
     - Run dummy inputs through original and recompiled models.
     - Assert `torch.allclose(y_orig, y_gen, atol=1e-4, rtol=1e-4)`.
3. Publish `d:\ML\ArchIDE\.agents\TEST_READY.md` containing:
   - Test Runner command: `pytest backend/tests/test_real_world_models.py -v`
   - Coverage Summary across Tiers 1-4
   - Feature checklist
4. Report:
   - Write completion report to `d:\ML\ArchIDE\.agents\test_writer_e2e_1\handoff.md`.
   - Send completion message to orchestrator.
