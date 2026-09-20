## 2026-09-20T07:42:45Z
You are Explorer 3 on Step 0: Survey for ArchIDE.
Your working directory is: d:\ML\ArchIDE\.agents\explorer_survey_3
Original user request reference: d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md

Mission:
Investigate the ArchIDE codebase specifically for Requirement R3 (AST Control Flow & Dynamic Shape Extraction) and Requirement R4 (Real-World Multi-Repo Benchmark & Roundtrip Equivalence).

Scope & Focus:
1. Control Flow & Unrolling in Decompiler:
   - Inspect `backend/compiler/python_decompiler.py` for control flow handling (`ast.For`, `ast.If`).
   - How to support `ast.For` loop unrolling over `nn.ModuleList` by tracing tensor mutations across iterations (e.g., `for block in self.blocks: x = block(x)` or `for layer in self.layers: x = layer(x, y)`).
   - How to handle structural conditional checks such as `if self.downsample is not None:` or `if self.use_residual:`.
2. Dynamic Shape Extraction:
   - Check if `ShapeExtractorBlock` exists in `backend/blocks/` or if a new block is needed.
   - How tuple unpackings like `B, C, H, W = x.shape` or `b, n, c = x.size()` should be mapped in the AST decompiler to `ShapeExtractorBlock` / Reshape / Flatten dataflow.
3. Roundtrip Compilation & Benchmark Suite (R4):
   - Inspect existing compiler (`backend/compiler/python_compiler.py` or similar), AST generator, and test suite (`backend/tests/`).
   - How roundtrip testing works: Code -> AST decompiler -> IR (`.ir.json`) -> AST compiler -> Python Code -> execution equivalence.
   - Survey canonical architectures to benchmark: ResNet / ConvNeXt (`torchvision`), ViT / MHA and Transformer MLP (`timm` / HuggingFace), UNet, and multi-class files.
   - Survey requirements for `test_real_world_models.py`, `torch.allclose` numerical verification with `atol=1e-4` on dummy inputs, and `npx tsc --noEmit`.

Boundaries:
- This is a READ-ONLY survey. DO NOT modify any code files.
- Produce a detailed handoff report in `d:\ML\ArchIDE\.agents\explorer_survey_3\handoff.md`.
- Include precise file paths, line numbers, test structures, and recommended implementation/benchmark design.
- Send a completion message back with a concise summary and the path to `handoff.md`.
