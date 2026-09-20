## 2026-09-20T07:50:34Z
You are Worker 1 executing Milestone M1 for ArchIDE.
Your working directory is: d:\ML\ArchIDE\.agents\worker_m1_1
Original user request reference: d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md
Project specification: d:\ML\ArchIDE\.agents\PROJECT.md
Survey & Implementation Specification: d:\ML\ArchIDE\.agents\explorer_survey_1\handoff.md

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Mission:
Implement Requirement R1: Core Block Extensions & Scalar Binary Operations.

Write Ownership (You exclusively own and may edit these files):
- `backend/blocks/activations.py`
- `backend/blocks/core.py`
- `backend/blocks/tensor_ops.py`
- `backend/blocks/__init__.py`
- `backend/compiler.py`
- `backend/python_decompiler.py` (M1 updates only: LAYER_MAP, FUNCTIONAL_MAP, shape lookahead)
- `backend/block_schema.json`
- `backend/tests/test_r1_blocks_and_scalars.py`

Required Implementation Details (see d:\ML\ArchIDE\.agents\explorer_survey_1\handoff.md for complete code snippets):
1. `GELUBlock` & `SiLUBlock` in `backend/blocks/activations.py`:
   - `GELUBlock`: approximate param ('none' or 'tanh'), infer_shapes, emit_init (`nn.GELU(approximate=...)`), emit_forward (`self.layer_xxx(in_var)`).
   - `SiLUBlock`: inplace param (bool), infer_shapes, emit_init (`nn.SiLU(inplace=...)`), emit_forward (`self.layer_xxx(in_var)`).
2. `Conv1DBlock` & `EmbeddingBlock` in `backend/blocks/core.py`:
   - `Conv1DBlock`: parameters (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), shape inference for 3D tensor `(B, C, L)`, LAZY support, error guards, emit_init (`nn.Conv1d` or `nn.LazyConv1d`), emit_forward.
   - `EmbeddingBlock`: parameters (num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse), shape inference `(*in_shape, embedding_dim)`, error guards, emit_init (`nn.Embedding(...)`), emit_forward.
3. Register in `backend/blocks/__init__.py`:
   - Add `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, `EmbeddingBlock` to `_BLOCK_INSTANCES`.
4. Scalar Binary Operations in `backend/blocks/tensor_ops.py`:
   - For `AddBlock`, `SubBlock`, `MulBlock`, `DivBlock`:
     - Add `scalar_a` and `scalar_b` to `BlockDef.params`.
     - In `emit_forward`: correctly check and emit `params.get("scalar_a")` and `params.get("scalar_b")` when operands are scalar or missing incoming edges.
     - Expressions like `x + 1`, `1 + x`, `x - 1`, `1 - x`, `x * 2`, `x / math.sqrt(self.d_model)` must compile without dropping scalar values or emitting invalid `None`.
5. Compiler Updates in `backend/compiler.py`:
   - Add `"import math"` to `imports` in `generate_pytorch_code` (so expressions like `math.sqrt(d)` execute).
   - Add `"conv1d"`, `"embedding"`, `"gelu"`, `"silu"` to layer naming tuples.
   - Support `LAZY` for `"conv1d"` in param substitution.
6. Python Decompiler Updates in `backend/python_decompiler.py`:
   - Add `"Embedding"` to `LAYER_MAP`.
   - Update `"SiLU"` and `"silu"` to map to `"silu"` instead of `"gelu"` in `LAYER_MAP` and `FUNCTIONAL_MAP`.
   - Add shape inference lookahead for `conv1d` and `embedding`.
7. Schema Dump:
   - Run `python backend/dump_block_schema.py` to regenerate `backend/block_schema.json` (must have 34 blocks).
8. Comprehensive Verification Tests in `backend/tests/test_r1_blocks_and_scalars.py`:
   - Shape inference, code generation, error checks for all 4 new blocks.
   - Scalar forward code generation for Add, Sub, Mul, Div.
   - Real PyTorch forward pass execution of generated code with `torch.allclose` or exact assertion on dummy tensors.

Verification Commands to Execute:
- `pytest backend/tests/test_r1_blocks_and_scalars.py`
- `pytest backend/tests/` (ensure 100% existing tests pass)
- `npx tsc --noEmit` (ensure 0 TypeScript errors)

Deliverables:
- Write full report to `d:\ML\ArchIDE\.agents\worker_m1_1\handoff.md` with:
  - Observation: what was changed across all files.
  - Logic Chain: rationale and architecture.
  - Verification: exact terminal commands and results (test outputs).
- Send completion message to orchestrator with path to `handoff.md`.
