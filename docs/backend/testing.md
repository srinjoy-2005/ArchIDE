# ArchIDE: Backend Testing Architecture

This document describes the automated test suite for the ArchIDE Python backend, covering API endpoints, block shape inference, topological sorting, broadcasting math, and code generation validation.

> [!NOTE]
> **Source Directory**: [`backend/tests/`](../../backend/tests/)

---

## 1. Test Suite Organization

| Test File | Target Scope | Key Assertions & Scenarios |
|---|---|---|
| [`test_api.py`](../../backend/tests/test_api.py) | FastAPI Endpoints (`/api/check`, `/api/compile`, `/api/blocks`) | - `GET /api/blocks`: Returns valid JSON list of `BlockDef` schemas.<br>- `POST /api/check`: Rejects mismatched multiplication shapes with HTTP `422` and structured `ShapeError` payload.<br>- `POST /api/compile`: Returns HTTP `400` on graph cycles; returns HTTP `200` with valid Python code on valid graphs. |
| [`test_blocks.py`](../../backend/tests/test_blocks.py) | Block Shape Inference & Auto-Inference | - `LinearBlock`: Validates output dimensions, auto-infers `in_features` from input when `-1`, raises `ValueError` on mismatched last dimension.<br>- `Conv2DBlock`: Computes spatial dimensions $((H + 2P - d(K-1) - 1)/S + 1)$, auto-infers `in_channels` when `-1`, catches negative spatial dimensions.<br>- `InputBlock`: Gracefully parses shape strings. |
| [`test_compiler.py`](../../backend/tests/test_compiler.py) | Kahn's Algorithm & Code Synthesis | - `topological_sort`: Verifies linear execution ordering.<br>- `topological_sort` cycle detection: Asserts `ValueError` raised on circular edges.<br>- `generate_pytorch_code`: Verifies generated code compiles cleanly (`compile(code, "<string>", "exec")`) and contains expected `nn.Module` class structure. |
| [`test_tensor_ops.py`](../../backend/tests/test_tensor_ops.py) | Tensor Broadcasting Utilities | - `broadcast_shapes`: Aligns dimensions right-to-left, tests padding with 1s, supports `"ANY"` wildcards, and raises `ValueError` on incompatible shapes. |

---

## 2. Running Backend Tests

Run all backend tests using `pytest` from the project root:

```bash
# Run entire backend test suite
pytest backend/tests/ -v

# Run a specific test file
pytest backend/tests/test_compiler.py -v
```
