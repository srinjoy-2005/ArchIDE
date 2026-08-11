# Implementation Plan: Tensor Shape Broadcasting & Testing Setup

## Goal
Implement strict PyTorch broadcasting validation for element-wise tensor operations (Add, Sub, Mul, Div) to catch mismatched shapes (like `(4, 3)` and `(3, 4)`). Furthermore, establish a testing mechanism (using `pytest`) for the backend to automate the validation of dimensions and API responses.

## Implemented Changes

### 1. Broadcasting Logic (`backend/blocks/tensor_ops.py`)
- Add a new `broadcast_shapes(*shapes: Tuple) -> Tuple` utility function.
- This function will align shapes to the right, pad with 1s, and validate dimensions according to PyTorch broadcasting semantics. 
- It will raise a `ValueError` if the shapes are incompatible.
- Update `infer_shapes` in `AddBlock`, `SubBlock`, `MulBlock`, and `DivBlock` to use this new utility instead of simply blindly returning the first input's shape.

#### [MODIFY] [tensor_ops.py](file:///home/iamsrinjoy/random_stuff/archide/backend/blocks/tensor_ops.py)
- Inject `broadcast_shapes` function.
- Modify `AddBlock`, `SubBlock`, `MulBlock`, `DivBlock` to pass all their incoming tensor shapes into this function.

### 2. Testing Framework Setup (`backend/tests/`)
- Install `pytest` and `httpx` in the backend virtual environment to support testing.
- Create unit tests for the block logic and integration tests for the FastAPI application.

#### [NEW] [test_tensor_ops.py](file:///home/iamsrinjoy/random_stuff/archide/backend/tests/test_tensor_ops.py)
- Unit tests verifying that `broadcast_shapes` correctly succeeds for compatible shapes (e.g. `(4, 1)` and `(1, 3)` -> `(4, 3)`) and raises exceptions for incompatible shapes (e.g. `(4, 3)` and `(3, 4)`).

#### [NEW] [test_api.py](file:///home/iamsrinjoy/random_stuff/archide/backend/tests/test_api.py)
- Integration tests using FastAPI's `TestClient` to send mock topologies (nodes + edges) to `/api/check`.
- Asserts that a topology containing a `(4, 3)` and `(3, 4)` multiplication fails shape validation and returns a `422 Unprocessable Content` response.

## Verification (Completed)
1. Install testing dependencies inside `backend/.venv`.
2. Run `pytest backend/tests/` and ensure all shape mismatch and broadcasting test cases pass.
3. Validate that the frontend now correctly surfaces the shape error when invalid shapes are connected to a `Multiply` block.

> [!IMPORTANT]  
> This plan has been successfully implemented. The broadcasting logic is active and the `pytest` suite is passing.
