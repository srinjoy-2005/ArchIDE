# ArchiDE: Roadmap & Future Features

This document outlines the short-term and long-term goals for the ArchiDE visual builder. 

## 🚧 Short-Term Features

### 1. Block Properties & Hyperparameter Tuning
- **Current State**: We have a Properties Sidebar that correctly syncs with uncontrolled React Flow nodes, allowing users to modify parameters.
- **To Do**: Expand the UI for parameter inputs (sliders, dropdowns for activation functions, etc.) and validate input types (e.g. ensuring `in_features` is an integer > 0).

### 2. Advanced Graph Validation & Shape Inference
- **Current State**: Kahn's algorithm detects cyclic dependencies in the backend. However, users can connect incompatible blocks without dimension validation.
- **To Do**: 
  - Implement a shape propagation engine (potentially running in the Python backend) to calculate tensor dimensions as they flow through the network.
  - Display visual warnings on the React Flow canvas if incompatible blocks are connected (e.g., dimension mismatch).

### 3. Complex Architectures (Non-Sequential)
- **Current State**: The backend compiler supports basic mathematical ops (`Add`, `Subtract`) and some stateful layers.
- **To Do**: Update the parsing logic in `backend/compiler.py` to fully support complex branching architectures, advanced skip connections (ResNets), and multi-input/output blocks (like `torch.cat`).

## 🚀 Long-Term Features

### 1. Save/Load & Export Capabilities
- **To Do**: 
  - Implement serialization so users can save their visual graph as a `.json` file (or a database record) and reload it later.
  - Add a feature to download the generated `.py` file directly to the user's local machine, or export the weights if a training loop is added.

### 2. Expand Block Library
- **To Do**: Gradually add more advanced blocks to `backend/registry.py` (e.g., Transformer Blocks, RNNs/LSTMs, custom loss functions, and optimizers).

### 3. JSON Schema Reference Design
- **Note**: Early on, a TypeScript-based JSON schema logic was proposed (detailed in `graph_ir_spec.md` and old IR preview files). While the actual implementation of the backend logic has been refactored entirely into Python (FastAPI/Pydantic), that original design remains a highly compelling reference. Future iterations involving complex graph serialization or frontend-backend synchronization should look to that design for inspiration.
