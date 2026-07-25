# ArchiDE: Progress & Status Report

This document outlines the current state of the ArchiDE visual builder, detailing what features have been successfully implemented and what remains for future iterations.

---

## ✅ What is Done

### 1. Project Scaffolding & Setup
- **Next.js App Router Setup**: Fully initialized with React 18, TypeScript, and TailwindCSS.
- **Project Configuration**: Set up strict TypeScript checks, ESLint configuration, and a proper Next/Node `.gitignore`.
- **Global State Management**: Integrated `zustand` for lightweight global state (managing the generated PyTorch code preview).

### 2. User Interface (UI) & Aesthetics
- **Dark Mode Layout**: Created a sleek, full-screen IDE interface featuring a block palette sidebar, a central canvas, and a right-side code preview panel.
- **Glassmorphism & Styling**: Implemented premium aesthetics (`backdrop-blur`, shadow glows, gradient text) as defined in the initial requirements.

### 3. Visual Canvas Engine (React Flow)
- **Drag and Drop Integration**: Wired HTML5 Drag & Drop API with React Flow. Users can drag blocks from the sidebar and drop them onto the canvas coordinate system.
- **Custom Nodes**: Built a `CustomNode` component to render visually distinct blocks with top/bottom handles for connections.
- **Node & Edge Management**: 
  - Standard edge drawing between node handles.
  - Node and Edge selection.
  - Deletion enabled via the `Backspace` or `Delete` keys.
- **Block Palette**: Added a comprehensive sidebar palette categorized into Core Layers, Activations, Regularization, and Pooling.

### 4. Code Generation Engine
- **Topological Parsing**: Developed an initial `codegen.ts` engine capable of translating a connected graph of nodes into PyTorch logic.
- **Instant Code Export**: Connected the "Export PyTorch" button to traverse the nodes and edges, instantly generating the `__init__` layer instantiations and the `forward` pass sequential logic.
- **Live Preview**: The generated code string is dynamically injected into the right-hand code panel.

---

## 🚧 What is Left (Next Steps & Future Roadmap)

### 1. Block Properties & Hyperparameter Tuning
- **Current State**: Nodes currently emit default parameters (e.g., Conv2D always generates `in_channels=3, out_channels=16`).
- **To Do**: Build a **Properties Sidebar** that appears when a user clicks a node. This will allow users to customize parameters (like `kernel_size`, `stride`, `in_features`, `p` for Dropout) and have those changes reflect instantly in the codegen.

### 2. Advanced Graph Validation & Shape Inference
- **Current State**: The UI allows connecting any block to any block without validating tensor dimensions.
- **To Do**: 
  - Implement a shape propagation engine to calculate tensor dimensions as they flow through the network.
  - Display visual warnings if incompatible blocks are connected (e.g., dimension mismatch).
  - Add cycle detection to prevent infinite loops in the DAG.

### 3. Complex Architectures (Non-Sequential)
- **Current State**: The codegen performs a basic sequential traversal (assuming one main path).
- **To Do**: Update the parsing logic in `codegen.ts` to fully support branching architectures, skip connections (ResNets), and multi-input blocks (like `torch.cat` or `Add`).

### 4. Save/Load & Export Capabilities
- **To Do**: 
  - Implement serialization so users can save their visual graph as a `.json` file and reload it later.
  - Add a feature to download the generated `.py` file directly to the user's local machine.

### 5. Expand Block Library
- **To Do**: Gradually add more advanced blocks based on the `block_registry_spec.md` (e.g., Transformer Blocks, RNNs/LSTMs, custom loss functions if training is added later).
