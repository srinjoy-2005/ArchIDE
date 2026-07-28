# ArchiDE - Documentation Index

Welcome to the ArchiDE documentation. This directory contains detailed technical specifications, architectural overviews, roadmaps, and historical changelogs.

## 🏗 Architecture & Overview
*   **[Architecture Overview](architecture_overview.md)**: High-level overview of the Next.js frontend and Python FastAPI backend architecture, along with details on the graph processing engine.
*   **[Compiler Design](compiler_design.md)**: Details on Kahn's topological sort algorithm, string-based AST generation, and cycle detection.

## 📖 Specifications
*   **[Block Registry Specification](block_registry_spec.md)**: Formal specification for all blocks (Linear, Conv2D, Add, etc.) defined in `backend/registry.py` and validated by `backend/models.py`.
*   **[Foundational Blocks Design](foundational_blocks_design.md)**: Reference material on foundational building blocks for neural networks.
*   **[Graph IR Spec (Legacy)](graph_ir_spec.md)**: The original TypeScript-based JSON schema specification. Note that actual logic is now handled in Python, but this remains as a strong design reference.

## 🚀 Roadmaps & Future
*   **[Roadmap & Features](roadmap_and_features.md)**: Short-term and long-term goals (Shape Inference, Model Export, Advanced Blocks).
*   **[MLForge Comparison](ml_forge_comparison.md)**: How ArchiDE compares and contrasts with the open-source MLForge desktop app.

## 📝 Changelog
*   **[July 28, 2026 - Srinjoy](changelog/2026_07_28_srinjoy.md)**: Refactored compiler architecture to use object-oriented block definitions for decoupling functionalities and enabling static shape inference.
*   **[July 26, 2026 - Gourav Roy](changelog/2026_07_26_gourav.md)**: Major architecture shift to Python (FastAPI) backend, Kahn's algorithm PyTorch compiler, and critical UI state synchronization fixes in React Flow.
*   **[July 25, 2026 - Srinjoy Mukherjee](changelog/2026_07_25_srinjoy.md)**: Initial project scaffolding, UI creation, dark mode glassmorphism layout, and base React Flow drag-and-drop canvas setup.
