# ArchiDE - Documentation Index

Welcome to the ArchiDE documentation. This directory contains detailed technical specifications, architectural overviews, roadmaps, and historical changelogs.

## 🏗 Architecture & Overview
*   **[Architecture Overview](architecture_overview.md)**: High-level overview of the Next.js frontend and Python FastAPI backend architecture.
*   **[Frontend Architecture](frontend/architecture.md)**: Technical guidelines for React Flow uncontrolled canvas, custom node/edge components, and Zustand store.
*   **[Compiler Design](backend/compiler_design.md)**: Details on Kahn's topological sort algorithm, static shape inference pass, and PyTorch AST code generation.
*   **[Backend Testing](backend/testing.md)**: Specifications for automated backend unit and integration test suites.
*   **[Rigorous Testing Strategy Plan](testing_plan.md)**: High-level roadmap and strategy for backend and frontend testing.

## 📖 Specifications
*   **[Block Registry Specification](backend/block_registry_spec.md)**: Formal specification for block schemas, port definitions, and OOP block interfaces.
*   **[Block Implementation Catalog & Status](backend/blocks_status.md)**: Catalog of 28 implemented blocks with exact file links and pending block roadmap.
*   **[Graph IR Spec (Legacy)](frontend/graph_ir_spec.md)**: The original TypeScript-based JSON schema specification (retained for design reference).

## 🚀 Roadmaps & Future
*   **[TODO / Roadmap](TODO.md)**: Short-term and long-term goals (Shape Inference, Model Export, Advanced Blocks).
*   **[MLForge Comparison](ml_forge_comparison.md)**: How ArchiDE compares and contrasts with the open-source MLForge desktop app.

## 📝 Changelog
*   **[August 14, 2026](changelog/2026_08_14.md)**: Overhauled docs with relative links, sync workflow (`scripts/sync.py`, CI), variadic ports, orphan edge resilience, and master roadmap overhaul.
*   **[August 11, 2026](changelog/2026_08_11.md)**: Implemented backend testing suite (block inference, cycle detection, AST compilation checks), testing plan, and TypeScript configs.
*   **[July 28, 2026](changelog/2026_07_28.md)**: Compiler OO architecture refactor, shape inference pass & ShapeError 422 responses, three-section Properties Panel, and 10 new blocks.
*   **[July 26, 2026](changelog/2026_07_26.md)**: Major architecture shift to Python (FastAPI) backend, Kahn's algorithm PyTorch compiler, and React Flow state synchronization fixes.
*   **[July 25, 2026](changelog/2026_07_25.md)**: Initial project scaffolding, dark mode IDE UI layout, and base React Flow drag-and-drop canvas setup.
