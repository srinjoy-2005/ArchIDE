# ArchiDE

A modern, web-native visual IDE for building machine learning model architectures by dragging, dropping, and connecting blocks — automatically generating clean, idiomatic PyTorch (`nn.Module`) code. Think SimulIDE, but for deep learning.

## Vision

ArchiDE provides a visual canvas where neural network layers (Linear, Conv2D, Flatten, Attention, Residual Add, Concat) can be connected into arbitrary Directed Acyclic Graphs (DAGs). The visual model is validated and shape-inferred in real-time, instantly emitting standard PyTorch code.

## Status

🚧 **PoC Architecture & Design Phase** — Specifications and documentation ready for implementation.

## Project Structure

```
archide/
├── docs/                          # Ready-context documentation for developers & AI agents
│   ├── implementation_plan.md     # Full project architecture, tech stack & implementation roadmap
│   ├── ml_forge_comparison.md     # In-depth architectural analysis of MLForge & ArchiDE adaptations
│   ├── block_registry_spec.md     # Block definitions, parameters, ports & shape inference rules
│   └── graph_ir_spec.md           # Graph IR JSON schema & PyTorch code generator pipeline
├── ml_forge-main/                 # Reference open-source desktop codebase
└── README.md
```

## Contributing & Agent Guidelines

> ⚠️ **IMPORTANT**: Before writing any code or prompting an AI coding agent, read through the documentation in [`docs/`](docs/) thoroughly.

### Key Documentation Links

1. **Architecture & Tech Stack Plan** — Read [`docs/implementation_plan.md`](docs/implementation_plan.md) for the high-level architecture, Next.js + React Flow framework decision, and phase roadmap.
2. **MLForge Adaptation Analysis** — Read [`docs/ml_forge_comparison.md`](docs/ml_forge_comparison.md) for what was adapted from the desktop MLForge app and what was stripped out.
3. **Block Registry Specification** — Read [`docs/block_registry_spec.md`](docs/block_registry_spec.md) for block metadata, parameters, input/output ports, and shape propagation rules.
4. **Graph IR & Codegen Spec** — Read [`docs/graph_ir_spec.md`](docs/graph_ir_spec.md) for the Graph IR JSON schema and code generation engine rules.

### Prompting AI Agents

When tasking AI coding agents with working on this project, provide the relevant docs as context:

```
Refer to docs/implementation_plan.md for overall architecture and roadmap.
Refer to docs/ml_forge_comparison.md for MLForge adaptations.
Refer to docs/block_registry_spec.md for block definitions and shape formulas.
Refer to docs/graph_ir_spec.md for the Graph IR schema and code generation pipeline.
```

## Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | Next.js (App Router) |
| **Canvas** | React Flow (`@xyflow/react`) |
| **State Management** | Zustand |
| **Graph IR** | JSON Schema |
| **Code Generation** | Client-side Topological Compiler |
| **Target Output** | PyTorch (`nn.Module`) |

## License

MIT
