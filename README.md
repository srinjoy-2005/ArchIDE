# ArchiDE

A visual, block-based IDE for building ML models — drag, drop, and connect layers to generate clean PyTorch code. Think SimulIDE, but for neural networks.

## Vision

Build neural network architectures visually by stacking and connecting blocks (Linear, Conv2d, Attention, Transformer, etc.) on a canvas. The graph is validated in real-time and converted directly to idiomatic PyTorch `nn.Module` code.

## Status

🚧 **Early development** — architecture planning phase.

## Project Structure

```
archide/
├── docs/                          # Architecture & design documentation
│   ├── implementation_plan.md     # Backend architecture & code generation pipeline
├── src/                           # Source code 
└── README.md
```

## Contributing

> **Before writing any code or prompting an AI agent**, read through the docs in [`docs/`](docs/) thoroughly.

1. **Start with the plan** — Read [`docs/implementation_plan.md`](docs/implementation_plan.md) for the full backend architecture, design decisions, and phased implementation order.


This ensures the agent follows established patterns and doesn't reinvent decisions that have already been made.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | tbd |
| Graph IR | JSON |
| Code Generation | tbd |
| Output | PyTorch (`nn.Module`) |

## License


