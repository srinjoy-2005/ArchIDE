# ArchiDE

A modern, web-native visual IDE for building machine learning model architectures by dragging, dropping, and connecting blocks — automatically generating clean, idiomatic PyTorch (`nn.Module`) code.

## 🚀 Running the Project

ArchiDE requires both the Next.js frontend and the FastAPI backend to run simultaneously.

### 1. Start the Python Backend
```bash
cd backend
# Optionally activate a virtual environment
pip install -r requirements.txt
uvicorn main:app --reload
```
*Runs on `http://localhost:8000`*

### 2. Start the Frontend
```bash
npm install
npm run dev
```
*Runs on `http://localhost:3000`*

## 🔄 Model Decompilation & Compilation Pipeline

ArchiDE supports seamless bidirectional conversion between pure PyTorch (`.py`), human/AI-readable Agentic IR (`.ir.json`), and visual canvas graphs (`.arch`).

To decompile your PyTorch code in `workspace/python/` into visual `.arch` graphs:
```bash
npm run decompile:py && npm run compile:arch
```

- **`npm run decompile:py`**: Decompiles all PyTorch models in `workspace/python/` into clean Agentic IR (`workspace/ir/*.ir.json`).
- **`npm run compile:arch`**: Compiles and validates IR into visual canvas graphs (`workspace/graphs/*.arch`) with full shape inference and syntax checks.
- *Note*: Running these commands will **never** modify or overwrite your original Python source files.

For deep architectural details on the AST parser and IR schema, see **[Model Decompilation Documentation](docs/decompilation.md)**.

## 📖 Documentation & Agent Guidelines

We maintain comprehensive documentation for human developers and AI coding agents.

> ⚠️ **IMPORTANT**: Before writing any code or prompting an AI coding agent, read through the documentation in [`docs/`](docs/) thoroughly.

*   **[Documentation Index](docs/index.md)**: Your starting point for understanding the architecture, Block Registry, PyTorch Compiler, and project roadmap.
*   **[Decompilation & IR Guide](docs/decompilation.md)**: Complete guide to the PyTorch AST decompiler, IR JSON schemas, and bidirectional pipelines.
*   **[Agent Guardrails](.agents/project_context.md)**: AI agents MUST read this file for specific constraints (like React Flow uncontrolled mode logic) before making any modifications.

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend Framework** | Next.js (App Router), TailwindCSS |
| **Visual Canvas** | React Flow (`@xyflow/react`) |
| **State Management** | Zustand |
| **Backend API** | Python (FastAPI), Pydantic |
| **Code Generation** | Python AST Generator via Kahn's Topological Sort |
| **Target Output** | PyTorch (`nn.Module`) |

## License
MIT
