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

## 📖 Documentation & Agent Guidelines

We maintain comprehensive documentation for human developers and AI coding agents.

> ⚠️ **IMPORTANT**: Before writing any code or prompting an AI coding agent, read through the documentation in [`docs/`](docs/) thoroughly.

*   **[Documentation Index](docs/index.md)**: Your starting point for understanding the architecture, Block Registry, PyTorch Compiler, and project roadmap.
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
