# MLForge Architectural Analysis & ArchiDE Adaptation

This document analyzes the open-source **MLForge** (`zaina-ml/ml_forge`) desktop application architecture, detailing what ArchiDE adapts, what unnecessary features are stripped out, and how ArchiDE's web architecture improves upon it.

---

## 1. Executive Summary & Comparison Matrix

MLForge is a Python desktop app built with DearPyGui designed for end-to-end computer vision model building and training. While MLForge covers data loading, augmentation, visual model building, and training execution, **ArchiDE** isolates and elevates the **Model Architecture Graph → PyTorch Code Generation** pipeline into an intuitive, web-native visual IDE.

| Dimension | MLForge (`zaina-ml/ml_forge`) | ArchiDE (Visual ML IDE) |
|---|---|---|
| **Platform** | Python Desktop App (DearPyGui / PySide) | Web Application (Next.js + React Flow) |
| **Primary Scope** | Full ML training suite (Data, Model, Train, Inference) | Model Architecture & PyTorch Code Generation |
| **Model Topology** | Primarily sequential single-chain models (`x = layer(x)`) | Full DAGs (Skip connections, multi-input, split, attention) |
| **Code Generation Target** | Full `train.py` script (DataLoader + Model + Training loop) | Standalone, clean PyTorch `nn.Module` class |
| **Shape Propagation** | Basic sequential shape inference | Graph-wide port-aware tensor shape propagation |
| **Execution Engine** | PyTorch training execution in app process | Client-side visual graph to code emitter (Zero setup) |

---

## 2. Deep Dive: MLForge Architecture Breakdown

Inspecting the `ml_forge` codebase reveals a clean modular engine structure:

```
ml_forge/
├── engine/
│   ├── blocks.py        # Master dictionary of block categories, parameters, and defaults
│   ├── graph.py         # Graph building, topological sort, shape inference logic
│   ├── generator.py     # Template-based PyTorch script string emitter
│   ├── autofill.py      # Downstream layer shape auto-population
│   ├── run.py           # Training loop execution (Stripped out for ArchiDE)
│   ├── metrics.py       # Loss/accuracy tracking (Stripped out for ArchiDE)
│   └── inference.py     # Real-time visual inference (Stripped out for ArchiDE)
└── graph/
    ├── nodes.py         # DearPyGui UI wrappers for node blocks
    └── links.py         # UI link manager
```

### Key Architectural Concepts Learned & Adapted from MLForge

1. **Declarative Block Metadata Registry (`blocks.py`)**:
   MLForge defines blocks declaratively with parameter definitions, default values, tooltip guidance, and input/output pins. ArchiDE expands this concept into a JSON schema that includes PyTorch template strings and shape calculation lambdas.

2. **Topological Sort Execution (`graph.py`)**:
   MLForge uses Kahn's Algorithm on adjacency lists to ensure layers in the visual graph are processed in strictly correct execution order. ArchiDE adapts this to handle arbitrary DAGs with multi-input merge nodes (like `torch.cat` or `x + residual`).

3. **Layer Code Mapping (`generator.py`)**:
   MLForge maps block labels to PyTorch constructors using parameter string templates (e.g., `nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size})`). ArchiDE adopts a similar template replacement engine for `__init__` and `forward()` calls.

4. **Shape Inference & Parameter Auto-Fill (`autofill.py`)**:
   MLForge calculates spatial dimensions `[B, C, H, W]` through Conv/Pool/Linear layers so `in_channels` or `in_features` can auto-fill. ArchiDE extends this to full tensor shape propagation across all DAG nodes.

---

## 3. Features Stripped Out for ArchiDE PoC

To build a focused, lightning-fast model design IDE, ArchiDE explicitly removes the following heavy desktop dependencies from MLForge:

- **Training Execution Engine (`run.py`)**: No PyTorch CUDA/CPU training execution or background processes.
- **Dataset & Augmentation Pipeline (`datasets`, `transforms`)**: No dataset loading (MNIST, CIFAR, ImageFolder) or image augmentation nodes.
- **Metrics & Live Charts (`metrics.py`)**: No real-time loss/accuracy monitoring or TensorBoard integration.
- **Inference Window (`inference.py`)**: No interactive model testing on local images.
- **PySide/DearPyGui Desktop GUI**: Replaced with modern web standards (HTML5 Canvas/SVG via React Flow).

---

## 4. Architectural Improvements in ArchiDE Web Version

1. **Web-Native DAG Canvas (Next.js + React Flow)**:
   - Modern, high-performance UI using `@xyflow/react`.
   - Smooth pan/zoom, customizable handles/pins, mini-map, alignment grid, dark glassmorphism design.

2. **Multi-Port DAG Topologies**:
   - MLForge's model tab was restricted to single-chain pipelines.
   - ArchiDE natively supports multi-input blocks (`Add`, `Concat`, `MultiHeadAttention` with Q/K/V pins) and fan-out skip connections (ResNet, DenseNet, Transformers).

3. **Live Code Generation & Instant Preview**:
   - As nodes are moved, edited, or connected, PyTorch `nn.Module` code is re-generated instantly in a side-by-side Monaco/Prism editor.
