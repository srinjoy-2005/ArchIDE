# ArchIDE: Custom Modules & Project Compilation Architecture

This document provides a comprehensive technical specification for the representation, resolution, shape inference, incremental compilation, and PyTorch code emission of custom submodules within ArchIDE.

---

## 1. Executive Summary & Design Decision

ArchIDE supports multi-tab, multi-module visual deep learning model authoring. Users can define reusable subgraphs (custom modules) as XML/JSON definitions and instantiate them inside larger models.

### Architectural Verdict
ArchIDE adopts a **Module Registry & Compiler-Linker Architecture with Dual-Target Codegen**:
1. **Primary Internal Representation & Development Target (Modular Python Package)**:
   - Each XML module compiles into an isolated, idiomatic Python file with relative imports (`from .modules.conv_block import ConvBlock`).
   - Powers fast incremental compilation ($O(1)$ updates), isolated namespace scoping, granular git diffs, and precise stack traces.
2. **Distribution & Scripting Target (Monolithic Single-File Linker)**:
   - For users needing single-file portability (Google Colab, Kaggle, quick script embeds), a compiler linker pass inlines all submodules in topological order into a single self-contained `model.py`.

```mermaid
flowchart TD
    subgraph Source["Source Files (Persistent / Git Tracked)"]
        P["archide.project.json (Project Manifest)"]
        M1["modules/conv_block.xml (ConvBlock)"]
        M2["modules/res_block.xml (ResidualBlock)"]
        Main["main_graph.xml (Main Classifier)"]
    end

    subgraph Compiler["ArchIDE Compiler Engine"]
        Parse["1. Module Registry & XML Parser"] --> DepDAG["2. Project Dependency Graph (DAG)"]
        DepDAG --> Tarjan["3. Cycle Detection (Tarjan / Kahn)"]
        Tarjan --> Shape["4. Inter-Module Shape Inference"]
        Shape --> IR["5. Typed Module AST IR"]
        IR --> TargetRouter{"Emission Target"}
    end

    subgraph Emission["Dual-Target Code Generation"]
        TargetRouter -->|Target A: Package (Default)| PkgGen["Modular Package Generator\n(__init__.py + module files)"]
        TargetRouter -->|Target B: Monolith (Export)| Linker["Linker & Topo-Inliner\n(Single-file consolidated AST)"]
    end

    P & M1 & M2 & Main --> Parse
    PkgGen --> OutPkg["build/dist/my_model_pkg/"]
    Linker --> OutSingle["dist/model.py"]
```

---

## 2. Evaluation of Core Approaches

| Architectural Dimension | Approach 1: Monolithic Single-File (`model.py`) | Approach 2: Modular Package (`my_model_pkg/`) | ArchIDE Hybrid: Dual-Target Linker |
|---|---|---|---|
| **Pythonic Readability** | ⚠️ Cluttered for large architectures with nested submodules. |  Clean, industry-standard package hierarchy. |  Package for codebases; single file for quick scripts. |
| **Incremental Compilation** | ❌ Full $O(N)$ recompilation of entire codebase on any edit. |  $O(1)$ local recompilation per module file. |  $O(1)$ during development; linking only on export. |
| **Namespace Isolation** | ❌ Single global namespace; risks collision between variable and class names. |  Fully scoped per Python module/file. |  Full module-level scoping with optional linker prefixing. |
| **Debugging & Stack Traces** | ⚠️ Unwieldy stack traces with fluctuating monolithic line numbers. |  Explicit file frames (`conv_block.py:L14`). |  Explicit file frames in dev; accurate source maps. |
| **Portability & Colab Usage**|  Trivial to copy-paste one file into notebook/script. | ⚠️ Requires directory structure or `pip install -e .`. |  Direct 1-click single-file export on demand. |
| **Git Version Control** | ❌ Editing any submodule causes massive diff churn. |  Granular, file-isolated diffs. |  Clean source XML and module diffs. |
| **Scalability (100+ modules)**| ❌ Memory and compile-time bottlenecks. |  Scales linearly; enables parallel builds. |  Optimized memory cache and DAG pruning. |

---

## 3. User Experience & IDE Workflow

The IDE eliminates the need for manual terminal commands during visual modeling. A dedicated CLI is available for CI/CD and automation.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  ArchIDE  —  [ ConvBnRelu.xml ]   [ MainGraph.xml* ]  [ + New Module ]                 │
│                                           [  Check Shapes ]  [ ▾ Export PyTorch ]      │
├───────────────┬────────────────────────────────────────────┬───────────────────────────┤
│ PALETTE       │ VISUAL CANVAS                              │ LIVE CODE PREVIEW         │
│ ▾ Core Layers │                                            │                           │
│   Linear      │  [ Input ] ──> [ ConvBnRelu ] ──> [ Linear ]│ class ConvBnRelu(...):    │
│   Conv2d      │                       │                    │     ...                   │
│ ▾ My Modules  │                       ▼                    │ class MainGraph(...):     │
│   ★ ConvBnRelu│                 [ Output ]                 │     ...                   │
└───────────────┴────────────────────────────────────────────┴───────────────────────────┘
```

### Visual IDE Workflow
1. **Multi-Tab Workspace**: Each open tab represents either the main model (`MainGraph.xml`) or a custom submodule (`ConvBnRelu.xml`).
2. **Automatic Palette Registration**: When a user creates and saves a module tab, ArchIDE automatically parses its interface and adds it as a draggable block under **"My Modules"** in the sidebar palette.
3. **Live Canvas Preview**: The code preview updates reactively, displaying code for the active module tab or the entire project.
4. **1-Click Export Modal**:
   - `Export as Python Package`: Generates a production-ready folder with `__init__.py` and relative imports.
   - `Export as Single File`: Generates a self-contained, inlined `model.py`.

### Optional Power-User CLI
For headless environments, automated testing, and CI/CD:
```bash
# Export single self-contained script
arch compile ./my_project -o model.py

# Export production package
arch compile ./my_project --target=package -o ./dist/my_model_pkg/

# Run static validation check
arch check ./my_project
```

---

## 4. Concrete End-to-End Walkthrough

### Step 1: User Defines Custom Submodule (`ConvBnRelu.xml`)
On tab `ConvBnRelu`, the user connects:
`Input (x)` $\rightarrow$ `Conv2d(in=3, out=32, k=3)` $\rightarrow$ `BatchNorm2d(num_features=32)` $\rightarrow$ `ReLU` $\rightarrow$ `Output (out)`.

**Generated XML (`modules/conv_bn_relu.xml`):**
```xml
<ArchModule id="mod_conv_bn_relu" name="ConvBnRelu">
  <Interface>
    <Inputs>
      <Port id="in_0" label="x" shape="[-1, 3, -1, -1]" />
    </Inputs>
    <Outputs>
      <Port id="out_0" label="out" shape="[-1, 32, -1, -1]" />
    </Outputs>
    <Parameters>
      <Param name="in_channels" type="int" default="3" />
      <Param name="out_channels" type="int" default="32" />
      <Param name="kernel_size" type="int" default="3" />
    </Parameters>
  </Interface>
  <Graph>
    <Node id="n1" block_id="input" />
    <Node id="n2" block_id="conv2d" in_channels="3" out_channels="32" kernel_size="3" padding="1" />
    <Node id="n3" block_id="batchnorm2d" num_features="32" />
    <Node id="n4" block_id="relu" />
    <Node id="n5" block_id="output" />
    <Edge source="n1" sourceHandle="out" target="n2" targetHandle="in" />
    <Edge source="n2" sourceHandle="out" target="n3" targetHandle="in" />
    <Edge source="n3" sourceHandle="out" target="n4" targetHandle="in" />
    <Edge source="n4" sourceHandle="out" target="n5" targetHandle="in" />
  </Graph>
</ArchModule>
```

---

### Step 2: User Uses Submodule in `MainGraph.xml`
In tab `MainGraph`, the user drags two `ConvBnRelu` blocks and one `Linear` block:
`Input` $\rightarrow$ `ConvBnRelu #1 (out=32)` $\rightarrow$ `ConvBnRelu #2 (out=64)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(in=64*8*8, out=10)` $\rightarrow$ `Output`.

---

### Step 3: Generated Code Comparison

#### Output Mode A: Modular Python Package (`dist/package/`)
**File 1: `modules/conv_bn_relu.py`**
```python
import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
```

**File 2: `main_graph.py`**
```python
import torch
import torch.nn as nn
from .modules.conv_bn_relu import ConvBnRelu

class MainGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBnRelu(in_channels=3, out_channels=32, kernel_size=3)
        self.block2 = ConvBnRelu(in_channels=32, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
```

#### Output Mode B: Single Consolidated File (`dist/standalone/model.py`)
```python
# Generated by ArchIDE Compiler (Monolithic Linker Target)
import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class MainGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBnRelu(in_channels=3, out_channels=32, kernel_size=3)
        self.block2 = ConvBnRelu(in_channels=32, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
```

---

## 5. Technical Implementation Details

### 5.1. Module Interface Contracts
Each custom module defines an **Interface Contract** separating its external signature from internal graph topology:
1. **Ports**: Explicit typed input and output sockets with symbolic or concrete shape constraints.
2. **Parameters**: Constructor arguments exposed to parent graphs (e.g. `in_channels`, `dropout_rate`).
3. **Shape Transfer Function**: Evaluates output dimensions from input dimensions without re-running intra-graph node traversals.

### 5.2. Dependency Graph & Cycle Detection
Inter-module references form a project Directed Acyclic Graph (DAG):
1. **Graph Construction**: An adjacency list maps each module to the modules it instantiates.
2. **Cycle Detection**: Kahn's algorithm or Tarjan's Strongly Connected Components (SCC) runs on the module dependency graph.
3. **Deterministic Sorting**: Submodules are compiled bottom-up in topological order.

### 5.3. Inter-Module Shape Inference
Shape inference proceeds in two passes:
1. **Intra-Module Verification**: Verifies internal node-to-node tensor compatibility starting from the module's declared `Input` ports.
2. **Inter-Module Contract Substitution**: When a parent graph invokes a custom module instance, the compiler binds the parent's actual tensor shapes and parameters to the module's interface contract, computing downstream tensor shapes.

### 5.4. Incremental Compilation & Build Caching
To maintain high responsiveness in the IDE:
1. **Two-Tier Hashing**:
   - `interface_hash`: SHA-256 of ports, parameters, and shape contracts.
   - `body_hash`: SHA-256 of internal nodes and connections.
2. **Selective Invalidation**:
   - If only `body_hash` changes, only that single module's `.py` file is re-emitted. Parent graphs skip re-inference and recompilation.
   - If `interface_hash` changes, direct and transitive consumers are invalidated, shape-checked, and re-compiled.

### 5.5. Namespaces & Collision Avoidance
1. **Module Scoping**: Class names follow PascalCase sanitized from the module name (`ConvBnRelu`).
2. **Collision Disambiguation**:
   - In package mode, modules in different subfolders reside in distinct Python files.
   - In monolithic single-file mode, duplicate names across folders are prefixed with their directory path (e.g., `Layers_Block`, `Heads_Block`).
3. **Keyword Protection**: Built-in Python keywords (`def`, `class`, `import`) and standard PyTorch layers (`Linear`, `Conv2d`) are prevented as bare module names.

### 5.6. Source Mapping & Error Attribution
1. **Source Maps (`.py.map`)**:
   Emitted Python files contain source maps mapping line ranges back to specific XML node IDs:
   ```json
   {
     "file": "conv_bn_relu.py",
     "mappings": [
       { "generated_line": 8, "xml_file": "modules/conv_bn_relu.xml", "node_id": "n2" }
     ]
   }
   ```
2. **Canvas Badging**:
   When runtime errors or shape mismatches occur, the IDE reads the source map or exception payload to highlight the exact visual node in red on the canvas.

### 5.7. File System Lifecycle: Source vs. Build Artifacts
- **Persistent Source Files**:
  - `archide.project.json`, `main_graph.xml`, `modules/*.xml`.
  - Stored in project root and tracked in Git.
- **Transient Build Artifacts**:
  - `.arch_cache/`, `build/`.
  - Automatically created by compiler for live preview and testing; excluded from Git via `.gitignore`.
- **Export Deliverables**:
  - `dist/package/` or `dist/standalone/`.
  - Generated on explicit user export.

---

## 6. Target Project Directory Structure

```text
my_model_project/
├── archide.project.json            # Project manifest & metadata
├── main_graph.xml                  # Main model graph
├── modules/                        # Reusable custom submodules
│   ├── conv_bn_relu.xml
│   ├── attention_head.xml
│   └── transformer_block.xml
├── .arch_cache/                    # Transient build cache (gitignored)
│   ├── manifest_cache.json
│   └── shape_cache.json
└── dist/                           # Final export outputs
    ├── package/                    # Target A: Modular Python Package
    │   ├── __init__.py
    │   ├── main_graph.py
    │   └── modules/
    │       ├── __init__.py
    │       ├── conv_bn_relu.py
    │       ├── attention_head.py
    │       └── transformer_block.py
    └── standalone/                 # Target B: Monolithic single-file model
        └── model.py
```
