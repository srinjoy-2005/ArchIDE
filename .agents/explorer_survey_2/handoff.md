# Handoff Report — Requirement R2: Multilevel Files & Multi-Class Decompilation

**Summary**: This survey investigates the ArchIDE PyTorch AST decompiler (`backend/python_decompiler.py`), the Agent Graph compiler (`backend/agent_compiler.py`), and the code generator (`backend/compiler.py`) for Requirement R2. It identifies the exact root causes of single-class truncation and attribute chain dropouts, and provides an actionable, end-to-end architecture to extract all `nn.Module` classes into distinct, interconnected IRs (`.ir.json`), resolve recursive imports, and map deep attribute calls (`self.backbone.layer1(x)`, `self.features[0].conv(x)`) into fully connected dataflow graphs.

---

## 1. Observation

### 1.1 Current Decompiler Architecture & Single-Class Truncation
In `backend/python_decompiler.py`:
- **Class Discovery Loop (Lines 161–171)**:
  ```python
  # 2. Find nn.Module Class
  module_class: Optional[ast.ClassDef] = None
  for stmt in tree.body:
      if isinstance(stmt, ast.ClassDef):
          base_names = [ast.unparse(b) for b in stmt.bases]
          if any("Module" in b for b in base_names) or not module_class:
              module_class = stmt

  if not module_class:
      raise ValueError("No nn.Module class found in Python source code.")
  ```
  The loop continuously reassigns `module_class = stmt` whenever a class has `"Module"` in its bases or `module_class` is None. Consequently, any file containing multiple `nn.Module` classes (e.g. `BasicBlock` and `ResNet`, or `EncoderLayer`, `DecoderLayer`, `Transformer`) overwrites `module_class` until only the **last class in the file** is retained. All earlier classes are completely discarded.

- **Class Processing and Return (Lines 172–193)**:
  ```python
  model_name = file_stem if file_stem != "main" else (module_class.name.lower() or "main")
  ...
  init_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
  if init_method:
      self._parse_init(init_method)

  forward_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
  if forward_method:
      self._parse_forward(forward_method)
  ...
  return {
      "name": model_name,
      "variables": self.init_variables,
      "nodes": self.nodes,
      "edges": self.edges,
  }
  ```
  `decompile_source()` only parses a single `ClassDef` and returns a single IR dictionary. Furthermore, `PyTorchASTDecompiler` holds mutable state (`self.init_variables`, `self.layer_instances`, `self.nodes`, `self.edges`, `self.env`, `self.node_counters`) on `self` without per-class isolation.

- **Import Tracking Limitations (Lines 150–160)**:
  ```python
  for stmt in tree.body:
      if isinstance(stmt, ast.ImportFrom):
          mod = stmt.module or ""
          mod_path = mod.strip(".").replace(".", "/")
          for alias in stmt.names:
              self.imports[alias.name] = f"{mod_path}" if mod_path else alias.name
      elif isinstance(stmt, ast.Import):
          for alias in stmt.names:
              self.imports[alias.name] = alias.name.strip(".").replace(".", "/")
  ```
  `alias.asname` is ignored. If `from modules.layers import Block as CustomBlock`, only `"Block"` is registered in `self.imports`, so references to `CustomBlock` in `__init__` fail to resolve. In addition, `mod.strip(".")` strips all leading dots without counting relative directory hierarchy levels (e.g., `from ..common import Layer`).

- **Local Class Reference Failure in `_parse_layer_instantiation` (Lines 305–324)**:
  ```python
  # Case 4: Custom Submodule / Class
  custom_mod_id = self.imports.get(raw_class_name)
  if not custom_mod_id:
      lowered = raw_class_name.lower()
      if self.workspace_dir:
          cand_paths = [...] # checks modules/lowered.arch, etc.
  ```
  If `ClassA` instantiates `ClassB` defined in the same file, `ClassB` is not in `self.imports`. The decompiler searches `self.workspace_dir/modules/classb.arch`, which does not exist because `ClassB` was never decompiled. It falls back to `custom_mod_id = "classb"` with an unknown block structure.

- **Sequential and ModuleList Incomplete Introspection (Lines 270–303)**:
  In `Sequential` (Lines 271–286), arguments are stored only as string literals (`params[f"param_{idx}"] = ...`) without extracting child layer definitions. In `ModuleList` (Lines 289–303), layers are only extracted if `call.args[0]` is a literal `ast.List` or `ast.Tuple`, missing list comprehensions (`[Block() for _ in range(N)]`).

---

### 1.2 Deep Object Attribute Chains in Forward Traversal
In `backend/python_decompiler.py`:
- **Current `_parse_call` Dispatch (Lines 472–542)**:
  ```python
  # 1A. Check if calling indexed module list on self: self.layers[i](x)
  if (
      isinstance(call.func, ast.Subscript)
      and isinstance(call.func.value, ast.Attribute)
      and isinstance(call.func.value.value, ast.Name)
      and call.func.value.value.id == "self"
  ):
      ...

  # 1B. Check if calling self.layer_xxx or self.custom_xxx
  if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
      ...
  ```
  - Branch 1A strictly matches only `self.<attr>[<slice>](...)` (1 attribute + 1 subscript).
  - Branch 1B strictly matches only `self.<attr>(...)` (1 attribute).

- **Handling of Nested Attribute Chains**:
  For expressions like `self.backbone.layer1(x)`:
  - `call.func` is `ast.Attribute(value=ast.Attribute(value=ast.Name(id='self'), attr='backbone'), attr='layer1')`.
  - Branch 1A fails (`call.func` is not `Subscript`).
  - Branch 1B fails (`call.func.value` is an `ast.Attribute`, NOT an `ast.Name`).
  - Falls to Branch 2 (functional calls, lines 544–621), which checks `torch.*`, `F.*`, tensor methods (`flatten`, `transpose`, etc.). None match `"layer1"`.
  - Drops to line 622: `return (None, "out")`.

  For expressions like `self.features[0].conv(x)`:
  - `call.func` is `ast.Attribute(value=ast.Subscript(value=ast.Attribute(self, 'features'), 0), attr='conv')`.
  - Branch 1A fails (`call.func` is not `Subscript`).
  - Branch 1B fails (`call.func.value` is `ast.Subscript`, NOT `ast.Name`).
  - Drops to line 622: `return (None, "out")`.

- **Silent Dataflow Destruction (Lines 381–387)**:
  ```python
  if isinstance(stmt, ast.Assign):
      for target in stmt.targets:
          if isinstance(target, ast.Name):
              out_var = target.id
              src_node, src_port = self._parse_expr(stmt.value, target_hint=out_var)
              if src_node:
                  self.env[out_var] = (src_node, src_port)
  ```
  When `_parse_expr` returns `(None, "out")`, `src_node` is `None`. `self.env[out_var]` is **never populated**. Subsequent statements that consume `out_var` look up `self.env.get(out_var)` and obtain `(None, "out")`, causing all subsequent edges and nodes to be silently omitted. The return statement `_parse_return(value)` also receives `(None, "out")`, producing an output block with 0 input connections.

---

### 1.3 Submodule Port Resolution & Multi-Graph Pipeline
- **Port Discovery in `backend/agent_compiler.py` (Lines 144–201)**:
  ```python
  def _load_custom_module_ports(self, custom_module_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
      ...
      with open(found_path, "r", encoding="utf-8") as f:
          data = json.load(f)
      inputs, outputs = [], []
      for n in data.get("nodes", []):
          b_id = n.get("data", {}).get("block_id")
  ```
  `_load_custom_module_ports` expects `data.get("nodes", [])` to be a **list of nodes** (the `.arch` format). In Agentic IR (`.ir.json`), `nodes` is a **dictionary** (`{"in": {"block": "input"}, "out": {"block": "output"}}`). If only `.ir.json` is present and `.arch` has not yet been compiled, this function fails silently and falls back to default single `in`/`out` ports.

- **Multi-Graph Code Generation in `backend/compiler.py` (Lines 715–775)**:
  `generate_pytorch_code()` topological sorts the multi-graph dictionary `graphs` and compiles each graph into a separate file, emitting clean modular imports:
  ```python
  for dep_id in custom_deps:
      dep_path = file_paths.get(dep_id, dep_id)
      dep_graph = graphs[dep_id]
      dep_class = _to_pascal_case(dep_graph.name)
      clean_mod = dep_path.replace(".arch", "").replace(".json", "").strip("/").replace("/", ".")
      imports.append(f"from {clean_mod} import {dep_class}")
  ```
  The compiler engine already supports modular multi-file architectures; the primary gap is that `python_decompiler.py` only extracts a single class and fails on nested calls.

---

### 1.4 Test Commands and Status
- Ran `pytest backend/tests/`: 42 passed in 2.36s.
- Ran `npx tsc --noEmit`: Exited 0 with 0 errors.

---

## 2. Logic Chain

### 2.1 Multi-Class & Multilevel File Decompilation
1. **From Observation 1.1**: The single `module_class = stmt` assignment in `backend/python_decompiler.py:167` overwrites every previous class. Therefore, any file with $N > 1$ `nn.Module` classes discards $N - 1$ classes.
2. **Class Identification**: An AST node `ast.ClassDef` represents an `nn.Module` if:
   - Any base in `stmt.bases` unparses to contain `"Module"` (`nn.Module`, `torch.nn.Module`, `Module`), OR
   - Its base name matches an already discovered class in the current file or imported local modules, OR
   - It defines a `def forward(self, ...)` method.
3. **Class Dependency Graph**:
   - In a multi-class file, submodules are defined earlier or referenced inside the constructors of other classes.
   - For every class $C_i$, scanning its `__init__` for `Call` nodes instantiating other discovered classes $C_j$ yields a dependency edge $C_i \to C_j$ ($C_i$ depends on $C_j$).
   - Topological sorting of this DAG determines compilation order: leaf submodules (in-degree 0 in dependency graph where no other local class is instantiated) are decompiled first, and consumer/parent classes are decompiled next.
   - Classes with out-degree 0 (no other class instantiates them) are root/entry-point models. If multiple independent models exist, each is a root.
4. **IR Representation of Submodule References**:
   - When parent class $C_i$ instantiates $C_j$ (`self.block = Block(64)`), the parent IR node has:
     - `"block": "custom_module"`
     - `"custom_module_id": "modules/block"` (or `"modules/<stem>_block"`)
     - `"params": {"channels": 64}`
   - The child class $C_j$ produces its own distinct `.ir.json` with `"name": "block"`, its own input/output nodes, and internal operators.
5. **Cross-File and Local Module Resolution**:
   - For `ImportFrom` and `Import`, inspect `alias.asname` or `alias.name`.
   - If relative (`stmt.level > 0`), compute candidate path from `os.path.dirname(current_file)` ascending `stmt.level - 1` directories.
   - If absolute, search candidate locations: `os.path.dirname(current_file)`, `workspace/python/`, `workspace/python/modules/`.
   - When a local `.py` file is located, recursively decompile it (using a `visited_paths: Set[str]` guard to prevent infinite loops on circular imports).
   - Register resolved classes in `self.imports[local_alias] = canonical_module_id`.

---

### 2.2 Deep Object Attribute Chains in Forward Traversal
1. **From Observation 1.2**: `_parse_call` in `backend/python_decompiler.py:472` only handles `call.func` matching `self.<attr>[<slice>]` or `self.<attr>`. Nested chains like `self.backbone.layer1(x)` or `self.features[0].conv(x)` return `(None, "out")`, terminating dataflow.
2. **AST Path Unification**:
   - Any layer call on `self` can be recursively unrolled into a sequence of attribute names and index values:
     ```python
     def _extract_self_chain(node: ast.AST) -> Optional[List[Union[str, int]]]:
         tokens = []
         curr = node
         while True:
             if isinstance(curr, ast.Attribute):
                 tokens.append(curr.attr)
                 curr = curr.value
             elif isinstance(curr, ast.Subscript):
                 slice_val = _eval_ast_literal(curr.slice, set())
                 tokens.append(slice_val)
                 curr = curr.value
             elif isinstance(curr, ast.Name):
                 if curr.id == "self":
                     tokens.reverse()
                     return tokens
                 return None
             else:
                 return None
     ```
   - For `self.backbone.layer1(x)`: `tokens = ["backbone", "layer1"]`.
   - For `self.features[0].conv(x)`: `tokens = ["features", 0, "conv"]`.
   - For `self.blocks[0](x)`: `tokens = ["blocks", 0]`.
3. **Submodule / Layer Resolution**:
   - **Case A: Exact match in `self.layer_instances`**: If key `"backbone.layer1"` exists (e.g. from nested assignment in `__init__`), use its `block` and `params`.
   - **Case B: Container traversal**: If `tokens[0]` is a `ModuleList` or `Sequential`:
     - Index `tokens[1]` selects child element layer from `layer_instances[tokens[0]]["layers"]`.
     - If further tokens exist (`conv`), resolve as submodule member.
   - **Case C: Custom submodule invocation**: If `tokens[0]` is in `self.layer_instances` as `custom_module`:
     - The call is invoking a component of that submodule.
     - Node `block` is `"custom_module"`, with `custom_module_id = f"modules/{tokens[0]}_{tokens[1]}"` or `f"{parent_mod_id}/{tokens[1]}"`.
   - **Case D: General fallback**: If root attribute is not in `self.layer_instances`, treat as dynamic custom block with `custom_module_id = f"modules/{'_'.join(str(t) for t in tokens)}"`.
4. **Dataflow Preservation**:
   - Create node alias: `base_alias = "_".join(str(t) for t in tokens)` $\to$ `node_id = self._next_node_id(base_alias)`.
   - Connect positional arguments: `call.args[i]` $\to$ `node_id.in` / `node_id.in_{i+1}`.
   - Connect keyword arguments: `call.keywords[k]` $\to$ `node_id.<kwarg>`.
   - Return `(node_id, "out")`.
   - `self.env[out_var]` receives `(node_id, "out")`, downstream nodes connect properly, and the graph remains fully connected and valid.

---

## 3. Caveats

1. **Non-Callable Attribute Reads**:
   Expressions like `x = x + self.fc.bias` or `dim = self.backbone.out_channels` are attribute loads (`ast.Attribute` without `ast.Call`). They represent parameter extraction or scalar access rather than layer calls. The decompiler should evaluate them as scalars or constants via `_eval_ast_literal`.
2. **Dynamic / Imperative Layer Instantiation**:
   PyTorch code using dynamic metaprogramming like `setattr(self, f"layer_{i}", layer)` or loops in `__init__` that cannot be statically unrolled will require AST interpretation of loop bounds in `__init__`.
3. **Third-Party External Packages**:
   Imports from packages outside the workspace (e.g. `torchvision.models`, `timm.models`) do not have local `.py` source files. They must be mapped to composite `custom_module` placeholders with inferred ports rather than triggering file-not-found errors during recursive file search.
4. **Interaction with Requirement R3**:
   `nn.ModuleList` for-loop unrolling (`ast.For`) and structural conditionals (`if self.downsample is not None:`) are scoped under Requirement R3. The attribute chain resolver designed here cleanly provides the prerequisite primitive: calling indexed module items (`self.layers[i](x)`) or conditional downsample branches (`self.downsample(x)`) will now resolve without dropping dataflow.

---

## 4. Conclusion & Implementation Plan

### 4.1 Conclusion
The ArchIDE decompiler can fully support multi-class files, recursive import resolution, and arbitrary deep attribute chains by:
1. Refactoring class discovery from a single variable assignment into a multi-class discovery and topological dependency sorter.
2. Generating distinct `.ir.json` files for each `nn.Module` class, linking them via `"block": "custom_module"` and `"custom_module_id": "modules/<submodule_name>"`.
3. Replacing the restricted 1A/1B checks in `_parse_call` with a generalized AST self-path extractor (`_extract_self_chain`) that creates valid IR nodes and wires input/output edges for any nested layer call.
4. Enhancing `agent_compiler.py:_load_custom_module_ports` to inspect `.ir.json` files directly (supporting dictionary-based `nodes`).

### 4.2 Step-by-Step Implementation Steps

#### Step 1: Deep Object Attribute Chain Resolver (`backend/python_decompiler.py`)
- Implement `_extract_self_chain(node: ast.AST) -> Optional[List[Union[str, int]]]`.
- In `_parse_call`, replace lines 472–542 with a unified handler:
  ```python
  chain = self._extract_self_chain(call.func)
  if chain:
      # 1. Resolve layer definition or submodule ID from chain tokens
      # 2. Assign node_id via self._next_node_id("_".join(str(t) for t in chain))
      # 3. Create node_entry in self.nodes
      # 4. Wire positional & keyword arguments to input ports
      # 5. Return (node_id, "out")
  ```
- Enhance `_parse_layer_instantiation` for `Sequential` to store child layer definitions in `self.layer_instances[attr]["layers"]`.

#### Step 2: Multi-Class Discovery & Dependency Extraction (`backend/python_decompiler.py`)
- In `decompile_source`:
  - Identify all `ast.ClassDef` nodes in `tree.body` inheriting from `Module` or defining `forward`.
  - For each class, inspect constructor assignments to identify dependencies on other local classes.
  - Build dependency graph and compute topological order.
  - Decompile each class using an isolated context (or fresh decompiler instance) to produce an IR dict per class.
  - Return the root model IR dict by default (preserving backward compatibility), while storing all generated IRs in `self.all_irs: Dict[str, Dict[str, Any]]`.
  - Add `decompile_all_classes(code_str: str, file_stem: str = "main") -> Dict[str, Dict[str, Any]]`.

#### Step 3: Local & Cross-File Recursive Import Resolution (`backend/python_decompiler.py`)
- In `_parse_imports`:
  - Support `alias.asname` for aliased imports (`from x import Y as Z`).
  - For non-standard imports, search relative to `workspace_dir`, `workspace/python/`, or directory of the source file.
  - When a target local `.py` file is identified, recursively decompile it if its `.ir.json` does not yet exist (with a recursion guard `visited_files`).

#### Step 4: Multi-Class Output File Management (`backend/python_decompiler.py` & `backend/agent_compiler.py`)
- In `decompile_python_to_ir`:
  - If multiple classes are extracted:
    - Root class written to `output_path` (e.g. `workspace/ir/resnet.ir.json`).
    - Submodule classes written to `workspace/ir/modules/<class_name>.ir.json`.
- In `backend/agent_compiler.py`:
  - Update `_load_custom_module_ports` to support `.ir.json` files where `nodes` is a dictionary (`"block": "input"` / `"output"`).

---

## 5. Verification Method

### 5.1 Independent Test Verification
1. **Unit Test Suite**:
   ```bash
   pytest backend/tests/test_python_decompiler.py
   pytest backend/tests/
   ```
2. **TypeScript / Frontend Check**:
   ```bash
   npx tsc --noEmit
   ```

### 5.2 Specific Test Cases to Construct
1. **Multi-Class Single File Test (`test_decompile_multi_class_single_file`)**:
   - Provide Python code containing `class BasicBlock(nn.Module)` and `class ResNet(nn.Module)` in the same file.
   - Decompile and verify that:
     - `BasicBlock` is extracted into its own IR with `name: "basicblock"`.
     - `ResNet` IR contains a node with `"block": "custom_module"` and `"custom_module_id": "modules/basicblock"`.
     - Both IRs compile cleanly with `compile_ir(ir, validate=True)`.
2. **Deep Attribute Chain Test (`test_decompile_deep_attribute_chains`)**:
   - Provide Python code calling `self.backbone.layer1(x)`, `self.features[0].conv(x)`, and `self.transformer.encoder.layers[0].norm(x)`.
   - Verify that:
     - Nodes `backbone_layer1`, `features_0_conv`, `transformer_encoder_layers_0_norm` exist in `ir["nodes"]`.
     - Edges connect input variables to these nodes and output variables to subsequent layers.
     - `ir["edges"]` form an unbroken directed path from `in` to `out`.
3. **Cross-File Recursive Decompilation Test (`test_decompile_cross_file_imports`)**:
   - Create two temporary `.py` files where `model_a.py` imports `ModelB` from `model_b.py`.
   - Decompile `model_a.py` and verify `model_b.ir.json` is generated automatically.

### 5.3 Invalidation Conditions
- If any node in a deep attribute chain has 0 incoming edges while consuming a forward argument.
- If `decompile_source` drops earlier classes in a multi-class file.
- If recursive import resolution causes recursion depth errors on circular imports.
