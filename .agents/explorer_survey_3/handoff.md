# Architectural Survey & Handoff Report: R3 (AST Control Flow & Dynamic Shape Extraction) and R4 (Real-World Multi-Repo Benchmark & Roundtrip Equivalence)

## Executive Summary
This survey provides a comprehensive architectural and code-level investigation of ArchIDE for **Requirement R3** (AST Control Flow & Dynamic Shape Extraction) and **Requirement R4** (Real-World Multi-Repo Benchmark & Roundtrip Equivalence). It analyzes the existing decompiler (`backend/python_decompiler.py`), compiler pipeline (`backend/compiler.py`, `backend/agent_compiler.py`), block registry (`backend/blocks/`), and test suite (`backend/tests/`). We identified critical blockers—including node ID collisions with intermediate `"out"` variables causing cyclic graphs, broken `ast.For` loop variable bindings, absence of `ast.If` structural conditional branching, character stripping in `ReshapeBlock`, and single-argument truncation in `reshape`/`view` calls—and designed the exact implementation and benchmark test suite (`backend/tests/test_real_world_models.py`) with `torch.allclose` numerical verification (`atol=1e-4`).

---

## 1. Observation

### 1.1 Decompiler Architecture & Control Flow (`backend/python_decompiler.py`)
- **Node ID Allocation & Collision (`python_decompiler.py:139-146`)**:
  ```python
  def _next_node_id(self, base_name: str) -> str:
      clean = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).strip('_').lower() or "node"
      count = self.node_counters.get(clean, 0) + 1
      self.node_counters[clean] = count
      if count == 1 and clean in {"in", "out", "input", "output", "flatten"}:
          return clean
      return f"{clean}_{count}" if count > 1 else clean
  ```
  When intermediate variables in models are named `out` (standard in PyTorch, e.g. `out = self.conv(x); out = out + identity; return out`), `_next_node_id("out")` returns `"out"`. Later in `_parse_return` (`python_decompiler.py:625-626`):
  ```python
  out_node_id = "out"
  self.nodes[out_node_id] = {"block": "output"}
  ```
  The terminal output block overwrites the intermediate node `"out"`, and creates a self-loop edge `out.out -> ... -> out.in`. When passed to `topological_sort` (`backend/compiler.py:89`), compilation crashes with:
  `ValueError: Cycle detected in graph! Cannot compile.`

- **`ast.For` Loop Unrolling Defect (`python_decompiler.py:406-424`)**:
  ```python
  elif isinstance(stmt, ast.For):
      if isinstance(stmt.iter, ast.Attribute) and isinstance(stmt.iter.value, ast.Name) and stmt.iter.value.id == "self":
          mod_list = self.layer_instances.get(stmt.iter.attr)
          if mod_list and mod_list.get("block") == "module_list":
              for sub_layer in mod_list.get("layers", []):
                  node_id = self._next_node_id(sub_layer.get("custom_module_id", "module").split("/")[-1])
                  self.nodes[node_id] = {
                      "block": "custom_module",
                      "custom_module_id": sub_layer.get("custom_module_id"),
                  }
                  if isinstance(stmt.target, ast.Name) and stmt.body:
                      loop_var = stmt.target.id
                      prev_src, prev_port = self.env.get(loop_var, ("in", "out"))
                      self.edges.append(f"{prev_src}.{prev_port} -> {node_id}.in")
                      self.env[loop_var] = (node_id, "out")
  ```
  - **Defect 1**: `loop_var = stmt.target.id` treats the loop iterator variable (`block` or `layer`) as the tensor variable! In standard PyTorch `for blk in self.blocks: x = blk(x)`, `stmt.target.id` is `"blk"`, NOT `"x"`. `self.env["blk"]` is mutated, while `self.env["x"]` remains untouched.
  - **Defect 2**: `stmt.body` statements are NEVER parsed or executed.
  - **Defect 3**: In `_parse_call` (`python_decompiler.py:472-544`), calls where `call.func` is `ast.Name` (e.g. `blk(x)` or `layer(x)`) are unhandled and return `(None, "out")`.
  - **Defect 4**: In `_parse_layer_instantiation` (`python_decompiler.py:289-303`), `nn.ModuleList` initialized via list comprehension `nn.ModuleList([Block(...) for _ in range(depth)])` is ignored because it only checks `isinstance(call.args[0], (ast.List, ast.Tuple))`.

- **Missing `ast.If` Structural Conditional Branching**:
  - In `_parse_init` (`python_decompiler.py:237-244`): Only top-level `ast.Assign` with `ast.Call` values are parsed. Attributes assigned to `None`, boolean flags (e.g., `self.downsample = None`, `self.use_residual = True`), or conditional assignments (`if stride != 1: self.downsample = ...`) are completely ignored.
  - In `_parse_statements` (`python_decompiler.py:380-424`): There is NO handler for `ast.If`. Structural conditionals such as `if self.downsample is not None:` or `if self.use_residual:` are completely ignored and never evaluated.

### 1.2 Dynamic Shape Extraction & `ShapeExtractorBlock`
- **Block Existence (`backend/blocks/core.py:305-353`)**:
  `ShapeExtractorBlock` already exists in `backend/blocks/core.py` and is exported in `backend/blocks/__init__.py`.
  ```python
  class ShapeExtractorBlock(BaseBlock):
      id="shape_extractor", name="Shape Extractor", category="Core Layers", is_functional=True
      inputs=[PortDef(id="in", name="Input")]
      outputs=[
          PortDef(id="shape", name="Shape", var_hint="shape"),
          PortDef(id="dim_0", name="Dim 0 (B)", var_hint="b"),
          PortDef(id="dim_1", name="Dim 1 (C)", var_hint="c"),
          PortDef(id="dim_2", name="Dim 2 (H)", var_hint="h"),
          PortDef(id="dim_3", name="Dim 3 (W)", var_hint="w"),
      ]
  ```
- **Tuple Unpacking Defect in Decompiler (`python_decompiler.py:388-395`)**:
  ```python
  elif isinstance(target, (ast.Tuple, ast.List)):
      src_node, src_port = self._parse_expr(stmt.value)
      if src_node: ...
  ```
  When `stmt.value` is `x.shape` (`ast.Attribute`) or `x.size()` (`ast.Call`), `_parse_expr` does not recognize them and returns `(None, "out")`. The target tuple `B, C, H, W` or `b, n, c` receives no mapping in `self.env`.
- **Character Stripping Defect in `ReshapeBlock` (`backend/blocks/shape.py:99-101`)**:
  ```python
  shape_str = params.get("shape", "-1")
  clean = "".join(c for c in str(shape_str) if c.isdigit() or c == ',' or c == '-')
  return f"{out_var} = {in_var}.reshape({clean})"
  ```
  This filter deletes all letters. If `shape_str` contains variable names like `(B, N, -1)` or `(b, c, -1)`, it emits `reshape(,-1)` which is a Python syntax error.
- **Argument Truncation in `reshape` / `view` Functional Calls (`python_decompiler.py:600-610`)**:
  In `FUNCTIONAL_MAP`: `"view": ("reshape", ["shape"], {})`.
  Because `pos_params` has only 1 element (`"shape"`), in calls with multiple positional arguments like `x.view(b, n, -1)`, only `b` is assigned to `params["shape"]`. The remaining arguments `n` and `-1` are discarded!

### 1.3 Roundtrip Compilation & Existing Test Baseline
- **Current Test Status**: `pytest backend/tests/` passes 42/42 tests in 2.97s.
- **Frontend TypeScript Status**: `npx tsc --noEmit` exits with 0 errors.
- **Environment Status**: PyTorch 2.11.0+cpu, `torchvision` (installed), `transformers` (installed).
- **Execution Equivalence Prototype**: Verified that on a clean DAG (without the `"out"` identifier collision), transferring weights (`p_gen.copy_(p_orig)`) and buffers (`b_gen.copy_(b_orig)`) between the original module and recompiled module achieves `max_diff = 0.0` on dummy inputs.

---

## 2. Logic Chain

### 2.1 Resolution of Control Flow & ModuleList Unrolling
```
Observation: PyTorch models define self.blocks = nn.ModuleList([Block(...) for _ in range(depth)])
             and forward() loops: for block in self.blocks: x = block(x)
Observation: _parse_layer_instantiation skips ast.ListComp; _parse_for binds loop_var to target.id
             and does not parse stmt.body; _parse_call drops ast.Name calls.
Logic Step 1: Detect ast.ListComp in _parse_layer_instantiation for nn.ModuleList:
              extract element Call, evaluate generator range count (resolving init_param default if variable).
Logic Step 2: In _parse_statements for ast.For:
              extract iterator target name (e.g. 'block' or 'layer').
              For each sublayer index i in the module list:
                create or bind an active local layer instance for iterator target name.
                execute self._parse_statements(stmt.body) within the loop iteration.
Logic Step 3: In _parse_call, add handler for ast.Name(id=iter_name):
              instantiate sublayer node for iteration i, link input from self.env, update self.env[target_var].
Conclusion: Tensor mutations across iterations chain deterministically through self.env,
            producing an unrolled DAG that compiles to execution-equivalent PyTorch code.
```

### 2.2 Resolution of Structural Conditionals (`ast.If`)
```
Observation: ResNet BasicBlock uses:
             if self.downsample is not None: identity = self.downsample(x)
             or models use: if self.use_residual: x = x + residual
Observation: _parse_statements completely ignores ast.If, and _parse_init ignores non-Call assigns.
Logic Step 1: In _parse_init, record all literal/default attribute assignments in self.attributes
              (e.g., self.downsample = None, self.use_residual = True).
Logic Step 2: In _parse_statements, add handler for ast.If:
              evaluate stmt.test:
              - 'self.attr is not None': True if attr in self.layer_instances and attr not None, else False.
              - 'self.attr is None': inverse.
              - 'self.attr': bool(self.attributes.get(attr, False)).
              - 'not self.attr': inverse.
Logic Step 3: Parse stmt.body if condition evaluates to True; parse stmt.orelse if False.
Conclusion: Compile-time structural branches resolve to the exact active graph topology without dead code.
```

### 2.3 Resolution of Dynamic Shape Extraction
```
Observation: Real-world models use B, C, H, W = x.shape or b, n, c = x.size()
             followed by x = x.reshape(B, N, -1) or x.view(b, c, -1).
Observation: ShapeExtractorBlock exists in backend/blocks/core.py with outputs:
             shape, dim_0, dim_1, dim_2, dim_3.
Observation: ReshapeBlock strips non-digits; _parse_call drops arguments 1..N for view/reshape.
Logic Step 1: In _parse_statements, when stmt.value is x.shape or x.size():
              create a shape_extractor node, link x -> shape_extractor.in.
              For each target variable target.elts[idx]:
              set self.env[elt.id] = (shape_node_id, f"dim_{idx}").
              Set _output_aliases on shape_extractor to preserve original variable names (B, C, H, W).
Logic Step 2: In python_decompiler.py FUNCTIONAL_MAP handling:
              when call.func is reshape or view with len(call.args) > 1:
              pack all args into a tuple params["shape"] = tuple(eval(a) for a in call.args).
Logic Step 3: In ReshapeBlock.emit_forward:
              preserve variable names and identifiers in shape string instead of stripping letters.
Conclusion: Shape tuple unpacking routes seamlessly into downstream Reshape/Flatten operations.
```

### 2.4 Resolution of Intermediate Variable `"out"` Collision
```
Observation: Naming an intermediate variable 'out' causes _next_node_id to return 'out',
             which collides with terminal output block 'out', creating a cyclic graph.
Logic Step: Reserve 'in' and 'out' exclusively for graph boundary ports.
            For any intermediate variable whose clean name is 'in' or 'out',
            always append a counter suffix (e.g., out_1, out_2).
Conclusion: Topological sort succeeds on all standard PyTorch code utilizing 'out' as accumulator.
```

---

## 3. Caveats

1. **Static vs Dynamic Branching**:
   ArchIDE compiles static computation graphs (`.arch` / `.ir.json`). Conditional statements (`ast.If`) are evaluated statically based on module initialization state (`self.layer_instances`, `self.attributes`, default hyperparameters). Dynamic data-dependent conditionals (e.g. `if x.sum() > 0:`) cannot be represented as a static DAG and will raise a descriptive compilation notice.
2. **Stochastic Regularization in Numerical Verification**:
   When verifying execution equivalence with `torch.allclose`, models containing stochastic operations (`nn.Dropout`, `StochasticDepth`) must be evaluated in `eval()` mode (`model.eval()`), which disables stochastic dropping and ensures exact mathematical determinism.
3. **Missing Blocks Dependency (R1)**:
   Models using `GELU` (e.g. ConvNeXt, ViT, Transformer MLP) or `Conv1d` depend on R1 implementation (`GELUBlock`, `SiLUBlock`, `Conv1DBlock`, `EmbeddingBlock`). In the current baseline, `compiler.py` skips unregistered blocks (`# WARNING: unknown block 'gelu' — skipped`). R1 must be in place or co-tested for full end-to-end numerical verification of GELU-based models.

---

## 4. Conclusion & Recommended Implementation Plan

### 4.1 Changes to `backend/python_decompiler.py`
1. **Fix `_next_node_id`**: Ensure `"out"` and `"in"` are never assigned to intermediate variable nodes.
2. **Implement `_parse_init` attribute tracking**: Store non-module attributes and defaults in `self.attributes`.
3. **Implement `ast.If` structural evaluation in `_parse_statements`**: Evaluate `self.attr is not None`, `self.attr`, etc., branching into `stmt.body` or `stmt.orelse`.
4. **Implement `ast.For` loop unrolling**:
   - Parse `ast.ListComp` in `_parse_layer_instantiation` for `nn.ModuleList`.
   - In `_parse_statements`, loop over sublayers, bind iterator variable, recursively execute `stmt.body`.
   - In `_parse_call`, support calls on the iterator variable name.
5. **Implement dynamic shape unpacking**:
   - Detect `x.shape` and `x.size()` assigns with `ast.Tuple`/`ast.List` targets.
   - Instantiate `ShapeExtractorBlock`, route ports `dim_0`..`dim_k`, map in `self.env`.
   - Pack varargs in `x.view(*args)` / `x.reshape(*args)` into a shape tuple.
6. **Support multi-class files (R2 synergy)**:
   - Traverse all `ast.ClassDef` in `tree.body` inheriting from `nn.Module`.
   - Return dictionary of IR graphs: `{class_name.lower(): ir_dict}`.

### 4.2 Changes to `backend/blocks/shape.py` & `backend/blocks/core.py`
1. **`ReshapeBlock.emit_forward`**: Strip only surrounding parentheses/spaces; preserve variable identifiers and expressions (e.g. `(b, c, -1)` or `(-1, 512)`).
2. **`ShapeExtractorBlock`**: Extend outputs to support `dim_0` through `dim_4` and ensure `_output_aliases` generates clean, direct variable assignments.

### 4.3 Benchmark Suite Structure (`backend/tests/test_real_world_models.py`)
Implement the following canonical tests verifying `torch.allclose(out_orig, out_gen, atol=1e-4)`:

| Test Case | Repository Origin | Key Structural Features Tested |
|---|---|---|
| `test_resnet_basicblock_with_downsample` | `torchvision.models.resnet` | `ast.If` (`if self.downsample is not None:`), residual addition, intermediate `"out"` variable |
| `test_resnet_basicblock_no_downsample` | `torchvision.models.resnet` | `ast.If` evaluating to False branch (`self.downsample is None`), residual addition |
| `test_convnext_block` | `torchvision.models.convnext` | Depthwise Conv2d (7x7), LayerNorm, GELU, 1x1 Linear/Conv, residual connection |
| `test_transformer_mlp` | `timm` / `transformers` | Linear -> GELU -> Linear, init param binding |
| `test_multihead_attention_shape_extract` | `timm` / HuggingFace | `B, N, C = x.shape` tuple unpacking, `ShapeExtractorBlock`, Q/K/V projections, scaled dot-product, matmul, transpose |
| `test_vit_block_module_list_unroll` | `timm` | `nn.ModuleList([Block(...) for _ in range(3)])`, `for blk in self.blocks: x = blk(x)` unrolling and tensor mutation chaining |
| `test_unet_double_conv_skip_cat` | Canonical UNet | Conv2d -> BatchNorm2d -> ReLU (x2), skip concatenation `torch.cat([skip, x], dim=1)` via `CatBlock` |
| `test_multi_class_file_roundtrip` | Multi-class pattern | Multi-class file (`DoubleConv` + `UNet` or `Mlp` + `TransformerBlock`), cross-module reference resolution |

---

## 5. Verification Method

To verify the implementation independently:

1. **Run Full Test Suite**:
   ```pwsh
   pytest backend/tests/
   pytest backend/tests/test_real_world_models.py -v
   ```
   **Expected**: 100% pass across all existing 42 tests and new real-world benchmark tests.

2. **Verify TypeScript Types & Contracts**:
   ```pwsh
   npx tsc --noEmit
   ```
   **Expected**: 0 errors.

3. **Verify Numerical Tolerance**:
   Every test in `test_real_world_models.py` must assert:
   ```python
   assert torch.allclose(y_orig, y_gen, atol=1e-4, rtol=1e-4)
   ```
   Ensuring mathematical output equivalence on dummy inputs.

4. **Invalidation Conditions**:
   - Any test failing with `Cycle detected in graph! Cannot compile.` indicates residual `"out"` node ID collision.
   - Any test failing with `NameError` or missing variable indicates failure in `ast.For` environment chaining or `ShapeExtractorBlock` alias mapping.
   - Any shape mismatch indicates improper parameter preservation in `ReshapeBlock` or `ModuleList`.
