# Handoff Report — Explorer 1: R1 Core Block Extensions & Scalar Binary Operations

**Task Reference**: Requirement R1 from `d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md`  
**Working Directory**: `d:\ML\ArchIDE\.agents\explorer_survey_1`  
**Date**: 2026-09-20  
**Target Component**: ArchIDE Block Registry, Compiler Code Generation, PyTorch AST Decompiler

---

## 1. Observation

### 1.1 Block Registry Architecture & Registration
- `BaseBlock` is defined in `backend/blocks/base.py:31-59`. All blocks inherit from `BaseBlock` and implement:
  - `@property def definition(self) -> BlockDef` (Pydantic model from `backend/models.py:100-109`).
  - `def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]`
  - `def emit_init(self, node_id: str, params: Dict[str, Any]) -> str`
  - `def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str`
- Centralized registration is located in `backend/blocks/__init__.py:16-53`:
  - `_BLOCK_INSTANCES` (lines 17-48) instantiates 30 blocks.
  - `_BLOCK_MAP` (lines 51-53) indexes blocks by `block.definition.id`.
  - `get_all_blocks()` and `get_all_block_defs()` expose the instances.
- Fast-API registry endpoint in `backend/main.py:34-36`:
  ```python
  @app.get("/api/blocks", response_model=List[BlockDef])
  def get_blocks():
      return get_all_block_defs()
  ```
- Offline schema tool `backend/dump_block_schema.py:36-41` iterates over `get_all_blocks()` and writes to `backend/block_schema.json` (currently contains 30 block schemas).
- Frontend dynamically queries `/api/blocks` on mount in `src/components/BlockLibrary.tsx:48-55` and `src/components/QuickInsertModal.tsx:47-54`. Fallback entries reside in `src/lib/constants.ts:23-33` (`FALLBACK_BLOCKS`).

### 1.2 Missing Blocks (`GELU`, `SiLU`, `Conv1d`, `Embedding`)
- `GELU`:
  - Missing in `backend/blocks/activations.py`.
  - Present in `backend/python_decompiler.py:20`: `"GELU": ("gelu", ["approximate"], {"approximate": "none"})`.
  - But `get_block_by_id("gelu")` returns `None`.
- `SiLU`:
  - Missing in `backend/blocks/activations.py`.
  - Misconfigured in `backend/python_decompiler.py:21`: `"SiLU": ("gelu", [], {})` and line 56: `"silu": ("gelu", [], {})` (aliased to `"gelu"` because `"silu"` didn't exist).
- `Conv1D`:
  - Missing in `backend/blocks/core.py`.
  - Present in `backend/python_decompiler.py:16`: `"Conv1d": ("conv1d", ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"], {...})`.
  - But `get_block_by_id("conv1d")` returns `None`.
- `Embedding`:
  - Missing in `backend/blocks/core.py` and `backend/blocks/`.
  - Completely absent from `backend/python_decompiler.py` `LAYER_MAP`.

### 1.3 Scalar Binary Operations Defect in `backend/blocks/tensor_ops.py`
In `backend/python_decompiler.py:430-464`, decompiling an `ast.BinOp` (e.g. `x + 1` or `x / math.sqrt(d)`):
```python
            if src_a:
                self.edges.append(f"{src_a}.{port_a} -> {node_id}.{in_port_a}")
            else:
                scalar_val = _eval_ast_literal(expr.left, self.init_var_names)
                self.nodes[node_id].setdefault("params", {})["scalar_a"] = scalar_val

            if src_b:
                self.edges.append(f"{src_b}.{port_b} -> {node_id}.{in_port_b}")
            else:
                scalar_val = _eval_ast_literal(expr.right, self.init_var_names)
                self.nodes[node_id].setdefault("params", {})["scalar_b"] = scalar_val
```
- In `backend/agent_compiler.py:239`, `node_info.get("params", {})` is preserved into `node["data"]["paramValues"]`.
- In `backend/compiler.py:874`, `params = dict(node.data.paramValues)`.
- In `backend/compiler.py:900-901`, variable substitution replaces `@var:name` with `self.name` inside `params`.
- **The Failure Point**:
  1. `AddBlock.emit_forward` (`backend/blocks/tensor_ops.py:64-76`):
     ```python
     def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
         out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
         src_vars = []
         for v in input_vars.values():
             if isinstance(v, list):
                 src_vars.extend([x for x in v if x and x != "None"])
             elif v and v != "None":
                 src_vars.append(v)
         if src_vars:
             add_expr = " + ".join(src_vars)
             return f"{out_var} = {add_expr}"
         return f"{out_var} = 0"
     ```
     `params.get("scalar_a")` and `params.get("scalar_b")` are **completely ignored**. For `x + 1`, `input_vars` has `["x"]`, so it emits `out = x`. The scalar `+ 1` is silently dropped.
  2. `MulBlock.emit_forward` (`backend/blocks/tensor_ops.py:140-152`):
     Same bug: only joins `input_vars.values()`. For `x * 2`, emits `out = x`.
  3. `SubBlock.emit_forward` (`backend/blocks/tensor_ops.py:100-105`):
     ```python
     def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
         out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
         a = input_vars.get("in_a", "None")
         b = input_vars.get("in_b", "None")
         return f"{out_var} = {a} - {b}"
     ```
     When `x - 1` is decompiled, `b` has no matching edge. In `compiler.py:941`, `input_vars["in_b"] = "None"`. `SubBlock.emit_forward` generates `out = x - None`.
  4. `DivBlock.emit_forward` (`backend/blocks/tensor_ops.py:176-181`):
     ```python
     def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
         out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
         a = input_vars.get("in_a", "None")
         b = input_vars.get("in_b", "None")
         return f"{out_var} = {a} / {b}"
     ```
     For `x / math.sqrt(d)`, `b` has no matching edge and is `"None"`, emitting `out = x / None`.
  5. Module Imports in `backend/compiler.py:744-747`:
     Currently only emits:
     ```python
     imports = [
         "import torch",
         "import torch.nn as nn",
     ]
     ```
     `import math` is missing from the header, causing `NameError: name 'math' is not defined` when executing expressions like `math.sqrt(d)`.

---

## 2. Logic Chain

1. **Missing Blocks Chain**:
   - `python_decompiler.py` already recognizes `GELU` (mapped to `"gelu"`), `Conv1d` (mapped to `"conv1d"`), and `SiLU` (mistakenly mapped to `"gelu"`).
   - When models with these layers are decompiled, the IR contains `{"block": "gelu"}`, `{"block": "conv1d"}`, etc.
   - When the compiler or visual IDE tries to resolve these block IDs via `get_block_by_id`, `None` is returned, causing compiler failures or unknown-block warnings.
   - Implementing `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, and `EmbeddingBlock` with their proper `BlockDef`, `infer_shapes`, `emit_init`, and `emit_forward` methods, and registering them in `_BLOCK_INSTANCES`, closes this gap completely.

2. **Scalar Binary Ops Chain**:
   - `python_decompiler.py:455-462` already places scalar operands into `node["params"]["scalar_a"]` and `node["params"]["scalar_b"]`.
   - `agent_compiler.py` faithfully passes `node["params"]` into `node["data"]["paramValues"]`.
   - `compiler.py:874` reads `node.data.paramValues`, substitutes variables like `@var:d_model` into `self.d_model`, and passes the resulting `params` dict to `block.emit_forward(node.id, input_vars, output_vars, params)`.
   - Therefore, the data pipeline is already fully intact up to the block's `emit_forward` call! The sole breakdown is that `AddBlock`, `SubBlock`, `MulBlock`, and `DivBlock` do not read `params.get("scalar_a")` or `params.get("scalar_b")`.
   - Updating `emit_forward` across `AddBlock`, `SubBlock`, `MulBlock`, and `DivBlock` to check and emit `scalar_a` and `scalar_b` restores the exact mathematical operations (`x + 1`, `x - 1`, `1 - x`, `x * 2`, `x / math.sqrt(self.d_model)`).
   - Adding `"import math"` to `compiler.py` ensures modules with `math.sqrt` compile and execute cleanly in Python runtime.

---

## 3. Caveats

1. **Binary Operator Precedence**: When a scalar parameter is an expression string containing low-precedence operators (e.g. `d + 1` in `x / (d + 1)`), parenthesization may be required if not already parenthesized. However, standard scalar literals and calls (`1`, `2.0`, `self.d_model`, `math.sqrt(self.d_model)`) have higher precedence than `/` or `*` or are atomic. A helper to wrap expressions with spaces/binary operators in parentheses `(...)` ensures safe division.
2. **Frontend Fallback Sync**: While the frontend dynamically queries `/api/blocks`, `FALLBACK_BLOCKS` in `src/lib/constants.ts` provides offline resilience. Updating `FALLBACK_BLOCKS` is non-breaking and recommended for consistency.
3. **Shape Inference for Pure Scalars**: When both operands are scalar (e.g. `1 + 2`), `broadcast_shapes` and `infer_shapes` will receive empty or `("ANY",)` inputs, yielding `("ANY",)`. This is standard for dynamically evaluated scalars.

---

## 4. Conclusion & Recommended Implementation Steps

### 4.1 File Modification Map
| File | Changes Required |
|---|---|
| `backend/blocks/activations.py` | Add `GELUBlock` (approximate='none'\|'tanh') and `SiLUBlock` (inplace=False\|True). |
| `backend/blocks/core.py` | Add `Conv1DBlock` (with 1D shape math, LAZY support, error guards) and `EmbeddingBlock` (with `(*, embedding_dim)` shape inference). |
| `backend/blocks/tensor_ops.py` | Update `AddBlock`, `SubBlock`, `MulBlock`, `DivBlock`: declare `scalar_a`, `scalar_b` in `BlockDef.params`, and preserve them in `emit_forward`. |
| `backend/blocks/__init__.py` | Import and add `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, `EmbeddingBlock` to `_BLOCK_INSTANCES`. |
| `backend/compiler.py` | Add `"import math"` to `imports` in `generate_pytorch_code` (line 745); add `"conv1d"`, `"embedding"`, `"gelu"`, `"silu"` to layer naming tuples (line 864, 885). |
| `backend/python_decompiler.py` | Add `Embedding` to `LAYER_MAP`; point `SiLU` to `"silu"` in `LAYER_MAP` and `FUNCTIONAL_MAP`; add lookahead shape inference for `conv1d` and `embedding`. |
| `backend/dump_block_schema.py` / `block_schema.json` | Run `python backend/dump_block_schema.py` to regenerate `block_schema.json` (34 blocks total). |
| `src/lib/constants.ts` (optional) | Add `gelu`, `silu`, `conv1d`, `embedding` to `FALLBACK_BLOCKS`. |

### 4.2 Detailed Implementation Code Snippets

#### A. `GELUBlock` & `SiLUBlock` in `backend/blocks/activations.py`
```python
class GELUBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="gelu",
            name="GELU",
            category="Activations",
            color="#f59e0b",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="activated")],
            params=[
                ParamDef(name="approximate", type="string", default="none", section="advanced", description="GELU approximation algorithm: 'none' or 'tanh'")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        approx = params.get("approximate", "none")
        if str(approx).strip("'\"").lower() == "tanh":
            return f"{layer_name} = nn.GELU(approximate='tanh')"
        return f"{layer_name} = nn.GELU(approximate='none')"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "Gaussian Error Linear Units activation function.",
            "details": "### nn.GELU\nApplies the Gaussian Error Linear Units function with optional tanh approximation."
        }


class SiLUBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="silu",
            name="SiLU",
            category="Activations",
            color="#f59e0b",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="activated")],
            params=[
                ParamDef(name="inplace", type="bool", default=False, section="advanced", description="Modify input in-place")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        inplace = params.get("inplace", False)
        return f"{layer_name} = nn.SiLU(inplace={inplace})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "Sigmoid Linear Unit (SiLU / Swish) activation function: x * sigmoid(x).",
            "details": "### nn.SiLU\nApplies the Sigmoid Linear Unit function: `x * sigmoid(x)`."
        }
```

#### B. `Conv1DBlock` & `EmbeddingBlock` in `backend/blocks/core.py`
```python
def parse_int_1d(val: Any, default: int) -> int:
    if val is None or val == "" or val == "None":
        return default
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return int(val[0])
    if isinstance(val, str):
        clean = "".join(c for c in val if c.isdigit() or c == '-')
        if clean:
            return int(clean)
    return default


class Conv1DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="conv1d",
            name="Conv1D",
            category="Core Layers",
            color="#3b82f6",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="conv1d_feat")],
            params=[
                ParamDef(name="in_channels",  type="int", default=3,  auto_infer=True, section="basic", description="Number of channels in the input"),
                ParamDef(name="out_channels", type="int", default=16, section="basic", description="Number of channels produced by the convolution"),
                ParamDef(name="kernel_size",  type="int", default=3,  section="basic", description="Size of the convolving kernel"),
                ParamDef(name="stride",       type="int", default=1,  section="advanced", description="Stride of the convolution"),
                ParamDef(name="padding",      type="int", default=0,  section="advanced", description="Padding added to both sides of the input"),
                ParamDef(name="dilation",     type="int", default=1,  section="advanced", description="Spacing between kernel elements"),
                ParamDef(name="groups",       type="int", default=1,  section="advanced", description="Number of blocked connections from input to output"),
                ParamDef(name="bias",         type="bool", default=True, section="advanced", description="If True, adds a learnable bias to the output"),
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or in_shape == ("ANY",):
            return {"out": ("ANY",)}

        if len(in_shape) != 3:
            raise ValueError(
                f"Conv1D expects a 3D tensor (Batch, Channels, Length), "
                f"but received {len(in_shape)}D tensor with shape {in_shape}."
            )

        B, C, L = in_shape
        in_channels = params.get("in_channels", 3)

        if in_channels == -1 and C != "ANY":
            in_channels = C
            params["in_channels"] = in_channels

        if in_channels != "LAZY" and C != "ANY" and C != in_channels:
            raise ValueError(
                f"Conv1D: expected in_channels={in_channels}, "
                f"but input has {C} channels."
            )

        k = parse_int_1d(params.get("kernel_size"), default=3)
        st = parse_int_1d(params.get("stride"), default=1)
        pad = parse_int_1d(params.get("padding"), default=0)
        dil = parse_int_1d(params.get("dilation"), default=1)

        if st <= 0:
            raise ValueError(f"Conv1D: stride must be greater than 0, got {params.get('stride')}")
        if k <= 0:
            raise ValueError(f"Conv1D: kernel_size must be greater than 0, got {params.get('kernel_size')}")
        if dil <= 0:
            raise ValueError(f"Conv1D: dilation must be greater than 0, got {params.get('dilation')}")
        if pad < 0:
            raise ValueError(f"Conv1D: padding must be non-negative, got {params.get('padding')}")

        groups = parse_int_1d(params.get("groups", 1), default=1)
        if groups <= 0:
            raise ValueError(f"Conv1D: groups must be greater than 0, got {params.get('groups')}")

        if in_channels != "ANY" and in_channels != -1 and in_channels != "LAZY":
            if in_channels <= 0:
                raise ValueError(f"Conv1D: in_channels must be greater than 0, got {in_channels}")
            if in_channels % groups != 0:
                raise ValueError(f"Conv1D: in_channels ({in_channels}) must be divisible by groups ({groups})")

        out_channels = params.get("out_channels", 16)
        try:
            out_channels = int(out_channels)
        except Exception:
            pass
        if isinstance(out_channels, int) and out_channels <= 0:
            raise ValueError(f"Conv1D: out_channels must be greater than 0, got {out_channels}")
        if isinstance(out_channels, int) and out_channels % groups != 0:
            raise ValueError(f"Conv1D: out_channels ({out_channels}) must be divisible by groups ({groups})")

        if L != "ANY":
            try:
                l_val = int(L)
                out_l = math.floor((l_val + 2 * pad - dil * (k - 1) - 1) / st + 1)
                if out_l <= 0:
                    raise ValueError(f"Conv1D: Negative spatial dimension length={out_l} with input L={L}, kernel={k}, stride={st}, padding={pad}, dilation={dil}")
            except ValueError as e:
                raise e
            except Exception:
                out_l = "ANY"
        else:
            out_l = "ANY"

        return {"out": (B, out_channels, out_l)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_ch    = params.get("in_channels",  3)
        out_ch   = params.get("out_channels", 16)
        k_size   = params.get("kernel_size",  3)
        stride   = params.get("stride",   1)
        padding  = params.get("padding",  0)
        dilation = params.get("dilation", 1)
        groups   = params.get("groups",   1)
        bias     = params.get("bias",     True)
        if in_ch == "LAZY":
            return f"{layer_name} = nn.LazyConv1d({out_ch}, {k_size}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups}, bias={bias})"
        return (
            f"{layer_name} = nn.Conv1d("
            f"{in_ch}, {out_ch}, {k_size}, "
            f"stride={stride}, padding={padding}, "
            f"dilation={dilation}, groups={groups}, bias={bias})"
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var  = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class EmbeddingBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="embedding",
            name="Embedding",
            category="Core Layers",
            color="#3b82f6",
            is_functional=False,
            inputs=[PortDef(id="in", name="Indices", var_hint="indices")],
            outputs=[PortDef(id="out", name="Output", var_hint="embedded")],
            params=[
                ParamDef(name="num_embeddings", type="int", default=1000, section="basic", description="Size of the dictionary of embeddings"),
                ParamDef(name="embedding_dim", type="int", default=128, section="basic", description="The size of each embedding vector"),
                ParamDef(name="padding_idx", type="int", default=None, section="advanced", description="If specified, padding entries do not contribute to gradient"),
                ParamDef(name="max_norm", type="float", default=None, section="advanced", description="If given, each embedding vector with norm larger than max_norm is renormalized"),
                ParamDef(name="norm_type", type="float", default=2.0, section="advanced", description="The p of the p-norm to compute for the max_norm option"),
                ParamDef(name="scale_grad_by_freq", type="bool", default=False, section="advanced", description="Scale gradients by inverse of word frequency"),
                ParamDef(name="sparse", type="bool", default=False, section="advanced", description="If True, gradient w.r.t. weight matrix will be a sparse tensor"),
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or in_shape == ("ANY",):
            return {"out": ("ANY",)}

        num_embeddings = params.get("num_embeddings", 1000)
        if isinstance(num_embeddings, int) and num_embeddings <= 0:
            raise ValueError(f"Embedding: num_embeddings must be greater than 0, got {num_embeddings}")

        embedding_dim = params.get("embedding_dim", 128)
        if isinstance(embedding_dim, int) and embedding_dim <= 0:
            raise ValueError(f"Embedding: embedding_dim must be greater than 0, got {embedding_dim}")

        return {"out": tuple(in_shape) + (embedding_dim,)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        num_emb = params.get("num_embeddings", 1000)
        emb_dim = params.get("embedding_dim", 128)
        args = [f"{num_emb}", f"{emb_dim}"]

        padding_idx = params.get("padding_idx")
        if padding_idx is not None and str(padding_idx).strip() not in ("", "None", "none"):
            args.append(f"padding_idx={padding_idx}")

        max_norm = params.get("max_norm")
        if max_norm is not None and str(max_norm).strip() not in ("", "None", "none"):
            args.append(f"max_norm={max_norm}")
            norm_type = params.get("norm_type", 2.0)
            if norm_type != 2.0:
                args.append(f"norm_type={norm_type}")

        if params.get("scale_grad_by_freq", False):
            args.append("scale_grad_by_freq=True")

        if params.get("sparse", False):
            args.append("sparse=True")

        return f"{layer_name} = nn.Embedding({', '.join(args)})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"
```

#### C. Scalar Binary Ops in `backend/blocks/tensor_ops.py`
Helper for scalar validation:
```python
def _is_valid_scalar(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "none"
```
Updates in `AddBlock`:
```python
class AddBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="add",
            name="Add",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[
                PortDef(id="in", name="Inputs", is_list=True)
            ],
            outputs=[PortDef(id="out", name="Out", var_hint="sum_out")],
            params=[
                ParamDef(name="scalar_a", type="string", default=None, section="advanced", description="Scalar left operand"),
                ParamDef(name="scalar_b", type="string", default=None, section="advanced", description="Scalar right operand"),
            ]
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        src_vars = []
        for v in input_vars.values():
            if isinstance(v, list):
                src_vars.extend([str(x) for x in v if x and x != "None"])
            elif v and v != "None":
                src_vars.append(str(v))
        if _is_valid_scalar(params.get("scalar_a")):
            src_vars.insert(0, str(params["scalar_a"]))
        if _is_valid_scalar(params.get("scalar_b")):
            src_vars.append(str(params["scalar_b"]))

        if src_vars:
            add_expr = " + ".join(src_vars)
            return f"{out_var} = {add_expr}"
        return f"{out_var} = 0"
```

Updates in `SubBlock`:
```python
class SubBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="sub",
            name="Subtract",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a", name="A"), PortDef(id="in_b", name="B")],
            outputs=[PortDef(id="out", name="Out", var_hint="diff")],
            params=[
                ParamDef(name="scalar_a", type="string", default=None, section="advanced", description="Scalar left operand"),
                ParamDef(name="scalar_b", type="string", default=None, section="advanced", description="Scalar right operand"),
            ]
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        if (a == "None" or not a) and _is_valid_scalar(params.get("scalar_a")):
            a = str(params["scalar_a"])
        if (b == "None" or not b) and _is_valid_scalar(params.get("scalar_b")):
            b = str(params["scalar_b"])
        return f"{out_var} = {a} - {b}"
```

Updates in `MulBlock`:
```python
class MulBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="mul",
            name="Multiply",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in", name="Inputs", is_list=True)],
            outputs=[PortDef(id="out", name="Out", var_hint="product")],
            params=[
                ParamDef(name="scalar_a", type="string", default=None, section="advanced", description="Scalar left operand"),
                ParamDef(name="scalar_b", type="string", default=None, section="advanced", description="Scalar right operand"),
            ]
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        src_vars = []
        for v in input_vars.values():
            if isinstance(v, list):
                src_vars.extend([str(x) for x in v if x and x != "None"])
            elif v and v != "None":
                src_vars.append(str(v))
        if _is_valid_scalar(params.get("scalar_a")):
            src_vars.insert(0, str(params["scalar_a"]))
        if _is_valid_scalar(params.get("scalar_b")):
            src_vars.append(str(params["scalar_b"]))

        if src_vars:
            mul_expr = " * ".join(src_vars)
            return f"{out_var} = {mul_expr}"
        return f"{out_var} = 1"
```

Updates in `DivBlock`:
```python
class DivBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="div",
            name="Divide",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a", name="A"), PortDef(id="in_b", name="B")],
            outputs=[PortDef(id="out", name="Out", var_hint="quotient")],
            params=[
                ParamDef(name="scalar_a", type="string", default=None, section="advanced", description="Scalar left operand"),
                ParamDef(name="scalar_b", type="string", default=None, section="advanced", description="Scalar right operand"),
            ]
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        if (a == "None" or not a) and _is_valid_scalar(params.get("scalar_a")):
            a = str(params["scalar_a"])
        if (b == "None" or not b) and _is_valid_scalar(params.get("scalar_b")):
            b = str(params["scalar_b"])
        return f"{out_var} = {a} / {b}"
```

#### D. Imports & Naming in `backend/compiler.py`
1. Line 745:
   ```python
   imports = [
       "import torch",
       "import torch.nn as nn",
       "import math",
   ]
   ```
2. Line 864:
   ```python
   elif block_id in ("linear", "conv1d", "conv2d", "embedding", "layernorm", "batchnorm2d", "maxpool2d", "avgpool2d", "adaptiveavgpool2d", "dropout", "relu", "gelu", "silu", "sigmoid", "tanh", "softmax"):
       member_cand = f"layer_{_sanitize(raw_lbl or block_id)}"
   ```
3. Line 885:
   ```python
   if params.get(k) == "LAZY" and block_id in ("linear", "conv1d", "conv2d"):
       continue
   ```

#### E. Decompiler updates in `backend/python_decompiler.py`
1. `LAYER_MAP`:
   ```python
   "Embedding": ("embedding", ["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"], {
       "padding_idx": None, "max_norm": None, "norm_type": 2.0, "scale_grad_by_freq": False, "sparse": False
   }),
   "SiLU": ("silu", ["inplace"], {"inplace": False}),
   ```
2. `FUNCTIONAL_MAP`:
   ```python
   "silu": ("silu", ["inplace"], {"inplace": False}),
   ```
3. Input shape lookahead in `_decompile_module_def` (around line 360):
   ```python
   elif linfo["block"] == "conv1d":
       in_ch = linfo["params"].get("in_channels", 3)
       inferred_shape = f"(1, {in_ch}, 128)"
       break
   elif linfo["block"] == "embedding":
       inferred_shape = "(1, 64)"
       break
   ```

---

## 5. Verification Method

Once implemented, the changes can be independently verified via the following steps:

1. **Unit Test Suite Execution**:
   ```bash
   pytest backend/tests/
   ```
   Must pass 100% of existing tests with 0 failures.

2. **Dedicated Test Module**:
   Implement `backend/tests/test_r1_blocks_and_scalars.py` verifying:
   - **GELUBlock**:
     - `infer_shapes` returns input shape.
     - `emit_init` emits `nn.GELU(approximate='none')` or `nn.GELU(approximate='tanh')`.
     - `emit_forward` calls `self.layer_xxx(x)`.
   - **SiLUBlock**:
     - `infer_shapes` returns input shape.
     - `emit_init` emits `nn.SiLU(inplace=False)` or `nn.SiLU(inplace=True)`.
   - **Conv1DBlock**:
     - Shape inference with 3D tensor `(4, 3, 100)` -> `(4, 16, 50)` with `stride=2`.
     - Auto-infer `in_channels` when set to `-1`.
     - Negative length raises `ValueError`.
     - Parameter checks (`stride <= 0`, `groups <= 0`, etc.) raise `ValueError`.
     - `emit_init` emits `nn.Conv1d` or `nn.LazyConv1d`.
   - **EmbeddingBlock**:
     - Shape inference with `(2, 64)` -> `(2, 64, 128)`.
     - Negative `num_embeddings` / `embedding_dim` raises `ValueError`.
     - `emit_init` emits `nn.Embedding` with optional `padding_idx`, `max_norm`.
   - **Scalar Ops (`Add`, `Sub`, `Mul`, `Div`)**:
     - `Add`: `emit_forward` with `scalar_b=1` produces `out = x + 1`.
     - `Add`: `emit_forward` with `scalar_a=1` produces `out = 1 + x`.
     - `Sub`: `emit_forward` with `scalar_b=1` produces `out = x - 1`.
     - `Sub`: `emit_forward` with `scalar_a=1` produces `out = 1 - x`.
     - `Mul`: `emit_forward` with `scalar_b=2` produces `out = x * 2`.
     - `Mul`: `emit_forward` with `scalar_a=2` produces `out = 2 * x`.
     - `Div`: `emit_forward` with `scalar_b="math.sqrt(self.d_model)"` produces `out = x / math.sqrt(self.d_model)`.
     - `Div`: `emit_forward` with `scalar_a=1` produces `out = 1 / x`.
   - **Python Execution & Roundtrip**:
     - Construct a model graph with `Embedding`, `Conv1d`, `GELU`, `SiLU`, and `Div` with `scalar_b="math.sqrt(self.d_model)"`.
     - Generate PyTorch code via `generate_pytorch_code`.
     - Verify generated code compiles via `compile(code, "<test>", "exec")`.
     - Execute the class with `exec`, instantiate it with dummy weights, and execute forward pass on dummy tensor inputs without exception.

3. **Block Schema Dump Verification**:
   ```bash
   python backend/dump_block_schema.py
   ```
   Verify stdout prints `Dumped 34 blocks to ...` and inspect `backend/block_schema.json` to confirm `"gelu"`, `"silu"`, `"conv1d"`, `"embedding"` entries exist.

4. **Frontend TypeScript Compilation**:
   ```bash
   npx tsc --noEmit
   ```
   Must exit with code 0 and 0 errors.

5. **Invalidation Conditions**:
   - If `emit_forward` still emits `None` for missing edge inputs when scalars are present, the fix is invalid.
   - If `import math` is omitted and `x / math.sqrt(d)` raises `NameError: name 'math' is not defined` during forward execution, the fix is invalid.
   - If `GELUBlock`, `SiLUBlock`, `Conv1DBlock`, or `EmbeddingBlock` are not present in `_BLOCK_INSTANCES` or `/api/blocks`, acceptance criterion R1 is unsatisfied.
