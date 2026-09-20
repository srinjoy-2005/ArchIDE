import math
from typing import Dict, Tuple, Any
from .base import BaseBlock, parse_int_or_tuple2d
from models import BlockDef, PortDef, ParamDef


class InputBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="input",
            name="Input",
            category="Core Layers",
            color="#10b981",
            is_functional=True,
            inputs=[],
            outputs=[PortDef(id="out", name="Output")],
            params=[
                ParamDef(
                    name="shape",
                    type="shape",
                    default="(1, 3, 224, 224)",
                    section="basic",
                    description="The shape of the input tensor, e.g. (batch, channels, H, W)"
                )
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shape_val = params.get("shape", "(1, 3, 224, 224)")
        if isinstance(shape_val, (int, float)):
            shape = (int(shape_val),)
        elif isinstance(shape_val, (list, tuple)):
            shape = tuple(int(s) for s in shape_val)
        else:
            try:
                clean = "".join(c for c in str(shape_val) if c.isdigit() or c in (',', '-'))
                shape = tuple(int(s) for s in clean.split(",") if s)
            except Exception:
                shape = (1, 3, 224, 224)

        if not shape:
            shape = (1, 3, 224, 224)

        if any(d <= 0 for d in shape):
            raise ValueError(f"Input: tensor dimensions must be greater than 0, got {shape}")

        return {"out": shape}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        return ""


class OutputBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="output",
            name="Output",
            category="Core Layers",
            color="#f43f5e",
            is_functional=True,
            inputs=[PortDef(id="in", name="Return Value", is_list=True)],
            outputs=[],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        # Return aggregation is handled by the compiler — this block emits nothing directly.
        return ""


class LinearBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="linear",
            name="Linear",
            category="Core Layers",
            color="#3b82f6",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="fc_out")],
            params=[
                # Basic section
                ParamDef(name="in_features",  type="int", default=128, auto_infer=True, section="basic", description="Size of each input sample"),
                ParamDef(name="out_features", type="int", default=64,  section="basic", description="Size of each output sample"),
                # Advanced section
                ParamDef(name="bias",         type="bool", default=True, section="advanced", description="If True, adds a learnable bias to the output"),
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or in_shape == ("ANY",):
            return {"out": ("ANY",)}

        # nn.Linear operates on the last dimension only: (*, H_in) -> (*, H_out).
        # It natively supports any number of leading dimensions, so no shape guard is needed.
        in_features = params.get("in_features", 128)

        if in_features != -1 and in_features != "LAZY" and isinstance(in_features, int) and in_features <= 0:
            raise ValueError(f"Linear: in_features must be greater than 0, got {in_features}")

        # Auto-infer in_features if set to -1
        if in_features == -1 and len(in_shape) > 0 and in_shape[-1] != "ANY":
            in_features = in_shape[-1]
            params["in_features"] = in_features

        out_features = params.get("out_features", 64)
        if isinstance(out_features, int) and out_features <= 0:
            raise ValueError(f"Linear: out_features must be greater than 0, got {out_features}")

        if in_features != "LAZY" and len(in_shape) > 0 and in_shape[-1] != "ANY" and in_shape[-1] != in_features:
            raise ValueError(
                f"Linear: expected in_features={in_features}, "
                f"but input last dim is {in_shape[-1]}."
            )

        out_shape = list(in_shape)
        if len(out_shape) > 0:
            out_shape[-1] = out_features
        else:
            out_shape = [out_features]
        return {"out": tuple(out_shape)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_feat  = params.get("in_features",  128)
        out_feat = params.get("out_features", 64)
        bias     = params.get("bias", True)
        if in_feat == "LAZY":
            return f"{layer_name} = nn.LazyLinear({out_feat}, bias={bias})"
        return f"{layer_name} = nn.Linear({in_feat}, {out_feat}, bias={bias})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var  = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "Applies a linear transformation to the incoming data: `y = xA^T + b`",
            "details": "### `nn.Linear`\nThis module creates a single layer feed forward network with `in_features` inputs and `out_features` outputs. It is commonly used as a fully connected layer."
        }


class Conv2DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="conv2d",
            name="Conv2D",
            category="Core Layers",
            color="#3b82f6",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="conv_feat")],
            params=[
                # Basic section
                ParamDef(name="in_channels",  type="int", default=3,  auto_infer=True, section="basic", description="Number of channels in the input image"),
                ParamDef(name="out_channels", type="int", default=16, section="basic", description="Number of channels produced by the convolution"),
                ParamDef(name="kernel_size",  type="int", default=3,  section="basic", description="Size of the convolving kernel"),
                # Advanced section
                ParamDef(name="stride",       type="int",  default=1,    section="advanced", description="Stride of the convolution"),
                ParamDef(name="padding",      type="int",  default=0,    section="advanced", description="Padding added to both sides of the input"),
                ParamDef(name="dilation",     type="int",  default=1,    section="advanced", description="Spacing between kernel elements"),
                ParamDef(name="groups",       type="int",  default=1,    section="advanced", description="Number of blocked connections from input to output"),
                ParamDef(name="bias",         type="bool", default=True, section="advanced", description="If True, adds a learnable bias to the output"),
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or in_shape == ("ANY",):
            return {"out": ("ANY",)}

        if len(in_shape) != 4:
            raise ValueError(
                f"Conv2D expects a 4D tensor (Batch, Channels, Height, Width), "
                f"but received {len(in_shape)}D tensor with shape {in_shape}."
            )

        B, C, H, W = in_shape
        in_channels = params.get("in_channels", 3)
        
        # Auto-infer in_channels if set to -1
        if in_channels == -1 and C != "ANY":
            in_channels = C
            params["in_channels"] = in_channels

        if in_channels != "LAZY" and C != "ANY" and C != in_channels:
            raise ValueError(
                f"Conv2D: expected in_channels={in_channels}, "
                f"but input has {C} channels."
            )

        kernel = parse_int_or_tuple2d(params.get("kernel_size"), default=(3, 3))
        stride = parse_int_or_tuple2d(params.get("stride"), default=(1, 1))
        padding = parse_int_or_tuple2d(params.get("padding"), default=(0, 0))
        dilation = parse_int_or_tuple2d(params.get("dilation"), default=(1, 1))

        kh, kw = kernel
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation

        if sh <= 0 or sw <= 0:
            raise ValueError(f"Conv2D: stride must be greater than 0, got {params.get('stride')}")
        if kh <= 0 or kw <= 0:
            raise ValueError(f"Conv2D: kernel_size must be greater than 0, got {params.get('kernel_size')}")
        if dh <= 0 or dw <= 0:
            raise ValueError(f"Conv2D: dilation must be greater than 0, got {params.get('dilation')}")
        if ph < 0 or pw < 0:
            raise ValueError(f"Conv2D: padding must be non-negative, got {params.get('padding')}")

        groups = params.get("groups", 1)
        try:
            groups = int(groups)
        except Exception:
            groups = 1
        if groups <= 0:
            raise ValueError(f"Conv2D: groups must be greater than 0, got {params.get('groups')}")

        if in_channels != "ANY" and in_channels != -1 and in_channels != "LAZY":
            if in_channels <= 0:
                raise ValueError(f"Conv2D: in_channels must be greater than 0, got {in_channels}")
            if in_channels % groups != 0:
                raise ValueError(f"Conv2D: in_channels ({in_channels}) must be divisible by groups ({groups})")

        out_channels = params.get("out_channels", 16)
        try:
            out_channels = int(out_channels)
        except Exception:
            pass
        if isinstance(out_channels, int) and out_channels <= 0:
            raise ValueError(f"Conv2D: out_channels must be greater than 0, got {out_channels}")
        if isinstance(out_channels, int) and out_channels % groups != 0:
            raise ValueError(f"Conv2D: out_channels ({out_channels}) must be divisible by groups ({groups})")

        if H != "ANY":
            try:
                h_val = int(H)
                out_h = math.floor((h_val + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
                if out_h <= 0:
                    raise ValueError(f"Conv2D: Negative spatial dimension height={out_h} with input H={H}, kernel={kh}, stride={sh}, padding={ph}, dilation={dh}")
            except ValueError as e:
                raise e
            except Exception:
                out_h = "ANY"
        else:
            out_h = "ANY"

        if W != "ANY":
            try:
                w_val = int(W)
                out_w = math.floor((w_val + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
                if out_w <= 0:
                    raise ValueError(f"Conv2D: Negative spatial dimension width={out_w} with input W={W}, kernel={kw}, stride={sw}, padding={pw}, dilation={dw}")
            except ValueError as e:
                raise e
            except Exception:
                out_w = "ANY"
        else:
            out_w = "ANY"

        return {"out": (B, out_channels, out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name  = f"self.layer_{node_id.replace('-', '_')}"
        in_ch    = params.get("in_channels",  3)
        out_ch   = params.get("out_channels", 16)
        k_size   = params.get("kernel_size",  3)
        stride   = params.get("stride",   1)
        padding  = params.get("padding",  0)
        dilation = params.get("dilation", 1)
        groups   = params.get("groups",   1)
        bias     = params.get("bias",     True)
        if in_ch == "LAZY":
            return f"{layer_name} = nn.LazyConv2d({out_ch}, {k_size}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups}, bias={bias})"
        return (
            f"{layer_name} = nn.Conv2d("
            f"{in_ch}, {out_ch}, {k_size}, "
            f"stride={stride}, padding={padding}, "
            f"dilation={dilation}, groups={groups}, bias={bias})"
        )

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var  = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class ShapeExtractorBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="shape_extractor",
            name="Shape Extractor",
            category="Core Layers",
            color="#a855f7",
            is_functional=True,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[
                PortDef(id="shape", name="Shape", var_hint="shape"),
                PortDef(id="dim_0", name="Dim 0 (B)", var_hint="b"),
                PortDef(id="dim_1", name="Dim 1 (C)", var_hint="c"),
                PortDef(id="dim_2", name="Dim 2 (H)", var_hint="h"),
                PortDef(id="dim_3", name="Dim 3 (W)", var_hint="w"),
            ],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        shape_dim = (len(in_shape),) if in_shape and in_shape != ("ANY",) else ("ANY",)
        return {
            "shape": shape_dim,
            "dim_0": (1,),
            "dim_1": (1,),
            "dim_2": (1,),
            "dim_3": (1,),
        }

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        in_var = input_vars.get("in", "None")
        lines = []
        if "shape" in output_vars:
            lines.append(f"{output_vars['shape']} = {in_var}.shape")
        for i in range(4):
            if f"dim_{i}" in output_vars:
                lines.append(f"{output_vars[f'dim_{i}']} = {in_var}.shape[{i}] if len({in_var}.shape) > {i} else None")
        return "\n        ".join(lines)

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "Extracts tensor shape and individual dimension sizes.",
            "details": "### Shape Extractor\nExtracts tensor `.shape` tuple and individual dimension sizes (`dim_0` to `dim_3`) for dynamic dimension routing."
        }


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

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "Applies a 1D convolution over an input signal composed of several input planes.",
            "details": "### nn.Conv1d\nApplies a 1D convolution over a 3D input tensor `(B, C, L)`."
        }


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

    def docs(self) -> Dict[str, str]:
        return {
            "intro": "A simple lookup table that stores embeddings of a fixed dictionary and size.",
            "details": "### nn.Embedding\nLookup table for word/token embeddings with dimension (num_embeddings, embedding_dim)."
        }

