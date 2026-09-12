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
                    type="string",
                    default="(1, 3, 224, 224)",
                    section="basic",
                    description="The shape of the input tensor, e.g. (batch, channels, H, W)"
                )
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shape_str = params.get("shape", "(1, 3, 224, 224)")
        try:
            # Very robust parsing: strip everything that's not a digit or comma
            clean = "".join(c for c in str(shape_str) if c.isdigit() or c == ',')
            shape = tuple(int(s) for s in clean.split(",") if s)
            return {"out": shape if shape else (1, 3, 224, 224)}
        except Exception:
            return {"out": (1, 3, 224, 224)}

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

        if in_features != -1 and isinstance(in_features, int) and in_features <= 0:
            raise ValueError(f"Linear: in_features must be greater than 0, got {in_features}")

        # Auto-infer in_features if set to -1
        if in_features == -1 and len(in_shape) > 0 and in_shape[-1] != "ANY":
            in_features = in_shape[-1]
            params["in_features"] = in_features

        out_features = params.get("out_features", 64)
        if isinstance(out_features, int) and out_features <= 0:
            raise ValueError(f"Linear: out_features must be greater than 0, got {out_features}")

        if len(in_shape) > 0 and in_shape[-1] != "ANY" and in_shape[-1] != in_features:
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
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)}

        B, C, H, W = in_shape
        in_channels = params.get("in_channels", 3)
        
        # Auto-infer in_channels if set to -1
        if in_channels == -1 and C != "ANY":
            in_channels = C
            params["in_channels"] = in_channels

        if C != "ANY" and C != in_channels:
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

        if in_channels != "ANY" and in_channels != -1:
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
