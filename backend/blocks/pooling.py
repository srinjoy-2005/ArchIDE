import math
from typing import Dict, Tuple, Any
from .base import BaseBlock, parse_int_or_tuple2d
from models import BlockDef, PortDef, ParamDef

class MaxPool2DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="maxpool2d",
            name="MaxPool2D",
            category="Pooling",
            color="#22c55e",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="pooled")],
            params=[
                ParamDef(name="kernel_size", type="int", default=2, section="basic", description="Size of the window to take a max over"),
                ParamDef(name="stride", type="int", default=2, section="basic", description="Stride of the window"),
                ParamDef(name="padding", type="int", default=0, section="advanced", description="Implicit zero padding to be added on both sides"),
                ParamDef(name="dilation", type="int", default=1, section="advanced", description="Parameter that controls the stride of elements in the window")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)}

        B, C, H, W = in_shape
        kernel = parse_int_or_tuple2d(params.get("kernel_size"), default=(2, 2))
        stride = parse_int_or_tuple2d(params.get("stride"), default=kernel)
        padding = parse_int_or_tuple2d(params.get("padding"), default=(0, 0))
        dilation = parse_int_or_tuple2d(params.get("dilation"), default=(1, 1))

        kh, kw = kernel
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation

        if sh <= 0 or sw <= 0:
            raise ValueError(f"MaxPool2D: stride must be greater than 0, got {params.get('stride')}")
        if kh <= 0 or kw <= 0:
            raise ValueError(f"MaxPool2D: kernel_size must be greater than 0, got {params.get('kernel_size')}")
        if dh <= 0 or dw <= 0:
            raise ValueError(f"MaxPool2D: dilation must be greater than 0, got {params.get('dilation')}")
        if ph < 0 or pw < 0:
            raise ValueError(f"MaxPool2D: padding must be non-negative, got {params.get('padding')}")

        if H != "ANY":
            try:
                h_val = int(H)
                out_h = math.floor((h_val + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
                if out_h <= 0:
                    raise ValueError(f"MaxPool2D: Output height {out_h} <= 0 with input H={H}, kernel={kh}, stride={sh}, padding={ph}, dilation={dh}")
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
                    raise ValueError(f"MaxPool2D: Output width {out_w} <= 0 with input W={W}, kernel={kw}, stride={sw}, padding={pw}, dilation={dw}")
            except ValueError as e:
                raise e
            except Exception:
                out_w = "ANY"
        else:
            out_w = "ANY"

        return {"out": (B, C, out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)
        dilation = params.get("dilation", 1)
        k_str = kernel if kernel is not None else 2
        s_str = f"stride={stride}" if stride is not None else f"stride={k_str}"
        p_str = f"padding={padding}" if padding is not None else "padding=0"
        d_str = f"dilation={dilation}" if dilation is not None else "dilation=1"
        return f"{layer_name} = nn.MaxPool2d({k_str}, {s_str}, {p_str}, {d_str})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class AvgPool2DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="avgpool2d",
            name="AvgPool2D",
            category="Pooling",
            color="#22c55e",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="pooled")],
            params=[
                ParamDef(name="kernel_size", type="int", default=2, section="basic", description="Size of the window"),
                ParamDef(name="stride", type="int", default=2, section="basic", description="Stride of the window"),
                ParamDef(name="padding", type="int", default=0, section="advanced", description="Implicit zero padding to be added on both sides")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)}

        B, C, H, W = in_shape
        kernel = parse_int_or_tuple2d(params.get("kernel_size"), default=(2, 2))
        stride = parse_int_or_tuple2d(params.get("stride"), default=kernel)
        padding = parse_int_or_tuple2d(params.get("padding"), default=(0, 0))

        kh, kw = kernel
        sh, sw = stride
        ph, pw = padding

        if sh <= 0 or sw <= 0:
            raise ValueError(f"AvgPool2D: stride must be greater than 0, got {params.get('stride')}")
        if kh <= 0 or kw <= 0:
            raise ValueError(f"AvgPool2D: kernel_size must be greater than 0, got {params.get('kernel_size')}")
        if ph < 0 or pw < 0:
            raise ValueError(f"AvgPool2D: padding must be non-negative, got {params.get('padding')}")

        if H != "ANY":
            try:
                h_val = int(H)
                out_h = math.floor((h_val + 2 * ph - kh) / sh + 1)
                if out_h <= 0:
                    raise ValueError(f"AvgPool2D: Output height {out_h} <= 0 with input H={H}, kernel={kh}, stride={sh}, padding={ph}")
            except ValueError as e:
                raise e
            except Exception:
                out_h = "ANY"
        else:
            out_h = "ANY"

        if W != "ANY":
            try:
                w_val = int(W)
                out_w = math.floor((w_val + 2 * pw - kw) / sw + 1)
                if out_w <= 0:
                    raise ValueError(f"AvgPool2D: Output width {out_w} <= 0 with input W={W}, kernel={kw}, stride={sw}, padding={pw}")
            except ValueError as e:
                raise e
            except Exception:
                out_w = "ANY"
        else:
            out_w = "ANY"

        return {"out": (B, C, out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)
        k_str = kernel if kernel is not None else 2
        s_str = f"stride={stride}" if stride is not None else f"stride={k_str}"
        p_str = f"padding={padding}" if padding is not None else "padding=0"
        return f"{layer_name} = nn.AvgPool2d({k_str}, {s_str}, {p_str})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class AdaptiveAvgPool2DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="adaptiveavgpool2d",
            name="AdaptiveAvgPool2D",
            category="Pooling",
            color="#22c55e",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="pooled")],
            params=[
                ParamDef(name="output_size", type="string", default="(1, 1)", section="basic", description="The target output size of the image of the form (H, W)")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)}

        B, C, _, _ = in_shape
        out_size_str = params.get("output_size", "(1, 1)")
        try:
            clean = "".join(c for c in str(out_size_str) if c.isdigit() or c == ',')
            size = tuple(int(s) for s in clean.split(",") if s)
            if len(size) != 2:
                raise ValueError("output_size must have 2 dimensions")
            return {"out": (B, C, size[0], size[1])}
        except Exception:
            return {"out": (B, C, "ANY", "ANY")}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        out_size = params.get("output_size", "(1, 1)")
        return f"{layer_name} = nn.AdaptiveAvgPool2d({out_size})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"
