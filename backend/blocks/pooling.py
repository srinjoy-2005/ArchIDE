import math
from typing import Dict, Tuple, Any
from .base import BaseBlock
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
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)
        dilation = params.get("dilation", 1)

        try:
            out_h = math.floor((H + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
            out_w = math.floor((W + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
        except Exception:
            out_h, out_w = "ANY", "ANY"

        return {"out": (B, C, out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)
        dilation = params.get("dilation", 1)
        return f"{layer_name} = nn.MaxPool2d({kernel}, stride={stride}, padding={padding}, dilation={dilation})"

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
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)

        try:
            out_h = math.floor((H + 2*padding - kernel) / stride + 1)
            out_w = math.floor((W + 2*padding - kernel) / stride + 1)
        except Exception:
            out_h, out_w = "ANY", "ANY"

        return {"out": (B, C, out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        kernel = params.get("kernel_size", 2)
        stride = params.get("stride", 2)
        padding = params.get("padding", 0)
        return f"{layer_name} = nn.AvgPool2d({kernel}, stride={stride}, padding={padding})"

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
