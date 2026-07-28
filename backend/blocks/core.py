import math
from typing import Dict, Tuple, Any
from .base import BaseBlock
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
                ParamDef(name="shape", type="string", default="(1, 3, 224, 224)")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shape_str = params.get("shape", "(1, 3, 224, 224)")
        try:
            # Basic parsing of tuple string like "(1, 3, 224, 224)"
            shape = tuple(map(int, shape_str.strip("()").split(",")))
            return {"out": shape}
        except:
            return {"out": (1, 3, 224, 224)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        return "" # Handled specially by compiler usually, or just pass


class GouravBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="gourav",
            name="Gourav",
            category="Core Layers",
            color="#10b981",
            is_functional=True,
            inputs=[],
            outputs=[PortDef(id="out", name="Output")],
            params=[
                ParamDef(name="shape", type="string", default="(1, 3, 224, 224)")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shape_str = params.get("shape", "(1, 3, 224, 224)")
        try:
            shape = tuple(map(int, shape_str.strip("()").split(",")))
            return {"out": shape}
        except:
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
            inputs=[PortDef(id="in", name="Return Value")],
            outputs=[],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {} # Doesn't output anything

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        in_var = input_vars.get("in", "None")
        return f"return {in_var}"


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
            outputs=[PortDef(id="out", name="Output")],
            params=[
                ParamDef(name="in_features", type="int", default=128),
                ParamDef(name="out_features", type="int", default=64)
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape:
            return {"out": ("ANY",)}
        
        in_features = params.get("in_features", 128)
        out_features = params.get("out_features", 64)
        
        if in_shape[-1] != "ANY" and in_shape[-1] != in_features:
            raise ValueError(f"Linear layer expected in_features={in_features}, but got input with last dim {in_shape[-1]}")
            
        out_shape = list(in_shape)
        out_shape[-1] = out_features
        return {"out": tuple(out_shape)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_feat = params.get("in_features", 128)
        out_feat = params.get("out_features", 64)
        return f"{layer_name} = nn.Linear({in_feat}, {out_feat})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


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
            outputs=[PortDef(id="out", name="Output")],
            params=[
                ParamDef(name="in_channels", type="int", default=3),
                ParamDef(name="out_channels", type="int", default=16),
                ParamDef(name="kernel_size", type="int", default=3)
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)} # Fallback
            
        B, C, H, W = in_shape
        in_channels = params.get("in_channels", 3)
        if C != "ANY" and C != in_channels:
            raise ValueError(f"Conv2D expected in_channels={in_channels}, but got {C}")
            
        kernel = params.get("kernel_size", 3)
        padding = params.get("padding", 0)
        stride = params.get("stride", 1)
        
        try:
            out_h = math.floor((H + 2*padding - kernel) / stride + 1)
            out_w = math.floor((W + 2*padding - kernel) / stride + 1)
        except:
            out_h, out_w = "ANY", "ANY"
            
        return {"out": (B, params.get("out_channels", 16), out_h, out_w)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_ch = params.get("in_channels", 3)
        out_ch = params.get("out_channels", 16)
        k_size = params.get("kernel_size", 3)
        return f"{layer_name} = nn.Conv2d({in_ch}, {out_ch}, {k_size})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"
