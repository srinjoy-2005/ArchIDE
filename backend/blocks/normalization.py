from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class BatchNorm2DBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="batchnorm2d",
            name="BatchNorm2D",
            category="Normalization",
            color="#0ea5e9",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="norm_out")],
            params=[
                ParamDef(name="num_features", type="int", default=1, auto_infer=True, section="basic", description="Number of features/channels in the input. Auto-inferred if checked."),
                ParamDef(name="eps", type="float", default=1e-05, section="advanced", description="Value added to the denominator for numerical stability"),
                ParamDef(name="momentum", type="float", default=0.1, section="advanced", description="Value used for the running_mean and running_var computation")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape or len(in_shape) != 4:
            return {"out": ("ANY",)}

        num_features = params.get("num_features", -1)
        if num_features == -1 and len(in_shape) > 1 and in_shape[1] != "ANY":
            num_features = in_shape[1]
            params["num_features"] = num_features

        if len(in_shape) > 1 and in_shape[1] != "ANY" and in_shape[1] != num_features:
            raise ValueError(f"BatchNorm2D expected {num_features} channels, but got {in_shape[1]}")

        return {"out": in_shape}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        num_features = params.get("num_features", 1)
        if num_features == -1:
            num_features = 1 # Fallback if not inferred
        eps = params.get("eps", 1e-05)
        momentum = params.get("momentum", 0.1)
        return f"{layer_name} = nn.BatchNorm2d({num_features}, eps={eps}, momentum={momentum})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class LayerNormBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="layernorm",
            name="LayerNorm",
            category="Normalization",
            color="#0ea5e9",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="norm_out")],
            params=[
                ParamDef(name="normalized_shape", type="string", default="?", auto_infer=True, section="basic", description="Input shape from an expected size, auto-inferred from input."),
                ParamDef(name="eps", type="float", default=1e-05, section="advanced", description="Value added to the denominator for numerical stability")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in")
        if not in_shape:
            return {"out": ("ANY",)}

        # Auto-infer normalized_shape as the last dimension by default if not set
        norm_shape_str = params.get("normalized_shape", "?")
        if norm_shape_str == "?" and len(in_shape) > 0 and in_shape[-1] != "ANY":
            params["normalized_shape"] = str(in_shape[-1:])

        return {"out": in_shape}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        norm_shape = params.get("normalized_shape", "(1,)")
        if norm_shape == "?":
            norm_shape = "(1,)"
        # Clean it up to be a valid tuple string
        clean = "".join(c for c in str(norm_shape) if c.isdigit() or c == ',')
        eps = params.get("eps", 1e-05)
        return f"{layer_name} = nn.LayerNorm([{clean}], eps={eps})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class DropoutBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="dropout",
            name="Dropout",
            category="Normalization",
            color="#0ea5e9",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="dropped")],
            params=[
                ParamDef(name="p", type="float", default=0.5, section="basic", description="Probability of an element to be zeroed"),
                ParamDef(name="inplace", type="bool", default=False, section="advanced", description="Modify input in-place")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        p = params.get("p", 0.5)
        inplace = params.get("inplace", False)
        return f"{layer_name} = nn.Dropout(p={p}, inplace={inplace})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"
