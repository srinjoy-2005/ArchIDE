from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class ReLUBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="relu",
            name="ReLU",
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
        return f"{layer_name} = nn.ReLU(inplace={inplace})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class SoftmaxBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="softmax",
            name="Softmax",
            category="Activations",
            color="#f59e0b",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="probs")],
            params=[
                ParamDef(name="dim", type="int", default=-1, section="basic", description="Dimension along which Softmax will be computed")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        dim = params.get("dim", -1)
        return f"{layer_name} = nn.Softmax(dim={dim})"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class SigmoidBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="sigmoid",
            name="Sigmoid",
            category="Activations",
            color="#f59e0b",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="sig_out")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        return f"{layer_name} = nn.Sigmoid()"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


class TanhBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="tanh",
            name="Tanh",
            category="Activations",
            color="#f59e0b",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="tanh_out")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        return f"{layer_name} = nn.Tanh()"

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        layer_name = f"self.layer_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "None")
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        return f"{out_var} = {layer_name}({in_var})"


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

