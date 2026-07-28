from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class SinBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="sin",
            name="Sin",
            category="Trig",
            color="#ec4899",
            is_functional=True,
            inputs=[PortDef(id="in",name="Input")],
            outputs=[PortDef(id="out",name="Out")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        in_var = input_vars.get("in", "None")
        return f"{out_var} = torch.sin({in_var})"


class CosBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="cos",
            name="Cos",
            category="Trig",
            color="#ec4899",
            is_functional=True,
            inputs=[PortDef(id="in",name="Input")],
            outputs=[PortDef(id="out",name="Out")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        in_var = input_vars.get("in", "None")
        return f"{out_var} = torch.cos({in_var})"
