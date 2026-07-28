from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class ArangeBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="arange",
            name="Arange",
            category="Generators",
            color="#14b8a6",
            is_functional=True,
            inputs=[],
            outputs=[PortDef(id="out",name="Out")],
            params=[
                ParamDef(name="start", type="int", default=0),
                ParamDef(name="end", type="int", default=10),
                ParamDef(name="step", type="int", default=1)
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        start = params.get("start", 0)
        end = params.get("end", 10)
        step = params.get("step", 1)
        length = max(0, (end - start + step - 1) // step) if step != 0 else 0
        return {"out": (length,)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        start = params.get("start", 0)
        end = params.get("end", 10)
        step = params.get("step", 1)
        return f"{out_var} = torch.arange(start={start}, end={end}, step={step})"
