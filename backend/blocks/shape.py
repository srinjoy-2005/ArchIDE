from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class FlattenBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="flatten",
            name="Flatten",
            category="Shape Ops",
            color="#ef4444",
            is_functional=True,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="flat")],
            params=[
                ParamDef(name="start_dim", type="int", default=1, section="basic", description="First dimension to flatten"),
                ParamDef(name="end_dim", type="int", default=-1, section="basic", description="Last dimension to flatten")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in", ("ANY",))
        if in_shape == ("ANY",):
            return {"out": ("ANY",)}

        start_dim = params.get("start_dim", 1)
        end_dim = params.get("end_dim", -1)
        
        if end_dim < 0:
            end_dim += len(in_shape)
        if start_dim < 0:
            start_dim += len(in_shape)
            
        try:
            flat_size = 1
            for i in range(start_dim, end_dim + 1):
                if in_shape[i] == "ANY":
                    flat_size = "ANY"
                    break
                flat_size *= in_shape[i]
                
            out_shape = list(in_shape[:start_dim]) + [flat_size] + list(in_shape[end_dim + 1:])
            return {"out": tuple(out_shape)}
        except Exception:
            return {"out": ("ANY",)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        in_var = input_vars.get("in", "None")
        start_dim = params.get("start_dim", 1)
        end_dim = params.get("end_dim", -1)
        return f"{out_var} = torch.flatten({in_var}, start_dim={start_dim}, end_dim={end_dim})"


class ReshapeBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="reshape",
            name="Reshape",
            category="Shape Ops",
            color="#ef4444",
            is_functional=True,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output", var_hint="reshaped")],
            params=[
                ParamDef(name="shape", type="string", default="(-1,)", section="basic", description="Target shape, e.g. -1, 256")
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in", ("ANY",))
        shape_str = params.get("shape", "(-1,)")
        
        try:
            clean = "".join(c for c in str(shape_str) if c.isdigit() or c == ',' or c == '-')
            shape = tuple(int(s) for s in clean.split(",") if s)
            return {"out": shape if shape else ("ANY",)}
        except Exception:
            return {"out": ("ANY",)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        in_var = input_vars.get("in", "None")
        shape_str = params.get("shape", "-1")
        clean = "".join(c for c in str(shape_str) if c.isdigit() or c == ',' or c == '-')
        return f"{out_var} = {in_var}.reshape({clean})"
