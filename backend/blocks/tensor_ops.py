from typing import Dict, Tuple, Any
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

class AddBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="add",
            name="Add",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[
                PortDef(id="in_0", name="Input 1"),
                PortDef(id="in_1", name="Input 2")
            ],
            outputs=[PortDef(id="out", name="Out", var_hint="sum")],
            params=[
                ParamDef(name="num_inputs", type="int", default=2)
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        # Basic shape broadcasting could be added here.
        # For now, just return the first found shape or ANY.
        for val in input_shapes.values():
            if val != ("ANY",):
                return {"out": val}
        return {"out": ("ANY",)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        src_vars = [var for var in input_vars.values() if var != "None"]
        if src_vars:
            add_expr = " + ".join(src_vars)
            return f"{out_var} = {add_expr}"
        return f"{out_var} = 0"


class SubBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="sub",
            name="Subtract",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
            outputs=[PortDef(id="out",name="Out", var_hint="diff")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in_a", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        return f"{out_var} = {a} - {b}"


class MulBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="mul",
            name="Multiply",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
            outputs=[PortDef(id="out",name="Out", var_hint="product")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in_a", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        return f"{out_var} = {a} * {b}"


class DivBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="div",
            name="Divide",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
            outputs=[PortDef(id="out",name="Out", var_hint="quotient")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in_a", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        return f"{out_var} = {a} / {b}"


class PowBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="pow",
            name="Power",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a",name="Base")],
            outputs=[PortDef(id="out",name="Out", var_hint="powered")],
            params=[ParamDef(name="exponent", type="float", default=2.0)]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": input_shapes.get("in_a", ("ANY",))}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        exp = params.get("exponent", 2.0)
        return f"{out_var} = torch.pow({a}, {exp})"


class MatMulBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="matmul",
            name="MatMul",
            category="Tensor Ops",
            color="#8b5cf6",
            is_functional=True,
            inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
            outputs=[PortDef(id="out",name="Out", var_hint="matmul_out")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        a = input_shapes.get("in_a", ("ANY",))
        b = input_shapes.get("in_b", ("ANY",))
        if a == ("ANY",) or b == ("ANY",):
            return {"out": ("ANY",)}
        
        # (..., M, K) x (..., K, N) -> (..., M, N)
        if len(a) < 2 or len(b) < 2:
            return {"out": ("ANY",)}
            
        if a[-1] != "ANY" and b[-2] != "ANY" and a[-1] != b[-2]:
            raise ValueError(f"MatMul shape mismatch: {a} and {b}")
            
        out_shape = list(a[:-1]) + [b[-1]]
        return {"out": tuple(out_shape)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        a = input_vars.get("in_a", "None")
        b = input_vars.get("in_b", "None")
        return f"{out_var} = torch.matmul({a}, {b})"


class UnsqueezeBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="unsqueeze",
            name="Unsqueeze",
            category="Tensor Ops",
            color="#ef4444",
            is_functional=True,
            inputs=[PortDef(id="in",name="Input")],
            outputs=[PortDef(id="out",name="Out", var_hint="unsqueezed")],
            params=[ParamDef(name="dim", type="int", default=0)]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        in_shape = input_shapes.get("in", ("ANY",))
        if in_shape == ("ANY",):
            return {"out": ("ANY",)}
            
        dim = params.get("dim", 0)
        out_shape = list(in_shape)
        if dim < 0:
            dim += len(out_shape) + 1
        out_shape.insert(dim, 1)
        return {"out": tuple(out_shape)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        in_var = input_vars.get("in", "None")
        dim = params.get("dim", 0)
        return f"{out_var} = torch.unsqueeze({in_var}, dim={dim})"


class CatBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="cat",
            name="Concat",
            category="Tensor Ops",
            color="#ef4444",
            is_functional=True,
            inputs=[
                PortDef(id="in_a",name="A"),
                PortDef(id="in_b",name="B"),
                PortDef(id="in_c",name="C")
            ],
            outputs=[PortDef(id="out",name="Out", var_hint="concat")],
            params=[ParamDef(name="dim", type="int", default=-1)]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": ("ANY",)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        dim = params.get("dim", -1)
        
        valid_vars = [v for k, v in input_vars.items() if v != "None"]
        if not valid_vars:
            return f"{out_var} = None"
            
        vars_str = ", ".join(valid_vars)
        return f"{out_var} = torch.cat([{vars_str}], dim={dim})"


class SplitBlock(BaseBlock):
    @property
    def definition(self) -> BlockDef:
        return BlockDef(
            id="split",
            name="Split (Chunk)",
            category="Tensor Ops",
            color="#ef4444",
            is_functional=True,
            inputs=[
                PortDef(id="in", name="Input Tensor")
            ],
            outputs=[
                PortDef(id="out_1", name="Chunk 1", var_hint="chunk_1"),
                PortDef(id="out_2", name="Chunk 2", var_hint="chunk_2")
            ],
            params=[
                ParamDef(name="chunks", type="int", default=2),
                ParamDef(name="dim", type="int", default=-1)
            ]
        )

    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        chunks = params.get("chunks", 2)
        in_shape = input_shapes.get("in", ("ANY",))
        
        out_shapes = {}
        for i in range(1, chunks + 1):
            out_shapes[f"out_{i}"] = ("ANY",)
            
        return out_shapes

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        chunks = params.get("chunks", 2)
        dim = params.get("dim", -1)
        in_var = input_vars.get("in", "None")
        
        out_var_names = []
        for i in range(1, chunks + 1):
            out_var_names.append(output_vars.get(f"out_{i}", f"x_{node_id.replace('-', '_')}_{i}"))
            
        out_vars_str = ", ".join(out_var_names)
        return f"{out_vars_str} = torch.chunk({in_var}, chunks={chunks}, dim={dim})"
