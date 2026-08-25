from typing import Dict, Tuple, Any, List
from .base import BaseBlock
from models import BlockDef, PortDef, ParamDef

def broadcast_shapes(shape_a: Tuple, shape_b: Tuple) -> Tuple:
    if shape_a == ("ANY",): return shape_b
    if shape_b == ("ANY",): return shape_a
    
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + shape_a
    pad_b = (1,) * (max_len - len(shape_b)) + shape_b
    
    out_shape = []
    for a, b in zip(pad_a, pad_b):
        if a == "ANY": out_shape.append(b)
        elif b == "ANY": out_shape.append(a)
        elif a == b: out_shape.append(a)
        elif a == 1: out_shape.append(b)
        elif b == 1: out_shape.append(a)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
    return tuple(out_shape)

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
                PortDef(id="in", name="Inputs", is_list=True)
            ],
            outputs=[PortDef(id="out", name="Out", var_hint="sum")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shapes = []
        for v in input_shapes.values():
            if isinstance(v, list):
                shapes.extend([s for s in v if s != ("ANY",)])
            elif v != ("ANY",):
                shapes.append(v)
        if not shapes:
            return {"out": ("ANY",)}
            
        current_shape = shapes[0]
        for shape in shapes[1:]:
            current_shape = broadcast_shapes(current_shape, shape)
            
        return {"out": current_shape}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        src_vars = []
        for v in input_vars.values():
            if isinstance(v, list):
                src_vars.extend([x for x in v if x and x != "None"])
            elif v and v != "None":
                src_vars.append(v)
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
        a = input_shapes.get("in_a", ("ANY",))
        b = input_shapes.get("in_b", ("ANY",))
        return {"out": broadcast_shapes(a, b)}

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
            inputs=[PortDef(id="in", name="Inputs", is_list=True)],
            outputs=[PortDef(id="out", name="Out", var_hint="product")],
            params=[]
        )

    def infer_shapes(self, input_shapes: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        shapes = []
        for v in input_shapes.values():
            if isinstance(v, list):
                shapes.extend([s for s in v if s != ("ANY",)])
            elif v != ("ANY",):
                shapes.append(v)
        if not shapes:
            return {"out": ("ANY",)}
            
        current_shape = shapes[0]
        for shape in shapes[1:]:
            current_shape = broadcast_shapes(current_shape, shape)
            
        return {"out": current_shape}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        src_vars = []
        for v in input_vars.values():
            if isinstance(v, list):
                src_vars.extend([x for x in v if x and x != "None"])
            elif v and v != "None":
                src_vars.append(v)
        if src_vars:
            mul_expr = " * ".join(src_vars)
            return f"{out_var} = {mul_expr}"
        return f"{out_var} = 1"


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
        a = input_shapes.get("in_a", ("ANY",))
        b = input_shapes.get("in_b", ("ANY",))
        return {"out": broadcast_shapes(a, b)}

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
                PortDef(id="in", name="Tensors", is_list=True)
            ],
            outputs=[PortDef(id="out", name="Out", var_hint="concat")],
            params=[ParamDef(name="dim", type="int", default=-1)]
        )

    def infer_shapes(self, input_shapes: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": ("ANY",)}

    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        return ""

    def emit_forward(self, node_id: str, input_vars: Dict[str, Any], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        out_var = output_vars.get("out", f"x_{node_id.replace('-', '_')}")
        dim = params.get("dim", -1)
        src_vars = []
        for v in input_vars.values():
            if isinstance(v, list):
                src_vars.extend([x for x in v if x and x != "None"])
            elif v and v != "None":
                src_vars.append(v)
        if not src_vars:
            return f"{out_var} = None"
        vars_str = ", ".join(src_vars)
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
