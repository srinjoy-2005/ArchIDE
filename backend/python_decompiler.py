import ast
import json
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple, Set


# ─── Standard Layer Mapping ──────────────────────────────────────────────────
LAYER_MAP: Dict[str, Tuple[str, List[str], Dict[str, Any]]] = {
    "Linear": ("linear", ["in_features", "out_features", "bias"], {"bias": True}),
    "LazyLinear": ("linear", ["out_features", "bias"], {"in_features": "LAZY", "bias": True}),
    "Conv2d": ("conv2d", ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"], {
        "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True
    }),
    "Conv1d": ("conv1d", ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"], {
        "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True
    }),
    "ReLU": ("relu", ["inplace"], {"inplace": False}),
    "GELU": ("gelu", ["approximate"], {"approximate": "none"}),
    "SiLU": ("gelu", [], {}),
    "Sigmoid": ("sigmoid", [], {}),
    "Tanh": ("tanh", [], {}),
    "Softmax": ("softmax", ["dim"], {"dim": -1}),
    "LayerNorm": ("layernorm", ["normalized_shape", "eps"], {"eps": 1e-05}),
    "BatchNorm2d": ("batchnorm2d", ["num_features", "eps", "momentum"], {"eps": 1e-05, "momentum": 0.1}),
    "BatchNorm1d": ("batchnorm2d", ["num_features", "eps", "momentum"], {"eps": 1e-05, "momentum": 0.1}),
    "Dropout": ("dropout", ["p", "inplace"], {"p": 0.5, "inplace": False}),
    "MaxPool2d": ("maxpool2d", ["kernel_size", "stride", "padding", "dilation"], {"stride": 2, "padding": 0, "dilation": 1}),
    "AvgPool2d": ("avgpool2d", ["kernel_size", "stride", "padding"], {"stride": 2, "padding": 0}),
    "AdaptiveAvgPool2d": ("adaptiveavgpool2d", ["output_size"], {"output_size": (1, 1)}),
}

FUNCTIONAL_MAP: Dict[str, Tuple[str, List[str], Dict[str, Any]]] = {
    "flatten": ("flatten", ["start_dim", "end_dim"], {"start_dim": 1, "end_dim": -1}),
    "transpose": ("transpose", ["dim0", "dim1"], {"dim0": -2, "dim1": -1}),
    "permute": ("transpose", ["dim0", "dim1"], {"dim0": -2, "dim1": -1}),
    "matmul": ("matmul", [], {}),
    "bmm": ("matmul", [], {}),
    "cat": ("cat", ["dim"], {"dim": -1}),
    "concat": ("cat", ["dim"], {"dim": -1}),
    "stack": ("cat", ["dim"], {"dim": 0}),
    "reshape": ("reshape", ["shape"], {}),
    "view": ("reshape", ["shape"], {}),
    "squeeze": ("reshape", ["shape"], {}),
    "unsqueeze": ("unsqueeze", ["dim"], {"dim": 0}),
    "sin": ("sin", [], {}),
    "cos": ("cos", [], {}),
    "add": ("add", [], {}),
    "sub": ("sub", [], {}),
    "mul": ("mul", [], {}),
    "div": ("div", [], {}),
    "pow": ("pow", ["exponent"], {"exponent": 2.0}),
    "relu": ("relu", ["inplace"], {"inplace": False}),
    "gelu": ("gelu", ["approximate"], {"approximate": "none"}),
    "silu": ("gelu", [], {}),
    "sigmoid": ("sigmoid", [], {}),
    "tanh": ("tanh", [], {}),
    "softmax": ("softmax", ["dim"], {"dim": -1}),
    "dropout": ("dropout", ["p"], {"p": 0.5}),
    "layer_norm": ("layernorm", ["normalized_shape", "eps"], {"eps": 1e-05}),
    "split": ("split", ["split_size_or_sections", "dim"], {"dim": 0}),
}


def _eval_ast_literal(node: ast.AST, init_vars: Set[str]) -> Any:
    """Evaluates an AST expression to a literal value or @var:<name> binding."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in init_vars:
            return f"@var:{node.id}"
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        return node.id
    if isinstance(node, ast.Attribute):
        # e.g. self.d_model -> @var:d_model
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"@var:{node.attr}"
        return f"{_eval_ast_literal(node.value, init_vars)}.{node.attr}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_ast_literal(node.operand, init_vars)
        if isinstance(val, (int, float)):
            return -val
        return f"-{val}"
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_eval_ast_literal(elt, init_vars) for elt in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Dict):
        return {
            _eval_ast_literal(k, init_vars): _eval_ast_literal(v, init_vars)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp):
        left = _eval_ast_literal(node.left, init_vars)
        right = _eval_ast_literal(node.right, init_vars)
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        op_str = op_map.get(type(node.op), "+")
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        return f"{left} {op_str} {right}"
    if isinstance(node, ast.Call):
        func_name = ast.unparse(node.func)
        args = [_eval_ast_literal(a, init_vars) for a in node.args]
        if func_name in ("math.sqrt", "sqrt") and args and isinstance(args[0], (int, float)):
            import math
            return math.sqrt(args[0])
        args_str = ", ".join(str(a) for a in args)
        return f"{func_name}({args_str})"
    try:
        return ast.unparse(node)
    except Exception:
        return str(node)


class PyTorchASTDecompiler:
    """
    Decompiles a PyTorch nn.Module Python file into ArchIDE Agentic IR (.ir.json).
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir
        self.imports: Dict[str, str] = {}  # ClassName -> module/path
        self.init_variables: List[Dict[str, Any]] = []
        self.init_var_names: Set[str] = set()
        self.layer_instances: Dict[str, Dict[str, Any]] = {}  # self.attr -> {block, params, custom_module_id, ...}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[str] = []
        self.env: Dict[str, Tuple[str, str]] = {}  # var_name -> (node_id, port_id)
        self.node_counters: Dict[str, int] = {}

    def _next_node_id(self, base_name: str) -> str:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).strip('_').lower() or "node"
        count = self.node_counters.get(clean, 0) + 1
        self.node_counters[clean] = count
        if count == 1 and clean in {"in", "out", "input", "output", "flatten"}:
            return clean
        return f"{clean}_{count}" if count > 1 else clean

    def decompile_source(self, code_str: str, file_stem: str = "main") -> Dict[str, Any]:
        tree = ast.parse(code_str)

        # 1. Collect Imports
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                mod = stmt.module or ""
                mod_path = mod.strip(".").replace(".", "/")
                for alias in stmt.names:
                    self.imports[alias.name] = f"{mod_path}" if mod_path else alias.name
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    self.imports[alias.name] = alias.name.strip(".").replace(".", "/")

        # 2. Find nn.Module Class
        module_class: Optional[ast.ClassDef] = None
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                base_names = [ast.unparse(b) for b in stmt.bases]
                if any("Module" in b for b in base_names) or not module_class:
                    module_class = stmt

        if not module_class:
            raise ValueError("No nn.Module class found in Python source code.")

        model_name = file_stem if file_stem != "main" else (module_class.name.lower() or "main")
        if model_name.startswith("modules/"):
            model_name = model_name[8:]

        # 3. Analyze __init__
        init_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
        if init_method:
            self._parse_init(init_method)

        # 4. Analyze forward
        forward_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
        if forward_method:
            self._parse_forward(forward_method)
        else:
            raise ValueError(f"No forward() method found in class {module_class.name}")

        return {
            "name": model_name,
            "variables": self.init_variables,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def _parse_init(self, func: ast.FunctionDef) -> None:
        args = func.args.args[1:]  # skip self
        defaults = func.args.defaults
        num_no_default = len(args) - len(defaults)

        for idx, arg in enumerate(args):
            var_name = arg.arg
            self.init_var_names.add(var_name)

            default_val = None
            if idx >= num_no_default:
                default_node = defaults[idx - num_no_default]
                default_val = _eval_ast_literal(default_node, set())

            type_hint = "int"
            if arg.annotation:
                ann_str = ast.unparse(arg.annotation).lower()
                if "str" in ann_str: type_hint = "string"
                elif "float" in ann_str: type_hint = "float"
                elif "bool" in ann_str: type_hint = "bool"
                elif "tuple" in ann_str or "shape" in ann_str: type_hint = "shape"
                elif "int" in ann_str: type_hint = "int"

            if default_val is not None:
                if isinstance(default_val, float):
                    type_hint = "float"
                elif isinstance(default_val, bool):
                    type_hint = "bool"
                elif isinstance(default_val, str) and not arg.annotation:
                    type_hint = "string"
                elif isinstance(default_val, (tuple, list)) and not arg.annotation:
                    type_hint = "shape"

            self.init_variables.append({
                "id": f"var_{var_name}",
                "name": var_name,
                "type": type_hint,
                "default": default_val if default_val is not None else 0,
                "description": "",
                "scope": "init_param",
            })

        for stmt in func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        attr_name = target.attr
                        if isinstance(stmt.value, ast.Call):
                            self._parse_layer_instantiation(attr_name, stmt.value)

    def _parse_layer_instantiation(self, attr_name: str, call: ast.Call) -> None:
        func_name = ast.unparse(call.func)
        raw_class_name = func_name.split(".")[-1]

        # Case 1: Standard nn.Module layer
        if raw_class_name in LAYER_MAP:
            block_id, pos_params, default_params = LAYER_MAP[raw_class_name]
            params = dict(default_params)

            for idx, arg in enumerate(call.args):
                if idx < len(pos_params):
                    p_name = pos_params[idx]
                    params[p_name] = _eval_ast_literal(arg, self.init_var_names)

            for kw in call.keywords:
                if kw.arg:
                    params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)

            self.layer_instances[attr_name] = {
                "block": block_id,
                "params": params,
                "is_custom": False,
            }
            return

        # Case 2: Sequential
        if raw_class_name == "Sequential":
            params = {}
            for idx, arg in enumerate(call.args):
                params[f"param_{idx}"] = _eval_ast_literal(arg, self.init_var_names)
            for kw in call.keywords:
                if kw.arg:
                    params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)

            self.layer_instances[attr_name] = {
                "block": "custom_module",
                "custom_module_id": "sequential",
                "params": params,
                "is_custom": True,
                "class_name": "Sequential",
            }
            return

        # Case 3: ModuleList
        if raw_class_name == "ModuleList":
            # Extract submodules in list
            list_layers = []
            if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                for elt in call.args[0].elts:
                    if isinstance(elt, ast.Call):
                        sub_func = ast.unparse(elt.func).split(".")[-1]
                        sub_mod_id = self.imports.get(sub_func, f"modules/{sub_func.lower()}")
                        list_layers.append({"block": "custom_module", "custom_module_id": sub_mod_id})
            self.layer_instances[attr_name] = {
                "block": "module_list",
                "layers": list_layers,
                "is_custom": True,
            }
            return

        # Case 4: Custom Submodule / Class
        custom_mod_id = self.imports.get(raw_class_name)
        if not custom_mod_id:
            lowered = raw_class_name.lower()
            if self.workspace_dir:
                cand_paths = [
                    os.path.join(self.workspace_dir, "modules", f"{lowered}.arch"),
                    os.path.join(self.workspace_dir, "modules", f"{lowered}.ir.json"),
                    os.path.join(self.workspace_dir, "modules", f"{lowered}.py"),
                    os.path.join(os.path.dirname(self.workspace_dir), "ir", "modules", f"{lowered}.ir.json"),
                    os.path.join(os.path.dirname(self.workspace_dir), "graphs", "modules", f"{lowered}.arch"),
                    os.path.join(os.path.dirname(self.workspace_dir), "python", "modules", f"{lowered}.py"),
                ]
                if any(os.path.exists(p) for p in cand_paths):
                    custom_mod_id = f"modules/{lowered}"
                else:
                    custom_mod_id = lowered
            else:
                custom_mod_id = lowered

        params = {}
        for kw in call.keywords:
            if kw.arg:
                params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)
        for idx, arg in enumerate(call.args):
            params[f"param_{idx}"] = _eval_ast_literal(arg, self.init_var_names)

        self.layer_instances[attr_name] = {
            "block": "custom_module",
            "custom_module_id": custom_mod_id,
            "params": params,
            "is_custom": True,
            "class_name": raw_class_name,
        }

    def _parse_forward(self, func: ast.FunctionDef) -> None:
        input_args = func.args.args[1:]  # skip self
        for idx, arg in enumerate(input_args):
            arg_name = arg.arg
            in_node_id = "in" if idx == 0 else f"in_{idx + 1}"

            # Look ahead for first downstream layer to set a compatible input shape
            inferred_shape = "(1, 3, 224, 224)"
            for stmt in func.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    if any(isinstance(a, ast.Name) and a.id == arg_name for a in stmt.value.args):
                        if isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr in self.layer_instances:
                            linfo = self.layer_instances[stmt.value.func.attr]
                            if linfo["block"] == "linear":
                                in_feat = linfo["params"].get("in_features", 128)
                                inferred_shape = f"(1, {in_feat})"
                                break
                            elif linfo["block"] == "conv2d":
                                in_ch = linfo["params"].get("in_channels", 3)
                                inferred_shape = f"(1, {in_ch}, 224, 224)"
                                break
                            elif linfo["block"] == "layernorm":
                                n_shape = linfo["params"].get("normalized_shape", 512)
                                inferred_shape = f"(1, 64, {n_shape})"
                                break
                            elif linfo.get("is_custom"):
                                c_params = linfo.get("params", {})
                                d_m = c_params.get("d_model", 512)
                                inferred_shape = f"(1, 64, {d_m})"
                                break

            self.nodes[in_node_id] = {
                "block": "input",
                "params": {"shape": inferred_shape}
            }
            self.env[arg_name] = (in_node_id, "out")

        self._parse_statements(func.body)

    def _parse_statements(self, statements: List[ast.AST]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        out_var = target.id
                        src_node, src_port = self._parse_expr(stmt.value, target_hint=out_var)
                        if src_node:
                            self.env[out_var] = (src_node, src_port)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Unpacking multiple return values: out, attn = ...
                        src_node, src_port = self._parse_expr(stmt.value)
                        if src_node:
                            for idx, elt in enumerate(target.elts):
                                if isinstance(elt, ast.Name):
                                    handle = f"out_{idx + 1}" if idx > 0 else "out"
                                    self.env[elt.id] = (src_node, handle)
            elif isinstance(stmt, ast.AugAssign):
                # e.g. x += residual -> x = x + residual
                if isinstance(stmt.target, ast.Name):
                    out_var = stmt.target.id
                    bin_op = ast.BinOp(left=ast.Name(id=out_var), op=stmt.op, right=stmt.value)
                    src_node, src_port = self._parse_expr(bin_op, target_hint=out_var)
                    if src_node:
                        self.env[out_var] = (src_node, src_port)
            elif isinstance(stmt, ast.Return):
                self._parse_return(stmt.value)
            elif isinstance(stmt, ast.For):
                # Unroll for loops over module lists if simple
                if isinstance(stmt.iter, ast.Attribute) and isinstance(stmt.iter.value, ast.Name) and stmt.iter.value.id == "self":
                    mod_list = self.layer_instances.get(stmt.iter.attr)
                    if mod_list and mod_list.get("block") == "module_list":
                        for sub_layer in mod_list.get("layers", []):
                            # Instantiate inline sublayer step
                            node_id = self._next_node_id(sub_layer.get("custom_module_id", "module").split("/")[-1])
                            self.nodes[node_id] = {
                                "block": "custom_module",
                                "custom_module_id": sub_layer.get("custom_module_id"),
                            }
                            # Connect loop variable
                            if isinstance(stmt.target, ast.Name) and stmt.body:
                                loop_var = stmt.target.id
                                prev_src, prev_port = self.env.get(loop_var, ("in", "out"))
                                self.edges.append(f"{prev_src}.{prev_port} -> {node_id}.in")
                                self.env[loop_var] = (node_id, "out")

    def _parse_expr(self, expr: ast.AST, target_hint: str = "") -> Tuple[Optional[str], str]:
        # Variable name reference
        if isinstance(expr, ast.Name):
            return self.env.get(expr.id, (None, "out"))

        # Binary Op: a + b, a - b, a * b, a / b, a @ b
        if isinstance(expr, ast.BinOp):
            op_map = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.MatMult: "matmul"
            }
            block_id = op_map.get(type(expr.op), "add")
            node_id = self._next_node_id(target_hint or block_id)

            self.nodes[node_id] = {"block": block_id}
            if target_hint and not target_hint.startswith("x_"):
                self.nodes[node_id]["var_name"] = target_hint

            src_a, port_a = self._parse_expr(expr.left)
            src_b, port_b = self._parse_expr(expr.right)

            in_port_a = "in_a" if block_id in {"div", "sub", "matmul"} else "in"
            in_port_b = "in_b" if block_id in {"div", "sub", "matmul"} else "in"

            if src_a:
                self.edges.append(f"{src_a}.{port_a} -> {node_id}.{in_port_a}")
            else:
                scalar_val = _eval_ast_literal(expr.left, self.init_var_names)
                self.nodes[node_id].setdefault("params", {})["scalar_a"] = scalar_val

            if src_b:
                self.edges.append(f"{src_b}.{port_b} -> {node_id}.{in_port_b}")
            else:
                scalar_val = _eval_ast_literal(expr.right, self.init_var_names)
                self.nodes[node_id].setdefault("params", {})["scalar_b"] = scalar_val

            return (node_id, "out")

        # Layer or Functional call
        if isinstance(expr, ast.Call):
            return self._parse_call(expr, target_hint)

        return (None, "out")

    def _parse_call(self, call: ast.Call, target_hint: str = "") -> Tuple[Optional[str], str]:
        # 1A. Check if calling indexed module list on self: self.layers[i](x)
        if (
            isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "self"
        ):
            attr = call.func.value.attr
            layer_info = self.layer_instances.get(attr)
            slice_val = _eval_ast_literal(call.func.slice, self.init_var_names)
            base_name = f"{attr}_{slice_val}" if slice_val is not None else attr
            node_id = self._next_node_id(base_name)
            node_entry: Dict[str, Any] = {"block": "custom_module"}
            if layer_info and layer_info.get("block") == "module_list":
                layers = layer_info.get("layers", [])
                if isinstance(slice_val, int) and 0 <= slice_val < len(layers):
                    node_entry["custom_module_id"] = layers[slice_val].get("custom_module_id", f"modules/{attr}")
                else:
                    node_entry["custom_module_id"] = f"modules/{attr}"
            else:
                node_entry["custom_module_id"] = f"modules/{attr}"
            if target_hint and not target_hint.startswith("x_"):
                node_entry["var_name"] = target_hint
            self.nodes[node_id] = node_entry

            for idx, arg in enumerate(call.args):
                src_node, src_port = self._parse_expr(arg)
                if src_node:
                    target_handle = "in" if idx == 0 else f"in_{idx + 1}"
                    self.edges.append(f"{src_node}.{src_port} -> {node_id}.{target_handle}")
            return (node_id, "out")

        # 1B. Check if calling self.layer_xxx or self.custom_xxx
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
            attr = call.func.attr
            layer_info = self.layer_instances.get(attr)
            if layer_info:
                block_id = layer_info["block"]
                clean_attr = attr.replace("layer_", "").replace("custom_", "")
                base_name = clean_attr or target_hint or block_id
                node_id = self._next_node_id(base_name)

                node_entry: Dict[str, Any] = {"block": block_id}
                if layer_info.get("is_custom"):
                    node_entry["custom_module_id"] = layer_info["custom_module_id"]
                if layer_info.get("params"):
                    node_entry["params"] = dict(layer_info["params"])
                if target_hint and not target_hint.startswith("x_"):
                    node_entry["var_name"] = target_hint

                self.nodes[node_id] = node_entry

                # Positional arguments
                for idx, arg in enumerate(call.args):
                    if isinstance(arg, ast.Constant) and arg.value is None:
                        continue
                    src_node, src_port = self._parse_expr(arg)
                    if src_node:
                        target_handle = "in" if idx == 0 else f"in_{idx + 1}"
                        self.edges.append(f"{src_node}.{src_port} -> {node_id}.{target_handle}")

                # Keyword arguments
                for kw in call.keywords:
                    if kw.arg and kw.value:
                        src_node, src_port = self._parse_expr(kw.value)
                        if src_node:
                            self.edges.append(f"{src_node}.{src_port} -> {node_id}.{kw.arg}")

                return (node_id, "out")

        # 2. Check functional calls: torch.<func>, F.<func>, nn.functional.<func>
        if isinstance(call.func, ast.Attribute):
            obj = call.func.value
            method = call.func.attr

            # 2A. torch.<func>, F.<func>, etc.
            is_functional_obj = (
                (isinstance(obj, ast.Name) and obj.id in {"torch", "F", "functional"})
                or (isinstance(obj, ast.Attribute) and obj.attr in {"functional", "nn"})
            )
            if is_functional_obj and method in FUNCTIONAL_MAP:
                block_id, pos_params, default_params = FUNCTIONAL_MAP[method]
                node_id = self._next_node_id(target_hint or block_id)
                params = dict(default_params)

                self.nodes[node_id] = {"block": block_id}
                if target_hint and not target_hint.startswith("x_"):
                    self.nodes[node_id]["var_name"] = target_hint

                if block_id == "matmul":
                    if len(call.args) >= 2:
                        src_a, port_a = self._parse_expr(call.args[0])
                        src_b, port_b = self._parse_expr(call.args[1])
                        if src_a: self.edges.append(f"{src_a}.{port_a} -> {node_id}.in_a")
                        if src_b: self.edges.append(f"{src_b}.{port_b} -> {node_id}.in_b")
                elif block_id == "cat":
                    if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                        for elt in call.args[0].elts:
                            src_e, port_e = self._parse_expr(elt)
                            if src_e: self.edges.append(f"{src_e}.{port_e} -> {node_id}.in")
                    if len(call.args) > 1:
                        params["dim"] = _eval_ast_literal(call.args[1], self.init_var_names)
                    for kw in call.keywords:
                        if kw.arg == "dim":
                            params["dim"] = _eval_ast_literal(kw.value, self.init_var_names)
                    if params:
                        self.nodes[node_id]["params"] = params
                else:
                    if call.args:
                        src_in, port_in = self._parse_expr(call.args[0])
                        if src_in: self.edges.append(f"{src_in}.{port_in} -> {node_id}.in")
                        for idx, arg in enumerate(call.args[1:]):
                            if idx < len(pos_params):
                                params[pos_params[idx]] = _eval_ast_literal(arg, self.init_var_names)
                    for kw in call.keywords:
                        if kw.arg:
                            params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)
                    if params:
                        self.nodes[node_id]["params"] = params

                return (node_id, "out")

            # 2B. Tensor passthroughs: x.contiguous(), x.clone(), x.detach()
            if method in {"contiguous", "clone", "detach", "to", "type", "float", "long", "cuda", "cpu"}:
                return self._parse_expr(obj, target_hint)

            # 2C. Tensor methods: x.transpose(...), x.flatten(...), x.view(...)
            elif method in FUNCTIONAL_MAP:
                block_id, pos_params, default_params = FUNCTIONAL_MAP[method]
                node_id = self._next_node_id(target_hint or block_id)
                params = dict(default_params)

                for idx, arg in enumerate(call.args):
                    if idx < len(pos_params):
                        params[pos_params[idx]] = _eval_ast_literal(arg, self.init_var_names)
                for kw in call.keywords:
                    if kw.arg:
                        params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)

                self.nodes[node_id] = {"block": block_id, "params": params} if params else {"block": block_id}
                if target_hint and not target_hint.startswith("x_"):
                    self.nodes[node_id]["var_name"] = target_hint

                src_node, src_port = self._parse_expr(obj)
                if src_node:
                    self.edges.append(f"{src_node}.{src_port} -> {node_id}.in")

                return (node_id, "out")

        return (None, "out")

    def _parse_return(self, value: ast.AST) -> None:
        out_node_id = "out"
        self.nodes[out_node_id] = {"block": "output"}

        if isinstance(value, (ast.Tuple, ast.List)):
            for idx, elt in enumerate(value.elts):
                src_node, src_port = self._parse_expr(elt)
                if src_node:
                    handle = f"in_{idx + 1}" if idx > 0 else "in"
                    self.edges.append(f"{src_node}.{src_port} -> {out_node_id}.{handle}")
        elif value:
            src_node, src_port = self._parse_expr(value)
            if src_node:
                self.edges.append(f"{src_node}.{src_port} -> {out_node_id}.in")


def decompile_python_to_ir(
    py_path_or_code: str,
    output_path: Optional[str] = None,
    ir_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompiles a Python file (or source string) into Agentic IR (.ir.json).
    """
    if os.path.isfile(py_path_or_code):
        with open(py_path_or_code, "r", encoding="utf-8") as f:
            code_str = f.read()
        stem = os.path.splitext(os.path.basename(py_path_or_code))[0]
    else:
        code_str = py_path_or_code
        stem = "model"

    decompiler = PyTorchASTDecompiler(workspace_dir=workspace_dir)
    ir_dict = decompiler.decompile_source(code_str, file_stem=stem)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ir_dict, f, indent=2)

    return ir_dict
