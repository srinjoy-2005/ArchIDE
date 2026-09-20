import ast
import json
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple, Set, Union


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
    "Embedding": ("embedding", ["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"], {
        "padding_idx": None, "max_norm": None, "norm_type": 2.0, "scale_grad_by_freq": False, "sparse": False
    }),
    "ReLU": ("relu", ["inplace"], {"inplace": False}),
    "GELU": ("gelu", ["approximate"], {"approximate": "none"}),
    "SiLU": ("silu", ["inplace"], {"inplace": False}),
    "Sigmoid": ("sigmoid", [], {}),
    "Tanh": ("tanh", [], {}),
    "Softmax": ("softmax", ["dim"], {"dim": -1}),
    "LayerNorm": ("layernorm", ["normalized_shape", "eps"], {"eps": 1e-05}),
    "BatchNorm2d": ("batchnorm2d", ["num_features", "eps", "momentum"], {"eps": 1e-05, "momentum": 0.1}),
    "BatchNorm1d": ("batchnorm1d", ["num_features", "eps", "momentum"], {"eps": 1e-05, "momentum": 0.1}),
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
    "silu": ("silu", ["inplace"], {"inplace": False}),
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
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
        op_str = op_map.get(type(node.op), "+")
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.FloorDiv): return left // right
            if isinstance(node.op, ast.Mod): return left % right
            if isinstance(node.op, ast.Pow): return left ** right
        return f"{left} {op_str} {right}"
    if isinstance(node, ast.Call):
        func_name = ast.unparse(node.func)
        args = [_eval_ast_literal(a, init_vars) for a in node.args]
        if func_name in ("math.sqrt", "sqrt") and args and isinstance(args[0], (int, float)):
            import math
            return math.sqrt(args[0])
        kwargs = [f"{kw.arg}={_eval_ast_literal(kw.value, init_vars)}" for kw in node.keywords if kw.arg]
        all_args = [str(a) for a in args] + kwargs
        args_str = ", ".join(all_args)
        return f"{func_name}({args_str})"
    try:
        return ast.unparse(node)
    except Exception:
        return str(node)


class PyTorchASTDecompiler:
    """
    Decompiles a PyTorch nn.Module Python file into ArchIDE Agentic IR (.ir.json).
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        visited_paths: Optional[Set[str]] = None,
    ):
        self.workspace_dir = workspace_dir
        self.visited_paths: Set[str] = visited_paths if visited_paths is not None else set()
        self.imports: Dict[str, str] = {}  # ClassName -> module/path
        self.init_variables: List[Dict[str, Any]] = []
        self.init_var_names: Set[str] = set()
        self.layer_instances: Dict[str, Dict[str, Any]] = {}  # self.attr -> {block, params, custom_module_id, ...}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[str] = []
        self.env: Dict[str, Tuple[str, str]] = {}  # var_name -> (node_id, port_id)
        self.node_counters: Dict[str, int] = {}
        self.all_irs: Dict[str, Dict[str, Any]] = {}
        self.known_class_inits: Dict[str, List[str]] = {}
        self.attributes: Dict[str, Any] = {}

    def _next_node_id(self, base_name: str) -> str:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).strip('_').lower() or "node"
        count = self.node_counters.get(clean, 0) + 1
        self.node_counters[clean] = count
        if clean in {"out", "output"}:
            return f"out_{count}"
        if count == 1 and clean in {"in", "input", "flatten"}:
            return clean
        return f"{clean}_{count}" if count > 1 else clean

    def _find_module_classes(self, tree: ast.AST) -> Dict[str, ast.ClassDef]:
        """Find all ast.ClassDef in tree that inherit from Module or define forward()."""
        module_classes: Dict[str, ast.ClassDef] = {}
        known_module_names: Set[str] = {"Module", "nn.Module", "torch.nn.Module"}

        # First pass: classes that explicitly inherit from Module or define forward
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                base_names = [ast.unparse(b) for b in stmt.bases]
                has_forward = any(isinstance(m, ast.FunctionDef) and m.name == "forward" for m in stmt.body)
                inherits_module = any(any(m in b for m in known_module_names) for b in base_names)
                if inherits_module or has_forward:
                    module_classes[stmt.name] = stmt
                    known_module_names.add(stmt.name)

        # Second pass: classes inheriting from known discovered classes
        changed = True
        while changed:
            changed = False
            for stmt in tree.body:
                if isinstance(stmt, ast.ClassDef) and stmt.name not in module_classes:
                    base_names = [ast.unparse(b) for b in stmt.bases]
                    if any(any(b_name == k or k in b_name for k in known_module_names) for b_name in base_names):
                        module_classes[stmt.name] = stmt
                        known_module_names.add(stmt.name)
                        changed = True

        # Fallback: if no classes matched, take any ClassDef
        if not module_classes:
            for stmt in tree.body:
                if isinstance(stmt, ast.ClassDef):
                    module_classes[stmt.name] = stmt

        return module_classes

    def _get_class_deps(self, cls_node: ast.ClassDef, module_classes: Dict[str, ast.ClassDef]) -> Set[str]:
        deps: Set[str] = set()
        for node in ast.walk(cls_node):
            if isinstance(node, ast.Call):
                func_id = ast.unparse(node.func).split(".")[-1]
                if func_id in module_classes and func_id != cls_node.name:
                    deps.add(func_id)
        for b in cls_node.bases:
            b_name = ast.unparse(b).split(".")[-1]
            if b_name in module_classes and b_name != cls_node.name:
                deps.add(b_name)
        return deps

    def _topological_sort_classes(self, module_classes: Dict[str, ast.ClassDef]) -> List[str]:
        deps: Dict[str, Set[str]] = {
            name: self._get_class_deps(cls_node, module_classes)
            for name, cls_node in module_classes.items()
        }

        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        order: List[str] = []

        def visit(n: str):
            if n in temp_mark:
                return  # Cycle detected; break gracefully
            if n not in visited:
                temp_mark.add(n)
                for dep in sorted(deps.get(n, set())):
                    visit(dep)
                temp_mark.remove(n)
                visited.add(n)
                order.append(n)

        for name in module_classes:
            if name not in visited:
                visit(name)

        return order

    def _resolve_import_file(
        self,
        module_name: str,
        imported_name: str,
        level: int,
        source_dir: Optional[str],
    ) -> Optional[str]:
        candidates: List[str] = []

        # Relative import
        if level > 0 and source_dir:
            base_dir = source_dir
            for _ in range(level - 1):
                base_dir = os.path.dirname(base_dir)
            if module_name:
                parts = module_name.split(".")
                candidates.append(os.path.join(base_dir, *parts, f"{imported_name}.py"))
                candidates.append(os.path.join(base_dir, *parts, f"{imported_name.lower()}.py"))
                candidates.append(os.path.join(base_dir, *parts, "__init__.py"))
                candidates.append(os.path.join(base_dir, *parts) + ".py")
            else:
                candidates.append(os.path.join(base_dir, f"{imported_name}.py"))
                candidates.append(os.path.join(base_dir, f"{imported_name.lower()}.py"))
                candidates.append(os.path.join(base_dir, imported_name, "__init__.py"))

        # Absolute or workspace import
        search_roots = []
        if source_dir:
            search_roots.append(source_dir)
        if self.workspace_dir:
            search_roots.append(self.workspace_dir)
            search_roots.append(os.path.join(self.workspace_dir, "python"))
            search_roots.append(os.path.join(self.workspace_dir, "modules"))
            search_roots.append(os.path.join(self.workspace_dir, "python", "modules"))
            search_roots.append(os.path.join(self.workspace_dir, "models"))
            search_roots.append(os.path.join(self.workspace_dir, "python", "models"))

        parts = module_name.split(".") if module_name else []
        for r in search_roots:
            if parts:
                candidates.append(os.path.join(r, *parts, f"{imported_name}.py"))
                candidates.append(os.path.join(r, *parts, f"{imported_name.lower()}.py"))
                candidates.append(os.path.join(r, *parts, "__init__.py"))
                candidates.append(os.path.join(r, *parts) + ".py")
            candidates.append(os.path.join(r, f"{imported_name}.py"))
            candidates.append(os.path.join(r, f"{imported_name.lower()}.py"))
            candidates.append(os.path.join(r, "modules", f"{imported_name}.py"))
            candidates.append(os.path.join(r, "modules", f"{imported_name.lower()}.py"))

        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def _recursively_decompile_file(self, abs_file: str) -> None:
        try:
            with open(abs_file, "r", encoding="utf-8") as f:
                sub_code = f.read()
            sub_stem = os.path.splitext(os.path.basename(abs_file))[0]
            sub_decompiler = PyTorchASTDecompiler(
                workspace_dir=self.workspace_dir,
                visited_paths=self.visited_paths,
            )
            sub_irs = sub_decompiler.decompile_all_classes(
                sub_code, file_stem=sub_stem, source_path=abs_file
            )
            for k, v in sub_irs.items():
                if k not in self.all_irs:
                    self.all_irs[k] = v
            # Do NOT write sidecar IR files here — the class-name-lowered keys
            # (e.g. 'convblock') would create camelCase duplicates alongside the
            # canonical snake_case files written by the --all-from-python loop.
        except Exception:
            pass

    def _parse_imports(self, tree: ast.AST, source_path: Optional[str] = None) -> None:
        source_dir = os.path.dirname(os.path.abspath(source_path)) if source_path else None

        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                mod = stmt.module or ""
                level = stmt.level
                for alias in stmt.names:
                    local_name = alias.asname if alias.asname else alias.name
                    imported_name = alias.name
                    found_file = self._resolve_import_file(mod, imported_name, level, source_dir)
                    canonical_id = f"modules/{imported_name.lower()}"

                    if found_file:
                        abs_file = os.path.abspath(found_file)
                        if abs_file not in self.visited_paths:
                            self.visited_paths.add(abs_file)
                            self._recursively_decompile_file(abs_file)
                        self.imports[local_name] = canonical_id
                        self.imports[imported_name] = canonical_id
                    else:
                        mod_path = mod.strip(".").replace(".", "/")
                        val = f"{mod_path}" if mod_path else alias.name
                        self.imports[local_name] = val
                        self.imports[imported_name] = val

            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    local_name = alias.asname if alias.asname else alias.name
                    imported_name = alias.name
                    found_file = self._resolve_import_file("", imported_name, 0, source_dir)
                    canonical_id = f"modules/{imported_name.lower()}"
                    if found_file:
                        abs_file = os.path.abspath(found_file)
                        if abs_file not in self.visited_paths:
                            self.visited_paths.add(abs_file)
                            self._recursively_decompile_file(abs_file)
                        self.imports[local_name] = canonical_id
                        self.imports[imported_name] = canonical_id
                    else:
                        clean_name = alias.name.strip(".").replace(".", "/")
                        self.imports[local_name] = clean_name
                        self.imports[imported_name] = clean_name

    def _extract_self_chain(self, node: ast.AST) -> Optional[List[Union[str, int]]]:
        """
        Unpacks nested ast.Attribute and ast.Subscript chains originating at self.
        Examples:
          self.backbone.layer1 -> ["backbone", "layer1"]
          self.features[0].conv -> ["features", 0, "conv"]
          self.blocks[0] -> ["blocks", 0]
          self.fc -> ["fc"]
        """
        tokens: List[Union[str, int]] = []
        curr = node
        while True:
            if isinstance(curr, ast.Attribute):
                tokens.append(curr.attr)
                curr = curr.value
            elif isinstance(curr, ast.Subscript):
                slice_val = _eval_ast_literal(curr.slice, self.init_var_names)
                tokens.append(slice_val)
                curr = curr.value
            elif isinstance(curr, ast.Name):
                if curr.id == "self":
                    tokens.reverse()
                    return tokens if tokens else None
                return None
            else:
                return None

    def _resolve_chain_layer(self, chain: List[Union[str, int]]) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Resolves a chain of attributes/subscripts into (block_id, custom_module_id, params).
        """
        # 1. Exact match in layer_instances (e.g. "layer1" or "features.0.conv")
        dot_name = ".".join(str(t) for t in chain)
        if dot_name in self.layer_instances:
            linfo = self.layer_instances[dot_name]
            return linfo["block"], linfo.get("custom_module_id"), dict(linfo.get("params", {}))

        # 2. Single attribute
        if len(chain) == 1:
            attr = str(chain[0])
            if attr in self.layer_instances:
                linfo = self.layer_instances[attr]
                return linfo["block"], linfo.get("custom_module_id"), dict(linfo.get("params", {}))
            custom_id = self.imports.get(attr, f"modules/{attr.lower()}")
            return "custom_module", custom_id, {}

        # 3. Container traversal (e.g. self.features[0], self.layers[0].conv)
        root = str(chain[0])
        if root in self.layer_instances:
            curr_info = self.layer_instances[root]
            if isinstance(chain[1], int) and curr_info.get("layers"):
                idx = chain[1]
                layers = curr_info.get("layers", [])
                if 0 <= idx < len(layers):
                    sub = layers[idx]
                    if len(chain) == 2:
                        return sub.get("block", "custom_module"), sub.get("custom_module_id"), dict(sub.get("params", {}))
                    else:
                        sub_id = sub.get("custom_module_id") or f"modules/{root}_{idx}"
                        sub_attr = "_".join(str(t) for t in chain[2:])
                        return "custom_module", f"{sub_id}_{sub_attr}", {}

            parent_id = curr_info.get("custom_module_id", f"modules/{root}")
            sub_name = "_".join(str(t) for t in chain[1:])
            custom_id = f"{parent_id}_{sub_name}" if not parent_id.endswith(sub_name) else parent_id
            return "custom_module", custom_id, {}

        # 4. General fallback
        chain_str = "_".join(str(t) for t in chain)
        return "custom_module", f"modules/{chain_str}", {}

    def decompile_source(self, code_str: str, file_stem: str = "main", source_path: Optional[str] = None) -> Dict[str, Any]:
        tree = ast.parse(code_str)

        # 1. Collect Imports & recursively decompile local files
        self._parse_imports(tree, source_path=source_path)

        # 2. Find all nn.Module classes
        module_classes = self._find_module_classes(tree)
        if not module_classes:
            raise ValueError("No nn.Module class found in Python source code.")

        # Register local classes and extract their __init__ parameter signatures
        for cls_name, cls_node in module_classes.items():
            lowered = cls_name.lower()
            self.imports[cls_name] = f"modules/{lowered}"
            self.imports[lowered] = f"modules/{lowered}"
            init_method = next((m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
            if init_method:
                self.known_class_inits[cls_name] = [a.arg for a in init_method.args.args[1:]]

        # 3. Topologically sort classes
        sorted_classes = self._topological_sort_classes(module_classes)

        # Determine root class
        deps_dict = {name: self._get_class_deps(module_classes[name], module_classes) for name in sorted_classes}
        all_deps = {dep for d_set in deps_dict.values() for dep in d_set}
        candidate_roots = [c for c in sorted_classes if c not in all_deps]
        root_class_name = candidate_roots[-1] if candidate_roots else sorted_classes[-1]

        # 4. Decompile each class with isolated state
        for cls_name in sorted_classes:
            cls_node = module_classes[cls_name]
            is_root = (cls_name == root_class_name)
            model_name = file_stem if (is_root and file_stem != "main") else cls_name.lower()
            if model_name.startswith("modules/"):
                model_name = model_name[8:]

            ir = self._decompile_single_class(cls_node, model_name=model_name, module_classes=module_classes)
            self.all_irs[cls_name.lower()] = ir

        # Set decompiler instance state to match the root class for backward compatibility
        root_ir = self.all_irs[root_class_name.lower()]
        self.init_variables = list(root_ir["variables"])
        self.init_var_names = {v["name"] for v in self.init_variables}
        self.nodes = dict(root_ir["nodes"])
        self.edges = list(root_ir["edges"])

        return root_ir

    def decompile_all_classes(self, code_str: str, file_stem: str = "main", source_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        self.decompile_source(code_str, file_stem=file_stem, source_path=source_path)
        return self.all_irs

    def _decompile_single_class(
        self,
        module_class: ast.ClassDef,
        model_name: str,
        module_classes: Optional[Dict[str, ast.ClassDef]] = None,
    ) -> Dict[str, Any]:
        self.module_classes = module_classes or {}
        # Reset per-class state
        self.init_variables = []
        self.init_var_names = set()
        self.layer_instances = {}
        self.attributes = {}
        self.nodes = {}
        self.edges = []
        self.env = {}
        self.node_counters = {}

        # Parse __init__
        init_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
        if init_method:
            self._parse_init(init_method)

        # Parse forward
        forward_method = next((m for m in module_class.body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
        if forward_method:
            self._parse_forward(forward_method)
        else:
            raise ValueError(f"No forward() method found in class {module_class.name}")

        return {
            "name": model_name,
            "class_name": module_class.name,
            "variables": list(self.init_variables),
            "attributes": dict(self.attributes),
            "nodes": dict(self.nodes),
            "edges": list(self.edges),
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
                        else:
                            val = _eval_ast_literal(stmt.value, self.init_var_names)
                            self.attributes[attr_name] = val
                            self.init_var_names.add(attr_name)

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

            seq_layers = []
            for arg in call.args:
                if isinstance(arg, ast.Call):
                    sub_func = ast.unparse(arg.func).split(".")[-1]
                    if sub_func in LAYER_MAP:
                        s_block, s_pos_params, s_default_params = LAYER_MAP[sub_func]
                        s_params = dict(s_default_params)
                        for p_idx, p_arg in enumerate(arg.args):
                            if p_idx < len(s_pos_params):
                                s_params[s_pos_params[p_idx]] = _eval_ast_literal(p_arg, self.init_var_names)
                        for p_kw in arg.keywords:
                            if p_kw.arg:
                                s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                        seq_layers.append({"block": s_block, "params": s_params, "class_name": sub_func, "is_custom": False})
                    else:
                        sub_mod_id = self.imports.get(sub_func, f"modules/{sub_func.lower()}")
                        s_params = {}
                        for p_kw in arg.keywords:
                            if p_kw.arg:
                                s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                        for p_idx, p_arg in enumerate(arg.args):
                            s_params[f"param_{p_idx}"] = _eval_ast_literal(p_arg, self.init_var_names)
                        seq_layers.append({"block": "custom_module", "custom_module_id": sub_mod_id, "params": s_params, "class_name": sub_func, "is_custom": True})

            self.layer_instances[attr_name] = {
                "block": "custom_module",
                "custom_module_id": "sequential",
                "params": params,
                "layers": seq_layers,
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
                        if sub_func in LAYER_MAP:
                            s_block, s_pos_params, s_default_params = LAYER_MAP[sub_func]
                            s_params = dict(s_default_params)
                            for p_idx, p_arg in enumerate(elt.args):
                                if p_idx < len(s_pos_params):
                                    s_params[s_pos_params[p_idx]] = _eval_ast_literal(p_arg, self.init_var_names)
                            for p_kw in elt.keywords:
                                if p_kw.arg:
                                    s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                            list_layers.append({"block": s_block, "params": s_params, "class_name": sub_func, "is_custom": False})
                        else:
                            sub_mod_id = self.imports.get(sub_func, f"modules/{sub_func.lower()}")
                            s_pos_names = self.known_class_inits.get(sub_func, [])
                            s_params = {}
                            for p_idx, p_arg in enumerate(elt.args):
                                p_name = s_pos_names[p_idx] if p_idx < len(s_pos_names) else f"param_{p_idx}"
                                s_params[p_name] = _eval_ast_literal(p_arg, self.init_var_names)
                            for p_kw in elt.keywords:
                                if p_kw.arg:
                                    s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                            list_layers.append({"block": "custom_module", "custom_module_id": sub_mod_id, "params": s_params, "class_name": sub_func, "is_custom": True})
            elif call.args and isinstance(call.args[0], ast.ListComp):
                elt = call.args[0].elt
                repeat_count = 1
                if call.args[0].generators:
                    gen = call.args[0].generators[0]
                    if isinstance(gen.iter, ast.Call) and ast.unparse(gen.iter.func) == "range":
                        if gen.iter.args:
                            r_arg = _eval_ast_literal(gen.iter.args[0], self.init_var_names)
                            if isinstance(r_arg, str) and r_arg.startswith("@var:"):
                                v_name = r_arg[5:]
                                for v in self.init_variables:
                                    if v["name"] == v_name and isinstance(v.get("default"), int):
                                        repeat_count = v["default"]
                                        break
                            elif isinstance(r_arg, int):
                                repeat_count = r_arg

                if isinstance(elt, ast.Call):
                    sub_func = ast.unparse(elt.func).split(".")[-1]
                    s_params = {}
                    if sub_func in LAYER_MAP:
                        s_block, s_pos_params, s_default_params = LAYER_MAP[sub_func]
                        s_params = dict(s_default_params)
                        for p_idx, p_arg in enumerate(elt.args):
                            if p_idx < len(s_pos_params):
                                s_params[s_pos_params[p_idx]] = _eval_ast_literal(p_arg, self.init_var_names)
                        for p_kw in elt.keywords:
                            if p_kw.arg:
                                s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                        for _ in range(repeat_count):
                            list_layers.append({"block": s_block, "params": dict(s_params), "class_name": sub_func, "is_custom": False})
                    else:
                        sub_mod_id = self.imports.get(sub_func, f"modules/{sub_func.lower()}")
                        s_pos_names = self.known_class_inits.get(sub_func, [])
                        if not s_pos_names and hasattr(self, "module_classes") and sub_func in self.module_classes:
                            target_init = next((m for m in self.module_classes[sub_func].body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
                            if target_init:
                                s_pos_names = [a.arg for a in target_init.args.args[1:]]
                        for p_idx, p_arg in enumerate(elt.args):
                            p_name = s_pos_names[p_idx] if p_idx < len(s_pos_names) else f"param_{p_idx}"
                            s_params[p_name] = _eval_ast_literal(p_arg, self.init_var_names)
                        for p_kw in elt.keywords:
                            if p_kw.arg:
                                s_params[p_kw.arg] = _eval_ast_literal(p_kw.value, self.init_var_names)
                        for _ in range(repeat_count):
                            list_layers.append({"block": "custom_module", "custom_module_id": sub_mod_id, "params": dict(s_params), "class_name": sub_func, "is_custom": True})

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
                    custom_mod_id = f"modules/{lowered}"
            else:
                custom_mod_id = f"modules/{lowered}"

        # Parameter binding: map positional args to constructor argument names if known
        pos_param_names = self.known_class_inits.get(raw_class_name, [])
        if not pos_param_names and hasattr(self, "module_classes") and raw_class_name in self.module_classes:
            target_init = next((m for m in self.module_classes[raw_class_name].body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
            if target_init:
                pos_param_names = [a.arg for a in target_init.args.args[1:]]

        params = {}
        for kw in call.keywords:
            if kw.arg:
                params[kw.arg] = _eval_ast_literal(kw.value, self.init_var_names)
        for idx, arg in enumerate(call.args):
            p_name = pos_param_names[idx] if idx < len(pos_param_names) else f"param_{idx}"
            if p_name not in params:
                params[p_name] = _eval_ast_literal(arg, self.init_var_names)

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
            def _resolve_val(v, default_val=128):
                if isinstance(v, str) and "@var:" in v:
                    v_name = v.split("@var:")[-1]
                    for iv in self.init_variables:
                        if iv.get("name") == v_name:
                            return iv.get("default", default_val)
                    return default_val
                return v if v is not None else default_val

            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    if any(isinstance(a, ast.Name) and a.id == arg_name for a in node.args):
                        chain = self._extract_self_chain(node.func)
                        if chain:
                            b_id, _, b_params = self._resolve_chain_layer(chain)
                            if b_id == "linear":
                                in_feat = _resolve_val(b_params.get("in_features", 128), 128)
                                inferred_shape = f"(1, {in_feat})"
                                break
                            elif b_id == "conv2d":
                                in_ch = _resolve_val(b_params.get("in_channels", 3), 3)
                                inferred_shape = f"(1, {in_ch}, 224, 224)"
                                break
                            elif b_id == "conv1d":
                                in_ch = _resolve_val(b_params.get("in_channels", 3), 3)
                                inferred_shape = f"(1, {in_ch}, 128)"
                                break
                            elif b_id == "embedding":
                                inferred_shape = "(1, 64)"
                                break
                            elif b_id == "layernorm":
                                n_shape = _resolve_val(b_params.get("normalized_shape", 512), 512)
                                inferred_shape = f"(1, 64, {n_shape})"
                                break
                            elif b_id == "custom_module":
                                d_m = _resolve_val(b_params.get("d_model", 512), 512)
                                inferred_shape = f"(1, 64, {d_m})"
                                break

            self.nodes[in_node_id] = {
                "block": "input",
                "params": {"shape": inferred_shape}
            }
            self.env[arg_name] = (in_node_id, "out")

        self._parse_statements(func.body)

    def _eval_test_operand(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            attr = node.attr
            if attr in self.layer_instances:
                return self.layer_instances[attr]
            if attr in self.attributes:
                return self.attributes[attr]
            return None
        return _eval_ast_literal(node, self.init_var_names)

    def _eval_condition(self, test: ast.AST) -> Optional[bool]:
        if isinstance(test, ast.Compare):
            left_val = self._eval_test_operand(test.left)
            if len(test.ops) == 1 and len(test.comparators) == 1:
                op = test.ops[0]
                right_val = self._eval_test_operand(test.comparators[0])
                if isinstance(op, ast.IsNot):
                    return left_val is not right_val
                elif isinstance(op, ast.Is):
                    return left_val is right_val
                elif isinstance(op, ast.Eq):
                    return left_val == right_val
                elif isinstance(op, ast.NotEq):
                    return left_val != right_val
        if isinstance(test, (ast.Attribute, ast.Name)):
            val = self._eval_test_operand(test)
            return bool(val)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            sub = self._eval_condition(test.operand)
            return not sub if sub is not None else None
        return None

    def _parse_statements(self, statements: List[ast.AST]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                # Check for tuple unpacking of x.shape or x.size(): B, N, C = x.shape
                is_shape_val = False
                shape_tensor_expr = None
                if isinstance(stmt.value, ast.Attribute) and stmt.value.attr == "shape":
                    is_shape_val = True
                    shape_tensor_expr = stmt.value.value
                elif isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr in ("size", "shape"):
                    is_shape_val = True
                    shape_tensor_expr = stmt.value.func.value

                if is_shape_val and any(isinstance(t, (ast.Tuple, ast.List)) for t in stmt.targets):
                    target_tuple = next(t for t in stmt.targets if isinstance(t, (ast.Tuple, ast.List)))
                    src_node, src_port = self._parse_expr(shape_tensor_expr)
                    extractor_id = self._next_node_id("shape_extractor")
                    output_aliases = {}
                    for idx, elt in enumerate(target_tuple.elts):
                        if isinstance(elt, ast.Name):
                            output_aliases[f"dim_{idx}"] = elt.id
                            self.env[elt.id] = (extractor_id, f"dim_{idx}")
                    self.nodes[extractor_id] = {
                        "block": "shape_extractor",
                        "params": {"_output_aliases": output_aliases} if output_aliases else {},
                    }
                    if src_node:
                        self.edges.append(f"{src_node}.{src_port} -> {extractor_id}.in")
                    continue

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
            elif isinstance(stmt, ast.If):
                cond = self._eval_condition(stmt.test)
                if cond is True:
                    self._parse_statements(stmt.body)
                elif cond is False:
                    self._parse_statements(stmt.orelse)
                else:
                    self._parse_statements(stmt.body)
            elif isinstance(stmt, ast.For):
                # Unroll for loops over module lists: for blk in self.blocks: x = blk(x)
                if isinstance(stmt.iter, ast.Attribute) and isinstance(stmt.iter.value, ast.Name) and stmt.iter.value.id == "self":
                    mod_list = self.layer_instances.get(stmt.iter.attr)
                    if mod_list and mod_list.get("block") == "module_list":
                        target_name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                        for sub_idx, sub_layer in enumerate(mod_list.get("layers", [])):
                            if target_name:
                                self.layer_instances[target_name] = sub_layer
                            self._parse_statements(stmt.body)
                            if target_name and target_name in self.layer_instances:
                                del self.layer_instances[target_name]

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
        # 0. Local layer instance call (e.g. unrolled loop variable blk(x))
        if isinstance(call.func, ast.Name) and call.func.id in self.layer_instances:
            layer_info = self.layer_instances[call.func.id]
            block_id = layer_info.get("block", "custom_module")
            custom_mod_id = layer_info.get("custom_module_id")
            params = dict(layer_info.get("params", {}))
            base_name = call.func.id or target_hint or block_id
            node_id = self._next_node_id(base_name)

            node_entry: Dict[str, Any] = {"block": block_id}
            if custom_mod_id:
                node_entry["custom_module_id"] = custom_mod_id
            if layer_info.get("class_name"):
                node_entry["label"] = layer_info["class_name"]
            if params:
                node_entry["params"] = params
            if target_hint and not target_hint.startswith("x_"):
                node_entry["var_name"] = target_hint
            self.nodes[node_id] = node_entry

            for idx, arg in enumerate(call.args):
                if isinstance(arg, ast.Constant) and arg.value is None:
                    continue
                src_node, src_port = self._parse_expr(arg)
                if src_node:
                    target_handle = "in" if idx == 0 else f"in_{idx + 1}"
                    self.edges.append(f"{src_node}.{src_port} -> {node_id}.{target_handle}")

            for kw in call.keywords:
                if kw.arg and kw.value:
                    src_node, src_port = self._parse_expr(kw.value)
                    if src_node:
                        self.edges.append(f"{src_node}.{src_port} -> {node_id}.{kw.arg}")

            return (node_id, "out")

        # 1. Calls on self: self.fc(x), self.layer1(x), self.backbone.layer1(x), self.features[0].conv(x)
        chain = self._extract_self_chain(call.func)
        if chain:
            block_id, custom_mod_id, params = self._resolve_chain_layer(chain)

            # Determine node naming
            if len(chain) == 1:
                raw_name = str(chain[0])
                if raw_name.startswith("layer_"):
                    clean_name = raw_name[6:]
                elif raw_name.startswith("custom_"):
                    clean_name = raw_name[7:]
                else:
                    clean_name = raw_name
                base_name = clean_name or target_hint or block_id
            else:
                base_name = "_".join(str(t) for t in chain)

            node_id = self._next_node_id(base_name)

            node_entry: Dict[str, Any] = {"block": block_id}
            if custom_mod_id:
                node_entry["custom_module_id"] = custom_mod_id
            linfo = self.layer_instances.get(chain[0], {}) if len(chain) == 1 else {}
            if linfo.get("class_name"):
                node_entry["label"] = linfo["class_name"]
            if params:
                node_entry["params"] = dict(params)
            if target_hint and not target_hint.startswith("x_"):
                node_entry["var_name"] = target_hint
            elif len(chain) == 1 and not str(chain[0]).startswith("layer_") and not str(chain[0]).startswith("custom_"):
                node_entry["var_name"] = str(chain[0])

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

                if method in ("reshape", "view"):
                    if len(call.args) > 1:
                        params["shape"] = tuple(_eval_ast_literal(a, self.init_var_names) for a in call.args)
                    elif len(call.args) == 1:
                        params["shape"] = _eval_ast_literal(call.args[0], self.init_var_names)
                    for a in call.args:
                        if isinstance(a, ast.Name) and a.id in self.env:
                            src_n, src_p = self.env[a.id]
                            edge_str = f"{src_n}.{src_p} -> {node_id}.{src_p}"
                            if edge_str not in self.edges:
                                self.edges.append(edge_str)
                elif method == "permute":
                    params["dims"] = tuple(_eval_ast_literal(a, self.init_var_names) for a in call.args)
                else:
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
        source_path = py_path_or_code
    else:
        code_str = py_path_or_code
        if output_path:
            stem = os.path.basename(output_path).replace(".ir.json", "").replace(".json", "")
        else:
            stem = "model"
        source_path = None

    decompiler = PyTorchASTDecompiler(workspace_dir=workspace_dir)
    ir_dict = decompiler.decompile_source(code_str, file_stem=stem, source_path=source_path)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ir_dict, f, indent=2)

    # If multiple classes decompiled, save submodule IRs
    base_dir = ir_dir or (os.path.dirname(os.path.abspath(output_path)) if output_path else None)
    if not base_dir and workspace_dir:
        base_dir = os.path.join(workspace_dir, "ir")

    if base_dir and len(decompiler.all_irs) > 1:
        modules_dir = os.path.join(base_dir, "modules")
        os.makedirs(modules_dir, exist_ok=True)
        for cls_stem, sub_ir in decompiler.all_irs.items():
            if sub_ir is not ir_dict:
                # Use the snake_case name from the IR payload itself if available,
                # falling back to the class-name-lowered key only as a last resort.
                ir_name = sub_ir.get("name", "") if isinstance(sub_ir, dict) else ""
                file_name = ir_name if ir_name else os.path.basename(cls_stem)
                sub_path = os.path.join(modules_dir, f"{file_name}.ir.json")
                with open(sub_path, "w", encoding="utf-8") as f:
                    json.dump(sub_ir, f, indent=2)

    return ir_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArchIDE PyTorch AST Decompiler")
    parser.add_argument("source", help="Path to Python file containing nn.Module definition(s)")
    parser.add_argument("--output", "-o", help="Output path for root Agentic IR (.ir.json or .json)", default=None)
    parser.add_argument("--workspace", "-w", help="Workspace directory for modular submodules and graphs", default=None)
    parser.add_argument("--compile", "-c", action="store_true", help="Also compile IR into ArchIDE visual graph (.arch)")

    args = parser.parse_args()

    print(f"Decompiling {args.source}...")
    decompiler = PyTorchASTDecompiler(workspace_dir=args.workspace)
    with open(args.source, "r", encoding="utf-8") as f:
        code_str = f.read()

    file_stem = os.path.splitext(os.path.basename(args.source))[0]
    all_irs = decompiler.decompile_all_classes(code_str, file_stem=file_stem, source_path=args.source)

    print(f"Discovered {len(all_irs)} classes: {list(all_irs.keys())}")

    root_ir = decompile_python_to_ir(
        args.source,
        output_path=args.output,
        workspace_dir=args.workspace,
    )

    if args.output:
        print(f"Saved root IR to: {args.output}")

    if args.compile:
        from agent_compiler import AgentGraphCompiler

        for k, ir_data in all_irs.items():
            comp = AgentGraphCompiler(ir_data, workspace_dir=args.workspace, all_irs=all_irs)
            arch = comp.compile()
            if args.workspace:
                out_arch_dir = os.path.join(args.workspace, "graphs" if k == file_stem.lower() else os.path.join("graphs", "modules"))
                os.makedirs(out_arch_dir, exist_ok=True)
                out_arch_path = os.path.join(out_arch_dir, f"{k}.arch")
                with open(out_arch_path, "w", encoding="utf-8") as af:
                    json.dump(arch, af, indent=2)
                print(f"Compiled {k} -> {out_arch_path}")
