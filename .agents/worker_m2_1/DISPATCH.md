## 2026-09-20T08:06:35Z
You are Worker 2 executing Milestone M2 for ArchIDE.
Your working directory is: d:\ML\ArchIDE\.agents\worker_m2_1
Original user request reference: d:\ML\ArchIDE\.agents\ORIGINAL_REQUEST.md
Project specification: d:\ML\ArchIDE\.agents\PROJECT.md
Survey & Implementation Specification: d:\ML\ArchIDE\.agents\explorer_survey_2\handoff.md

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Mission:
Implement Requirement R2: Multilevel Files & Multi-Class Decompilation and Deep Object Attribute Chains.

Write Ownership (You exclusively own and may edit these files):
- `backend/python_decompiler.py`
- `backend/agent_compiler.py`
- `backend/tests/test_r2_multiclass_and_attributes.py`

Required Implementation Details (see d:\ML\ArchIDE\.agents\explorer_survey_2\handoff.md for full analysis and snippets):
1. Deep Object Attribute Chain Resolver in `backend/python_decompiler.py`:
   - Implement `_extract_self_chain(node: ast.AST) -> Optional[List[Union[str, int]]]`:
     - Unpacks nested `ast.Attribute` and `ast.Subscript` chains originating at `self`.
     - Examples: `self.backbone.layer1(x)` -> `["backbone", "layer1"]`; `self.features[0].conv(x)` -> `["features", 0, "conv"]`.
   - In `_parse_call`:
     - Unify calls on self: if `chain = self._extract_self_chain(call.func)`:
     - Check if chain corresponds to known layer instance or custom module.
     - Generate a valid node ID (e.g. `self._next_node_id("_".join(str(t) for t in chain))`).
     - Wire arguments (`call.args`, `call.keywords`) to input ports (`node_id.in`, `node_id.in_2`, etc.).
     - Record node in `self.nodes` and return `(node_id, "out")`.
     - This ensures `self.env` receives a valid source node instead of `None`, keeping the dataflow unbroken.
   - For `Sequential`, record child layers in `layer_instances[attr]["layers"]`.
2. Multi-Class Extraction & Dependency Topological Sort in `backend/python_decompiler.py`:
   - In `decompile_source`:
     - Find all `ast.ClassDef` in `tree.body` that inherit from `Module` (or define `forward`).
     - Build internal dependency graph by inspecting constructor calls in each class's `__init__` for references to other local classes.
     - Topologically sort classes so leaf submodules are processed first and parent/root modules are processed next.
     - Decompile each class with isolated state to produce distinct IRs.
     - Provide `decompile_all_classes(code_str: str, file_stem: str = "main") -> Dict[str, Dict[str, Any]]` returning `{class_name.lower(): ir_dict}`.
     - Keep `decompile_source(...)` returning the root module's IR (with `self.all_irs` storing all classes) for backward compatibility.
   - Submodule References:
     - When parent class instantiates local submodule (`self.block = BasicBlock(...)`), parent IR records `"block": "custom_module"`, `"custom_module_id": "modules/basicblock"` (or configured path) and preserves constructor parameter bindings.
3. Recursive Import Resolution in `backend/python_decompiler.py`:
   - In `_parse_imports`:
     - Support `alias.asname` (`from x import Y as Z`).
     - Resolve relative imports (`stmt.level > 0`) and local absolute imports relative to source file directory or workspace.
     - When a local `.py` file is discovered, recursively decompile it (using a `visited_paths: Set[str]` guard to prevent recursion loops).
     - Register imported classes in `self.imports[alias] = module_id`.
4. Agent Compiler Port Loading in `backend/agent_compiler.py`:
   - Update `_load_custom_module_ports` so that if target file is `.ir.json` (where `nodes` is a dictionary: `{"in": {"block": "input"}, ...}`), it properly discovers input and output ports rather than assuming `nodes` is always a list.
5. Comprehensive Test Suite in `backend/tests/test_r2_multiclass_and_attributes.py`:
   - Test single Python file containing multiple `nn.Module` classes (e.g. `BasicBlock` and `ResNet`, or `DoubleConv` and `UNet`), verifying distinct IRs are created with custom_module linkages.
   - Test deep attribute calls (`self.backbone.layer1(x)`, `self.features[0].conv(x)`), verifying unbroken dataflow edges from input to output.
   - Test cross-file recursive imports.
   - Verify generated IRs compile with `compile_ir(ir, validate=True)` or `generate_pytorch_code`.

Verification Commands:
- `pytest backend/tests/test_r2_multiclass_and_attributes.py`
- `pytest backend/tests/test_r1_blocks_and_scalars.py`
- `pytest backend/tests/ --ignore=backend/tests/test_real_world_models.py`
- `npx tsc --noEmit`
