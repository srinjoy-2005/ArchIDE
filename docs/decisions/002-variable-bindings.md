# ADR 002: Variable Bindings

## Date
September 2026

## Context
Hardcoding parameter values (like `in_channels=64`) inside blocks restricts model flexibility. We needed a way to declare global/file-level variables and bind them to node parameters so that generated PyTorch code exposes arguments in `__init__` or module-level constants.

## Decision
1. **Per-File Variables**: Variables are stored natively on the `GraphFile` state array (not globally).
2. **String Prefix Binding**: Node parameter inputs bind variables via the `"@var:<name>"` string prefix stored natively in `paramValues`.
3. **Compiler Resolution**: The backend compiler extracts variables, splits them into `init_param` (generating `__init__(self, channels=64)`) and `local_const` (generating `CHANNELS = 64`), and performs a string-replacement pass during code emission.
4. **Active Canvas Trimming**: To ensure safe deletion of variables, `useReactFlow().setNodes()` is called immediately to clear dangling `@var:` bindings from the active canvas, allowing the debounced auto-save to propagate the cleansed state to the backend safely.

## Consequences
- **Positive**: Extremely flexible dynamic codegen matching standard PyTorch practices. Visual indication in the UI via purple pills.
- **Negative**: Requires strict compile guards to catch unresolved `@var:` strings if variables are deleted improperly or edge cases are missed.
