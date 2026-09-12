from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import os
import sys

# Ensure we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import BlockDef

def parse_int_or_tuple2d(val: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if val is None or val == "" or val == "None":
        return default
    if isinstance(val, (int, float)):
        v = int(val)
        return (v, v)
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            v = int(val[0])
            return (v, v)
        if len(val) >= 2:
            return (int(val[0]), int(val[1]))
    if isinstance(val, str):
        clean = "".join(c for c in val if c.isdigit() or c in (',', '-'))
        parts = [int(p) for p in clean.split(',') if p]
        if len(parts) == 1:
            return (parts[0], parts[0])
        elif len(parts) >= 2:
            return (parts[0], parts[1])
    return default

class BaseBlock(ABC):
    @property
    @abstractmethod
    def definition(self) -> BlockDef:
        """Returns the Pydantic schema for the frontend registry."""
        pass

    @abstractmethod
    def infer_shapes(self, input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]:
        """Calculates output shapes given input port shapes and block parameters."""
        pass

    @abstractmethod
    def emit_init(self, node_id: str, params: Dict[str, Any]) -> str:
        """Generates the nn.Module instantiation code (if stateful)."""
        pass

    def docs(self) -> Dict[str, str]:
        """Returns the documentation dictionary for this block."""
        return {
            "intro": f"The {self.definition.name} block.",
            "details": "Documentation for this block has not been written yet."
        }

    @abstractmethod
    def emit_forward(self, node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str:
        """Generates the functional execution code."""
        pass
