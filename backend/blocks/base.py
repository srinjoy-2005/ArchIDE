from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import os
import sys

# Ensure we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import BlockDef

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
