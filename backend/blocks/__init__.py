from .base import BaseBlock
from .core import InputBlock, OutputBlock, LinearBlock, Conv2DBlock
from .activations import ReLUBlock, SoftmaxBlock
from .tensor_ops import (
    AddBlock, SubBlock, MulBlock, DivBlock, PowBlock, 
    MatMulBlock, UnsqueezeBlock, CatBlock, SplitBlock
)
from .trig import SinBlock, CosBlock
from .generators import ArangeBlock
from typing import Dict, List
from models import BlockDef

# Instantiate all available blocks
_BLOCK_INSTANCES = [
    InputBlock(),
    OutputBlock(),
    LinearBlock(),
    Conv2DBlock(),
    ReLUBlock(),
    SoftmaxBlock(),
    AddBlock(),
    SplitBlock(),
    SubBlock(),
    MulBlock(),
    DivBlock(),
    PowBlock(),
    SinBlock(),
    CosBlock(),
    MatMulBlock(),
    UnsqueezeBlock(),
    CatBlock(),
    ArangeBlock()
]

# Create a mapping for O(1) lookups
_BLOCK_MAP: Dict[str, BaseBlock] = {
    block.definition.id: block for block in _BLOCK_INSTANCES
}

def get_all_blocks() -> List[BaseBlock]:
    """Return a list of all instantiated blocks."""
    return _BLOCK_INSTANCES

def get_block_by_id(block_id: str) -> BaseBlock:
    """Return a block instance by its ID."""
    return _BLOCK_MAP.get(block_id)

def get_all_block_defs() -> List[BlockDef]:
    """Return a list of all BlockDefs for the registry endpoint."""
    return [block.definition for block in _BLOCK_INSTANCES]
