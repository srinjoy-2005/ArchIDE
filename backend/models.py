from pydantic import BaseModel
from typing import List, Any, Optional

class Edge(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str

class NodeData(BaseModel):
    block_id: str = ""
    label: str
    is_functional: bool = False
    paramValues: dict = {}

class Node(BaseModel):
    id: str
    data: NodeData

class CompileRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class CheckRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class PortDef(BaseModel):
    id: str
    name: str
    type: str = "tensor"
    # if true, the frontend lets the user connect multiple edges
    # to this single port 
    is_list: bool = False
    # Suggested Python variable name for this port's output tensor
    var_hint: Optional[str] = None

class ParamDef(BaseModel):
    name: str
    type: str                   # "int", "float", "string", "bool"
    default: Any
    read_only: bool = False     # If True, shown greyed-out in UI (e.g. inferred shapes)
    section: str = "basic"      # "shape" | "basic" | "advanced"
    description: str = ""       # Tooltip text shown in the UI

class BlockDef(BaseModel):
    id: str
    name: str
    category: str
    color: str
    is_functional: bool
    inputs: List[PortDef]
    outputs: List[PortDef]
    params: List[ParamDef]
