from pydantic import BaseModel
from typing import List, Any

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

class PortDef(BaseModel):
    id: str
    name: str # User-facing name
    type: str = "tensor"
    # if true, the frontend lets the user connect multiple edges
    # to this single port 
    is_list: bool = False  
    
class ParamDef(BaseModel):
    name: str
    type: str
    default: Any

class BlockDef(BaseModel):
    id: str
    name: str
    category: str
    color: str

    # Tells the backend: if True, put in forward()
    # If False, instantiate in __init__()
    is_functional: bool
    inputs: List[PortDef]
    outputs: List[PortDef]

    params: List[ParamDef]
