from pydantic import BaseModel
from typing import List, Any, Optional, Dict

class Edge(BaseModel):
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str

class NodeData(BaseModel):
    block_id: str = ""
    label: str
    is_functional: bool = False
    paramValues: dict = {}
    varName: str = ""  # optional user-defined output variable name
    custom_module_id: str = ""

class Node(BaseModel):
    id: str
    data: NodeData
    position: Optional[Dict[str, float]] = None

class ParamDef(BaseModel):
    name: str
    type: str                   # "int", "float", "string", "bool"
    default: Any
    read_only: bool = False     # If True, shown greyed-out in UI (e.g. inferred shapes)
    section: str = "basic"      # "shape" | "basic" | "advanced"
    description: str = ""       # Tooltip text shown in the UI

class GraphData(BaseModel):
    name: str = "Model"
    parameters: List[ParamDef] = []
    nodes: List[Node]
    edges: List[Edge]

class CompileRequest(BaseModel):
    main_graph_id: str
    graphs: Dict[str, GraphData]

class CheckRequest(BaseModel):
    main_graph_id: str
    graphs: Dict[str, GraphData]

class FolderData(BaseModel):
    id: str
    name: str
    parentId: Optional[str] = None
    isExpanded: bool = True

class ProjectFileData(BaseModel):
    id: str
    name: str
    parentId: Optional[str] = None
    path: Optional[str] = None
    parameters: List[ParamDef] = []
    nodes: List[Node] = []
    edges: List[Edge] = []

class ArchIDEProject(BaseModel):
    name: str = "MyModelProject"
    version: str = "1.0.0"
    entry_point: str = "main"
    folders: List[FolderData] = []
    files: List[ProjectFileData] = []

class PortDef(BaseModel):
    id: str
    name: str
    type: str = "tensor"
    # if true, the frontend lets the user connect multiple edges
    # to this single port 
    is_list: bool = False
    # Suggested Python variable name for this port's output tensor
    var_hint: Optional[str] = None

class BlockDef(BaseModel):
    id: str
    name: str
    category: str
    color: str
    is_functional: bool
    inputs: List[PortDef]
    outputs: List[PortDef]
    params: List[ParamDef]
