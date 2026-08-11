import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compiler import topological_sort, generate_pytorch_code, ShapeError
from models import Node, Edge, NodeData

def test_topological_sort_success():
    nodes = [
        Node(id="n1", data=NodeData(label="In")),
        Node(id="n2", data=NodeData(label="Hidden")),
        Node(id="n3", data=NodeData(label="Out"))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n3", targetHandle="in")
    ]
    
    sorted_nodes = topological_sort(nodes, edges)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]

def test_topological_sort_cycle():
    nodes = [
        Node(id="n1", data=NodeData(label="A")),
        Node(id="n2", data=NodeData(label="B"))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n1", targetHandle="in")
    ]
    
    with pytest.raises(ValueError, match="Cycle detected"):
        topological_sort(nodes, edges)

def test_pytorch_execution():
    """Domain C: PyTorch Compiler Syntax Validation"""
    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(2, 10)"})),
        Node(id="n2", data=NodeData(block_id="linear", label="Linear", paramValues={"in_features": 10, "out_features": 5})),
        Node(id="n3", data=NodeData(block_id="relu", label="ReLU", paramValues={})),
        Node(id="n4", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n3", targetHandle="in"),
        Edge(id="e3", source="n3", sourceHandle="out", target="n4", targetHandle="in")
    ]
    
    sorted_nodes = topological_sort(nodes, edges)
    code = generate_pytorch_code(sorted_nodes, edges)
    
    # 1. Syntax Check (Ensures it is valid Python code)
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None
    
    # 2. String assertions (Since we aren't installing PyTorch to do a runtime check)
    assert "class Model(nn.Module):" in code
    assert "self.layer_n2 = nn.Linear(10, 5, bias=True)" in code
    assert "self.layer_n3 = nn.ReLU(inplace=False)" in code
    assert "def forward(self, x_input):" in code
    assert "return activated" in code
