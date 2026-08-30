import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compiler import topological_sort, generate_pytorch_code, ShapeError
from models import Node, Edge, NodeData, GraphData

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
    
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files = generate_pytorch_code(graphs, "main")
    code = files["main"]
    
    # 1. Syntax Check (Ensures it is valid Python code)
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None
    
    # 2. String assertions (Since we aren't installing PyTorch to do a runtime check)
    assert "class Model(nn.Module):" in code
    assert "self.layer_n2 = nn.Linear(10, 5, bias=True)" in code
    assert "self.layer_n3 = nn.ReLU(inplace=False)" in code
    assert "def forward(self, x_input):" in code
    assert "return activated" in code


def test_orphan_edges_handled_gracefully():
    """Ensure orphan edges from deleted nodes (e.g. node_102) do not raise KeyError."""
    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(1, 10)"})),
        Node(id="n2", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        # Orphan edges referencing non-existent nodes
        Edge(id="e_orphan_src", source="node_102", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e_orphan_tgt", source="n1", sourceHandle="out", target="node_999", targetHandle="in"),
    ]
    sorted_nodes = topological_sort(nodes, edges)
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files = generate_pytorch_code(graphs, "main")
    code = files["main"]
    assert "def forward(self, x_input):" in code
    assert "return x_input" in code


def test_variadic_add_multi_input():
    """Ensure variadic AddBlock accepts multiple incoming connections into a single port."""
    nodes = [
        Node(id="in1", data=NodeData(block_id="input", label="Input 1", paramValues={"shape": "(1, 64)"})),
        Node(id="in2", data=NodeData(block_id="input", label="Input 2", paramValues={"shape": "(1, 64)"})),
        Node(id="in3", data=NodeData(block_id="input", label="Input 3", paramValues={"shape": "(1, 64)"})),
        Node(id="add1", data=NodeData(block_id="add", label="Add", paramValues={})),
        Node(id="out1", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="in1", sourceHandle="out", target="add1", targetHandle="in"),
        Edge(id="e2", source="in2", sourceHandle="out", target="add1", targetHandle="in"),
        Edge(id="e3", source="in3", sourceHandle="out", target="add1", targetHandle="in"),
        Edge(id="e4", source="add1", sourceHandle="out", target="out1", targetHandle="in"),
    ]
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files = generate_pytorch_code(graphs, "main")
    code = files["main"]
    assert "x_input_1 + x_input_2 + x_input_3" in code
    assert "return sum" in code


def test_multi_output_aggregation():
    """Ensure multiple tensors connected to Output or multiple Output blocks emit unified return tuple."""
    nodes = [
        Node(id="in1", data=NodeData(block_id="input", label="Input 1", paramValues={"shape": "(1, 10)"})),
        Node(id="in2", data=NodeData(block_id="input", label="Input 2", paramValues={"shape": "(1, 20)"})),
        Node(id="out1", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="in1", sourceHandle="out", target="out1", targetHandle="in"),
        Edge(id="e2", source="in2", sourceHandle="out", target="out1", targetHandle="in"),
    ]
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files = generate_pytorch_code(graphs, "main")
    code = files["main"]
    assert "return x_input_1, x_input_2" in code

