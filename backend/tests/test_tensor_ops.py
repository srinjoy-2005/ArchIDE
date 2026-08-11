import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blocks.tensor_ops import broadcast_shapes

def test_broadcast_shapes_valid():
    assert broadcast_shapes((1, 3), (4, 1)) == (4, 3)
    assert broadcast_shapes((3, 1, 5), (1, 4, 1)) == (3, 4, 5)
    assert broadcast_shapes((4, 3), (3,)) == (4, 3)
    assert broadcast_shapes((256, 256, 3), (3,)) == (256, 256, 3)

def test_broadcast_shapes_any():
    assert broadcast_shapes(("ANY",), (4, 3)) == (4, 3)
    assert broadcast_shapes((4, 3), ("ANY",)) == (4, 3)

def test_broadcast_shapes_invalid():
    with pytest.raises(ValueError, match="not broadcastable"):
        broadcast_shapes((4, 3), (3, 4))
    
    with pytest.raises(ValueError, match="not broadcastable"):
        broadcast_shapes((2, 5, 3), (2, 4, 3))

if __name__ == "__main__":
    pytest.main(["-v", __file__])
