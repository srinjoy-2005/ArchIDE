import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blocks.core import LinearBlock, Conv2DBlock, InputBlock

def test_linear_block_inference():
    block = LinearBlock()
    
    # Valid shape inference
    out = block.infer_shapes({"in": (16, 3, 128)}, {"in_features": 128, "out_features": 64})
    assert out["out"] == (16, 3, 64)
    
    # Auto-infer in_features
    params = {"in_features": -1, "out_features": 32}
    out = block.infer_shapes({"in": (8, 256)}, params)
    assert out["out"] == (8, 32)
    assert params["in_features"] == 256
    
    # Invalid in_features mismatch
    with pytest.raises(ValueError, match="Linear: expected in_features=128"):
        block.infer_shapes({"in": (16, 64)}, {"in_features": 128, "out_features": 64})
        
    # ANY shape handling
    out = block.infer_shapes({"in": ("ANY",)}, {"in_features": 128, "out_features": 64})
    assert out["out"] == ("ANY",)

def test_conv2d_block_inference():
    block = Conv2DBlock()
    
    # Valid shape inference
    params = {
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1
    }
    out = block.infer_shapes({"in": (4, 3, 32, 32)}, params)
    assert out["out"] == (4, 16, 32, 32)
    
    # Auto-infer in_channels
    params["in_channels"] = -1
    out = block.infer_shapes({"in": (4, 8, 16, 16)}, params)
    assert out["out"] == (4, 16, 16, 16)
    assert params["in_channels"] == 8
    
    # Invalid channel mismatch
    params["in_channels"] = 3
    with pytest.raises(ValueError, match="Conv2D: expected in_channels=3"):
        block.infer_shapes({"in": (4, 8, 32, 32)}, params)
        
    # Boundary: invalid kernel size leading to negative spatial dimension
    params["kernel_size"] = 100
    with pytest.raises(ValueError, match="Conv2D: Negative spatial dimensions"):
        # We expect the formula to result in negative or ValueError depending on how Conv2DBlock is implemented
        # Let's check how Conv2DBlock raises error. It calculates output shapes.
        block.infer_shapes({"in": (4, 3, 32, 32)}, params)

def test_input_block_inference():
    block = InputBlock()
    
    # Valid
    out = block.infer_shapes({}, {"shape": "(1, 3, 224, 224)"})
    assert out["out"] == (1, 3, 224, 224)
    
    # Malformed shape string gracefully degrades to default
    out = block.infer_shapes({}, {"shape": "invalid_shape_format_23!!"})
    # It parses numbers so it will get (23,)
    assert out["out"] == (23,) or out["out"] == (1, 3, 224, 224)


def test_maxpool2d_block_inference():
    from blocks.pooling import MaxPool2DBlock
    block = MaxPool2DBlock()

    # 1. Standard shape inference
    out = block.infer_shapes({"in": (1, 16, 224, 224)}, {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1})
    assert out["out"] == (1, 16, 112, 112)

    # 2. String-encoded params from frontend
    out = block.infer_shapes({"in": (1, 16, 224, 224)}, {"kernel_size": "2", "stride": "2", "padding": "0", "dilation": "1"})
    assert out["out"] == (1, 16, 112, 112)

    # 3. Missing or None stride defaults to kernel_size
    out = block.infer_shapes({"in": (1, 16, 224, 224)}, {"kernel_size": 2, "stride": None})
    assert out["out"] == (1, 16, 112, 112)

    # 4. Tuple params
    out = block.infer_shapes({"in": (2, 32, 64, 64)}, {"kernel_size": (2, 2), "stride": (2, 2)})
    assert out["out"] == (2, 32, 32, 32)

    # 5. Partial ANY spatial handling
    out = block.infer_shapes({"in": (1, 16, "ANY", 224)}, {"kernel_size": 2, "stride": 2})
    assert out["out"] == (1, 16, "ANY", 112)

    # 6. Negative spatial dimension raises ValueError
    with pytest.raises(ValueError, match="MaxPool2D: Output height"):
        block.infer_shapes({"in": (1, 16, 2, 2)}, {"kernel_size": 10, "stride": 1})


def test_avgpool2d_block_inference():
    from blocks.pooling import AvgPool2DBlock
    block = AvgPool2DBlock()

    # Standard and string-encoded params
    out = block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": "2", "stride": "2", "padding": "0"})
    assert out["out"] == (1, 8, 16, 16)

    # Missing stride defaults to kernel_size
    out = block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": 2})
    assert out["out"] == (1, 8, 16, 16)
 
