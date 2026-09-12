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
    with pytest.raises(ValueError, match="Conv2D: Negative spatial dimension"):
        block.infer_shapes({"in": (4, 3, 32, 32)}, params)

    # Edge values: divisor stride=0, kernel_size=0, dilation=0, groups=0
    with pytest.raises(ValueError, match="Conv2D: stride must be greater than 0"):
        block.infer_shapes({"in": (4, 3, 32, 32)}, {"kernel_size": 3, "stride": 0, "in_channels": 3, "out_channels": 16})

    with pytest.raises(ValueError, match="Conv2D: kernel_size must be greater than 0"):
        block.infer_shapes({"in": (4, 3, 32, 32)}, {"kernel_size": 0, "stride": 1, "in_channels": 3, "out_channels": 16})

    with pytest.raises(ValueError, match="Conv2D: dilation must be greater than 0"):
        block.infer_shapes({"in": (4, 3, 32, 32)}, {"kernel_size": 3, "stride": 1, "dilation": 0, "in_channels": 3, "out_channels": 16})

    with pytest.raises(ValueError, match="Conv2D: groups must be greater than 0"):
        block.infer_shapes({"in": (4, 3, 32, 32)}, {"kernel_size": 3, "stride": 1, "groups": 0, "in_channels": 3, "out_channels": 16})

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

    # 7. Divisor stride=0 and kernel_size=0
    with pytest.raises(ValueError, match="MaxPool2D: stride must be greater than 0"):
        block.infer_shapes({"in": (1, 16, 224, 224)}, {"kernel_size": 2, "stride": 0})

    with pytest.raises(ValueError, match="MaxPool2D: kernel_size must be greater than 0"):
        block.infer_shapes({"in": (1, 16, 224, 224)}, {"kernel_size": 0, "stride": 1})


def test_avgpool2d_block_inference():
    from blocks.pooling import AvgPool2DBlock
    block = AvgPool2DBlock()

    # Standard and string-encoded params
    out = block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": "2", "stride": "2", "padding": "0"})
    assert out["out"] == (1, 8, 16, 16)

    # Missing stride defaults to kernel_size
    out = block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": 2})
    assert out["out"] == (1, 8, 16, 16)

    # Divisor stride=0 and kernel_size=0
    with pytest.raises(ValueError, match="AvgPool2D: stride must be greater than 0"):
        block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": 2, "stride": 0})

    with pytest.raises(ValueError, match="AvgPool2D: kernel_size must be greater than 0"):
        block.infer_shapes({"in": (1, 8, 32, 32)}, {"kernel_size": 0, "stride": 1})


def test_edge_zero_values_other_blocks():
    from blocks.core import LinearBlock
    from blocks.tensor_ops import SplitBlock
    from blocks.generators import ArangeBlock

    linear = LinearBlock()
    with pytest.raises(ValueError, match="Linear: out_features must be greater than 0"):
        linear.infer_shapes({"in": (1, 128)}, {"in_features": 128, "out_features": 0})

    with pytest.raises(ValueError, match="Linear: in_features must be greater than 0"):
        linear.infer_shapes({"in": (1, 128)}, {"in_features": 0, "out_features": 10})

    split = SplitBlock()
    with pytest.raises(ValueError, match="Split: chunks must be greater than 0"):
        split.infer_shapes({"in": (1, 10)}, {"chunks": 0})

    arange = ArangeBlock()
    with pytest.raises(ValueError, match="Arange: step must not be 0"):
        arange.infer_shapes({}, {"start": 0, "end": 10, "step": 0})
 
