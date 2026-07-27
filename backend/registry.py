from models import BlockDef, PortDef, ParamDef

REGISTRY = [
    BlockDef(
        id="input",
        name="Input",
        category="Core Layers",
        color="#10b981",
        is_functional=True,
        inputs=[],
        outputs=[PortDef(id="out", name="Output")],
        params=[
            ParamDef(name="shape", type="string", default="(1, 3, 224, 224)")
        ]
    ),
    BlockDef(
        id="gourav",
        name="Gourav",
        category="Core Layers",
        color="#10b981",
        is_functional=True,
        inputs=[],
        outputs=[PortDef(id="out", name="Output")],
        params=[
            ParamDef(name="shape", type="string", default="(1, 3, 224, 224)")
        ]
    ),
    BlockDef(
        id="output",
        name="Output",
        category="Core Layers",
        color="#f43f5e", # Rose color to stand out
        is_functional=True,
        inputs=[PortDef(id="in", name="Return Value")],
        outputs=[], # No outputs on the right!
        params=[]
    ),
    BlockDef(
        id="linear", 
        name="Linear", 
        category="Core Layers", 
        color="#3b82f6",
        is_functional=False,
        inputs=[PortDef(id="in", name="Input")],
        outputs=[PortDef(id="out", name="Output")],
        params=[
            ParamDef(name="in_features", type="int", default=128),
            ParamDef(name="out_features", type="int", default=64)
        ]
    ),
    BlockDef(
        id="conv2d",
        name="Conv2D",
        category="Core Layers",
        color="#3b82f6",
        is_functional=False,
        inputs=[PortDef(id="in", name="Input")],
        outputs=[PortDef(id="out", name="Output")],
        params=[
            ParamDef(name="in_channels", type="int", default=3),
            ParamDef(name="out_channels", type="int", default=16),
            ParamDef(name="kernel_size", type="int", default=3)
        ]
    ),
    BlockDef(
        id="relu",
        name="ReLU",
        category="Activations",
        color="#f59e0b",
        is_functional=False,
        inputs=[PortDef(id="in", name="Input")],
        outputs=[PortDef(id="out", name="Output")],
        params=[]
    ),
    BlockDef(
        id="softmax",
        name="Softmax",
        category="Activations",
        color="#f59e0b",
        is_functional=False,
        inputs=[PortDef(id="in", name="Input")],
        outputs=[PortDef(id="out", name="Output")],
        params=[
            ParamDef(name="dim", type="int", default=1)
        ]
    ),
    BlockDef(
        id="add",
        name="Add",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[
            PortDef(id="in_0", name="Input 1"),
            PortDef(id="in_1", name="Input 2")
        ],
        outputs=[PortDef(id="out", name="Out")],
        params=[
            ParamDef(name="num_inputs", type="int", default=2)
        ]
    ),
    BlockDef(
        id="split",
        name="Split (Chunk)",
        category="Tensor Ops",
        color="#ef4444",
        is_functional=True,
        inputs=[
            PortDef(id="in", name="Input Tensor")
        ],
        outputs=[
            PortDef(id="out_1", name="Chunk 1"),
            PortDef(id="out_2", name="Chunk 2")
        ],
        params=[
            ParamDef(name="chunks", type="int", default=2),
            ParamDef(name="dim", type="int", default=-1)
        ]
    ),
    BlockDef(
        id="sub",
        name="Subtract",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="mul",
        name="Multiply",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="div",
        name="Divide",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="pow",
        name="Power",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[PortDef(id="in_a",name="Base")],
        outputs=[PortDef(id="out",name="Out")],
        params=[ParamDef(name="exponent", type="float", default=2.0)]
    ),
    BlockDef(
        id="sin",
        name="Sin",
        category="Trig",
        color="#ec4899",
        is_functional=True,
        inputs=[PortDef(id="in",name="Input")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="cos",
        name="Cos",
        category="Trig",
        color="#ec4899",
        is_functional=True,
        inputs=[PortDef(id="in",name="Input")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="matmul",
        name="MatMul",
        category="Tensor Ops",
        color="#8b5cf6",
        is_functional=True,
        inputs=[PortDef(id="in_a",name="A"), PortDef(id="in_b",name="B")],
        outputs=[PortDef(id="out",name="Out")],
        params=[]
    ),
    BlockDef(
        id="unsqueeze",
        name="Unsqueeze",
        category="Tensor Ops",
        color="#ef4444",
        is_functional=True,
        inputs=[PortDef(id="in",name="Input")],
        outputs=[PortDef(id="out",name="Out")],
        params=[ParamDef(name="dim", type="int", default=0)]
    ),
    BlockDef(
        id="cat",
        name="Concat",
        category="Tensor Ops",
        color="#ef4444",
        is_functional=True,
        inputs=[
            PortDef(id="in_a",name="A"),
            PortDef(id="in_b",name="B"),
            PortDef(id="in_c",name="C")
        ],
        outputs=[PortDef(id="out",name="Out")],
        params=[ParamDef(name="dim", type="int", default=-1)]
    ),
    BlockDef(
        id="arange",
        name="Arange",
        category="Generators",
        color="#14b8a6",
        is_functional=True,
        inputs=[],
        outputs=[PortDef(id="out",name="Out")],
        params=[
            ParamDef(name="start", type="int", default=0),
            ParamDef(name="end", type="int", default=10),
            ParamDef(name="step", type="int", default=1)
        ]
    )
]
