# ArchIDE: PyTorch Block Catalog & Implementation Status

> [!NOTE]
> **Status**: 🟢 28 Blocks Implemented across 8 Categories.
> Registry module loader: [`backend/blocks/__init__.py`](../../backend/blocks/__init__.py#L16-L64).

---

## 1. Implemented Block Catalog

### 1. Core Layers ([`backend/blocks/core.py`](../../backend/blocks/core.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters & Auto-Inference |
|---|---|---|---|---|
| `input` | [`InputBlock`](../../backend/blocks/core.py#L7-L44) | — | `out` | `shape` (str, default `"(1, 3, 224, 224)"`) |
| `output` | [`OutputBlock`](../../backend/blocks/core.py#L46-L69) | `in` | — | Return collector emitted in model signature |
| `linear` | [`LinearBlock`](../../backend/blocks/core.py#L71-L136) | `in` | `out` (`var_hint="fc_out"`) | `in_features` (int, default `128`, auto-inferred if `-1`), `out_features` (int, default `64`), `bias` (bool) |
| `conv2d` | [`Conv2DBlock`](../../backend/blocks/core.py#L138-L221) | `in` | `out` (`var_hint="conv_feat"`) | `in_channels` (int, default `3`, auto-inferred if `-1`), `out_channels` (int, `16`), `kernel_size` (`3`), `stride` (`1`), `padding` (`0`), `dilation` (`1`), `groups` (`1`), `bias` (bool) |

### 2. Activations ([`backend/blocks/activations.py`](../../backend/blocks/activations.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters |
|---|---|---|---|---|
| `relu` | [`ReLUBlock`](../../backend/blocks/activations.py#L5-L34) | `in` | `out` (`var_hint="activated"`) | `inplace` (bool, default `False`) |
| `softmax` | [`SoftmaxBlock`](../../backend/blocks/activations.py#L36-L65) | `in` | `out` (`var_hint="probs"`) | `dim` (int, default `-1`) |
| `sigmoid` | [`SigmoidBlock`](../../backend/blocks/activations.py#L67-L93) | `in` | `out` (`var_hint="sig_out"`) | None |
| `tanh` | [`TanhBlock`](../../backend/blocks/activations.py#L95-L121) | `in` | `out` (`var_hint="tanh_out"`) | None |

### 3. Pooling ([`backend/blocks/pooling.py`](../../backend/blocks/pooling.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters |
|---|---|---|---|---|
| `maxpool2d` | [`MaxPool2DBlock`](../../backend/blocks/pooling.py#L6-L57) | `in` | `out` (`var_hint="pooled"`) | `kernel_size` (int, `2`), `stride` (`2`), `padding` (`0`), `dilation` (`1`) |
| `avgpool2d` | [`AvgPool2DBlock`](../../backend/blocks/pooling.py#L59-L107) | `in` | `out` (`var_hint="pooled"`) | `kernel_size` (int, `2`), `stride` (`2`), `padding` (`0`) |
| `adaptiveavgpool2d` | [`AdaptiveAvgPool2DBlock`](../../backend/blocks/pooling.py#L109-L151) | `in` | `out` (`var_hint="pooled"`) | `output_size` (str, default `"(1, 1)"`) |

### 4. Normalization & Regularization ([`backend/blocks/normalization.py`](../../backend/blocks/normalization.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters & Auto-Inference |
|---|---|---|---|---|
| `batchnorm2d` | [`BatchNorm2DBlock`](../../backend/blocks/normalization.py#L5-L52) | `in` | `out` (`var_hint="norm_out"`) | `num_features` (int, default `-1`, auto-inferred), `eps` (`1e-5`), `momentum` (`0.1`) |
| `layernorm` | [`LayerNormBlock`](../../backend/blocks/normalization.py#L54-L98) | `in` | `out` (`var_hint="norm_out"`) | `normalized_shape` (str, default `"?"`, auto-inferred from trailing dim), `eps` (`1e-5`) |
| `dropout` | [`DropoutBlock`](../../backend/blocks/normalization.py#L100-L131) | `in` | `out` (`var_hint="dropped"`) | `p` (float, default `0.5`), `inplace` (bool, default `False`) |

### 5. Shape Operations ([`backend/blocks/shape.py`](../../backend/blocks/shape.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters |
|---|---|---|---|---|
| `flatten` | [`FlattenBlock`](../../backend/blocks/shape.py#L5-L57) | `in` | `out` (`var_hint="flat"`) | `start_dim` (int, default `1`), `end_dim` (int, default `-1`) |
| `reshape` | [`ReshapeBlock`](../../backend/blocks/shape.py#L59-L95) | `in` | `out` (`var_hint="reshaped"`) | `shape` (str, default `"(-1,)"`) |

### 6. Multi-Port Tensor Operations & Math ([`backend/blocks/tensor_ops.py`](../../backend/blocks/tensor_ops.py))
| Block ID | Class & Lines | Inputs | Outputs | Shape Logic & Formula |
|---|---|---|---|---|
| `add` | [`AddBlock`](../../backend/blocks/tensor_ops.py#L24-L64) | `in_0`, `in_1` | `out` (`var_hint="sum"`) | Element-wise sum with dynamic broadcasting ([`broadcast_shapes`](../../backend/blocks/tensor_ops.py#L5-L22)) |
| `sub` | [`SubBlock`](../../backend/blocks/tensor_ops.py#L66-L93) | `in_a`, `in_b` | `out` (`var_hint="diff"`) | Element-wise difference with broadcasting |
| `mul` | [`MulBlock`](../../backend/blocks/tensor_ops.py#L95-L122) | `in_a`, `in_b` | `out` (`var_hint="product"`) | Element-wise product with broadcasting |
| `div` | [`DivBlock`](../../backend/blocks/tensor_ops.py#L124-L151) | `in_a`, `in_b` | `out` (`var_hint="quotient"`) | Element-wise division with broadcasting |
| `pow` | [`PowBlock`](../../backend/blocks/tensor_ops.py#L153-L178) | `in_a` | `out` (`var_hint="powered"`) | `torch.pow(in_a, exponent)` with `exponent` (float, default `2.0`) |
| `matmul` | [`MatMulBlock`](../../backend/blocks/tensor_ops.py#L180-L218) | `in_a`, `in_b` | `out` (`var_hint="matmul_out"`) | Matrix multiplication: `(..., M, K) x (..., K, N) -> (..., M, N)` |
| `unsqueeze` | [`UnsqueezeBlock`](../../backend/blocks/tensor_ops.py#L220-L254) | `in` | `out` (`var_hint="unsqueezed"`) | `torch.unsqueeze(in, dim=dim)` with `dim` (int, default `0`) |
| `cat` | [`CatBlock`](../../backend/blocks/tensor_ops.py#L256-L290) | `in_a`, `in_b`, `in_c` | `out` (`var_hint="concat"`) | `torch.cat([in_a, in_b, ...], dim=dim)` with `dim` (int, default `-1`) |
| `split` | [`SplitBlock`](../../backend/blocks/tensor_ops.py#L292-L338) | `in` | `out_1`, `out_2` | `torch.chunk(in, chunks=chunks, dim=dim)` |

### 7. Trigonometric Operations ([`backend/blocks/trig.py`](../../backend/blocks/trig.py))
| Block ID | Class & Lines | Inputs | Outputs | Operation |
|---|---|---|---|---|
| `sin` | [`SinBlock`](../../backend/blocks/trig.py#L5-L29) | `in` | `out` (`var_hint="sin_out"`) | `torch.sin(in)` |
| `cos` | [`CosBlock`](../../backend/blocks/trig.py#L31-L55) | `in` | `out` (`var_hint="cos_out"`) | `torch.cos(in)` |

### 8. Tensor Generators ([`backend/blocks/generators.py`](../../backend/blocks/generators.py))
| Block ID | Class & Lines | Inputs | Outputs | Key Parameters |
|---|---|---|---|---|
| `arange` | [`ArangeBlock`](../../backend/blocks/generators.py#L5-L39) | — | `out` (`var_hint="indices"`) | `start` (int, `0`), `end` (int, `10`), `step` (int, `1`) |

---

## 2. Pending Blocks (Roadmap / TODO)

The following blocks are planned for upcoming phases:

### Phase A: Additional Activations & Normalizations
- [ ] `gelu` (`nn.GELU`)
- [ ] `leakyrelu` (`nn.LeakyReLU`, `negative_slope=0.01`)
- [ ] `elu` (`nn.ELU`, `alpha=1.0`)
- [ ] `silu` (`nn.SiLU`)
- [ ] `batchnorm1d` (`nn.BatchNorm1d`)
- [ ] `groupnorm` (`nn.GroupNorm`)
- [ ] `instancenorm2d` (`nn.InstanceNorm2d`)

### Phase B: Advanced Tensor & Shape Transformations
- [ ] `transpose` (`torch.transpose(in, dim0, dim1)`)
- [ ] `permute` (`torch.permute(in, dims)`)
- [ ] `squeeze` (`torch.squeeze(in, dim)`)
- [ ] `stack` (`torch.stack(tensors, dim)`)
- [ ] `clamp` (`torch.clamp(in, min, max)`)
- [ ] `sum` / `mean` / `max` / `min` reduction blocks

### Phase C: Transformer & Sequence Blocks
- [ ] `embedding` (`nn.Embedding(num_embeddings, embedding_dim)`)
- [ ] `multiheadattention` (`nn.MultiheadAttention(embed_dim, num_heads, dropout)`)
- [ ] `lstm` (`nn.LSTM`)
- [ ] `gru` (`nn.GRU`)
- [ ] `conv_transpose2d` (`nn.ConvTranspose2d`)
