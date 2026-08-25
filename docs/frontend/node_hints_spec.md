# Node Hints & Contextual Warnings — Feature Spec

**Status**: Planning  
**Scope**: Frontend only  
**Touches**: `store.ts`, `CustomNode.tsx`, `TensorEdge.tsx`, `page.tsx`

---

## 1. Problem Statement

ArchIDE currently surfaces two states for a node: **ok** (no error) and **red / shape-error**
(hard `ValueError` from the backend). There is no middle ground.

The motivating case: connecting a `Conv2D → ReLU → Linear` without a `Flatten` in between
is not *wrong* (PyTorch's `nn.Linear` operates on the last dim of any tensor), but it is
almost certainly unintentional when the input is a 4D spatial tensor. The user needs a nudge,
not a hard stop.

The broader need is a whole class of **"you can do this, but are you sure?"** situations that
are model-topology-aware, not just dtype/shape-mismatch errors.

---

## 2. Design Goals

| Goal | Note |
|---|---|
| Purely frontend | No new backend endpoint; hints are derived from `nodeShapes` already in the Zustand store after `/api/check` |
| Lightweight | A single pure function per hint rule, evaluated synchronously in JS |
| Toggleable | Global **Hints** toggle (on by default); per-hint **verbose** mode shows explanatory text |
| Non-blocking | Hints are `warning` or `info` severity — never prevent compilation |
| Composable | Easy to add new rules without touching existing components |

---

## 3. Hint Severity Model

```
info     ──  grey/blue chip   ── "Did you know?" / informational
warning  ──  amber chip       ── "This might not be what you want"
```

There is no `error` severity in hints — hard errors already come from the backend
`shapeErrorNodeId` path and are displayed as red node borders.

---

## 4. The Hint Rule System

### 4.1 Architecture

A hint rule is a **pure function** with this signature:

```ts
type HintContext = {
  node: Node;                              // the node being evaluated
  allNodes: Node[];
  allEdges: Edge[];
  nodeShapes: Record<string, Record<string, number[]>>;  // from store
};

type Hint = {
  nodeId: string;
  severity: "warning" | "info";
  code: string;          // stable machine-readable ID, e.g. "LINEAR_HIGH_DIM_INPUT"
  short: string;         // <= 6 words, shown in compact mode
  verbose: string;       // 1-2 sentence explanation, shown in verbose mode
};

type HintRule = (ctx: HintContext) => Hint | null;
```

All rules live in `src/lib/hints.ts`. `computeHints(allNodes, allEdges, nodeShapes)` maps
over every node, runs all rules, and returns `Hint[]`. This is called once after each
successful `/api/check` response, exactly where `nodeShapes` is already set.

### 4.2 Store Addition

```ts
// store.ts — add alongside shapeErrorNodeId
hints: Hint[];
setHints: (hints: Hint[]) => void;
hintsEnabled: boolean;
toggleHints: () => void;
hintsVerbose: boolean;
toggleHintsVerbose: () => void;
```

---

## 5. Planned Hint Rules

This is the core intellectual work. Rules are grouped by the *trigger node* — the node
at which the hint is surfaced, not the upstream node that caused the situation.

---

### Rule 1: `LINEAR_HIGH_DIM_INPUT` ⚠️ warning

**Trigger node**: `linear`  
**Condition**: Input tensor rank >= 3 AND input is connected AND shapes are known  
**NOT triggered if**: Input comes directly from a `flatten` or `reshape` node (those are
intentional dimensionality operations)

```
short:   "Input is N-D — is Flatten needed?"
verbose: "nn.Linear applies to the last dimension only. Your input has shape
          [B × T × C × ...]. If this is a CNN feature map and you intend a fully
          connected layer, insert a Flatten node first. If this is a sequence model
          (Transformer, RNN), this is intentional — dismiss this hint."
```

**Why the carve-out matters**: a `(B, T, embed_dim) → Linear` is a perfectly valid
per-token projection. The hint should only fire for spatial CNN contexts, so the
upstream-is-flatten/reshape check is essential.

---

### Rule 2: `BATCHNORM_AFTER_LINEAR` ⚠️ warning

**Trigger node**: `batchnorm2d`  
**Condition**: Input tensor rank != 4 AND shapes are known  
**Short**: `"BatchNorm2D expects 4D input"`  
**Verbose**: `"BatchNorm2D requires (N, C, H, W) input. Your upstream tensor has rank
             {rank}. For 1D/2D data after a Linear layer, use BatchNorm1D instead."`

---

### Rule 3: `SOFTMAX_BEFORE_LOSS` ⚠️ warning

**Trigger node**: `softmax`  
**Condition**: There exists a downstream path from this node to the `output` node AND
the output is the terminal node (i.e., likely used as classification logits)  
**Short**: `"Softmax before output — use logits?"`  
**Verbose**: `"PyTorch's CrossEntropyLoss and BCEWithLogitsLoss expect raw logits, not
             softmax probabilities. Applying Softmax here will reduce numerical stability
             and produce incorrect gradients with those loss functions."`

This one requires **graph traversal** (is there a path to Output?), which makes it the
most complex rule. We only traverse up to depth ~5 to stay O(1) in practice.

---

### Rule 4: `DROPOUT_FIRST_LAYER` ℹ️ info

**Trigger node**: `dropout`  
**Condition**: The only upstream node is `input`  
**Short**: `"Dropout on raw input — intentional?"`  
**Verbose**: `"Applying Dropout directly to the input tensor discards raw features before
             any learning. This is unusual; Dropout is typically placed after activation
             functions or between hidden layers."`

---

### Rule 5: `LINEAR_DIM_EXPLOSION` ⚠️ warning

**Trigger node**: `linear`  
**Condition**: `out_features > in_features * 4`  
**Short**: `"Large output expansion (×{ratio})"`  
**Verbose**: `"This Linear layer expands the feature dimension by {ratio}x
             ({in} → {out}). Large expansions are uncommon and may indicate a
             mis-configured parameter."`

---

### Rule 6: `CONV_NO_BATCHNORM` ℹ️ info  *(verbose-only)*

**Trigger node**: `conv2d`  
**Condition**: The immediately downstream node (ignoring activation pass-throughs like
`relu`, `sigmoid`, `tanh`) is **not** a `batchnorm2d`  
**Suppressed in compact mode** — only appears when verbose is on  
**Short**: `"Conv without BatchNorm"`  
**Verbose**: `"In most modern CNN architectures (ResNet, VGG, EfficientNet), Conv2D is
             followed by BatchNorm2D before the activation. Skipping BatchNorm can make
             training less stable, though it is valid for small networks or specific
             designs."`

---

### Rule 7: `ADAPTIVEAVGPOOL_THEN_FLATTEN` ℹ️ info

**Trigger node**: `flatten`  
**Condition**: Immediate upstream is `adaptiveavgpool2d`  
**Short**: `"Flatten after AdaptiveAvgPool — consider view()"`  
**Verbose**: `"AdaptiveAvgPool2D with output_size=(1,1) collapses spatial dims to 1x1.
             The subsequent Flatten is equivalent to .view(B, -1). This is valid; just
             note that some architectures use .flatten(1) or squeeze() instead."`
*(This is purely informational — confirms the user knows what is happening)*

---

## 6. UI Rendering

### 6.1 Hint Badge on `CustomNode`

A small amber dot/chip appears in the **top-right corner** of the node, separate from
the red error badge (bottom-left). Clicking it opens an inline tooltip.

```
┌─────────────────────────┐
│ Conv2D              [ℹ] │  ← info badge (grey-blue)
│                         │  OR
│ Linear              [⚠] │  ← warning badge (amber)
└─────────────────────────┘
```

In **compact mode** (hints on, verbose off): badge + `short` text in a small chip below
the node label.

In **verbose mode** (hints on, verbose on): badge + `short` text + expandable `verbose`
explanation rendered inside the node card or a hover popover.

The badge must be **visually distinct** from the existing red shape-error chip (different
corner, different icon — `Info` / `AlertTriangle` from lucide).

### 6.2 Global Hints Toggle

Lives in the **toolbar** (same row as "Run Static Tensor Check" and "Export PyTorch"),
implemented as a small toggle pill:

```
[ ⚠ Hints  ON  ] [ Verbose ]
```

- **Hints ON/OFF**: hides/shows all hint badges and chips globally
- **Verbose**: switches between compact (short text only) and verbose (full explanation)
  mode. Only relevant when Hints is ON.

Toggle state persists in Zustand (not localStorage — session-only is fine for now).

### 6.3 Hint Count Summary (optional / later)

A small "3 hints" counter in the status bar at the bottom of the canvas, clickable to
scroll to the first hinted node.

---

## 7. Implementation Plan (Phases)

### Phase 1 — Infrastructure
- [ ] Add `hints`, `hintsEnabled`, `hintsVerbose` to `store.ts`
- [ ] Create `src/lib/hints.ts` with `HintRule` type, `computeHints()`, and rules 1–3
- [ ] Wire `computeHints()` call into the `/api/check` success handler in `page.tsx`

### Phase 2 — UI
- [ ] Add hint badge to `CustomNode.tsx` (amber corner chip, compact + verbose render)
- [ ] Add toggle pill to toolbar in `page.tsx`

### Phase 3 — Remaining Rules
- [ ] Add rules 4–7 to `hints.ts`
- [ ] Add optional hint count summary to status bar

---

## 8. Open Questions

**Q1 — Hint persistence**: Should hint state (enabled/verbose) be saved to
`localStorage` so it survives page refresh? Or is session-only fine?

**Q2 — Hints without shape check**: Rules 1–2 and 5 require `nodeShapes` (i.e., the
user must have run the static check first). Should we also evaluate shape-independent
rules (like rule 4 `DROPOUT_FIRST_LAYER`, rule 5 `LINEAR_DIM_EXPLOSION`) eagerly on
every graph change, even before a check is run?

**Q3 — Rule 3 (`SOFTMAX_BEFORE_LOSS`)**: This requires knowing what loss function
will be applied, which ArchIDE does not model yet. For now the rule could fire whenever
`softmax` connects to `output` (the terminal node), which is a reasonable proxy. Worth
discussing.

**Q4 — Per-node dismiss**: Should individual hints be dismissable per-session (e.g.
clicking X on a hint suppresses it for that node until the graph changes)? This adds
complexity but improves the experience for users who know what they are doing.
