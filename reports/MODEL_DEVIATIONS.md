# Recursive solver and adapters

The implementation is a custom TRM-inspired graph solver with separate maze and
circuit models. It is not an exact TRM or RSM reproduction. The source is in
[`models/recursive.py`](../src/state_repair/models/recursive.py),
[`models/adapters.py`](../src/state_repair/models/adapters.py), and
[`models/policy.py`](../src/state_repair/models/policy.py).

## Recurrence and representation

For encoded observations `e`, answer latent `a`, and work latent `z`:

```text
e = Linear(observed node features)
a = broadcast learned vector a0; z = broadcast learned vector z0
repeat K outer cycles:
    repeat L inner cycles: z = F(e + a + z, observed_relations)
    a = F(a + z, observed_relations)
logits = Linear(a)
```

Maze features are normalized row/column coordinates and start/goal indicators;
the head predicts six actions. Circuit features are five one-hot operator classes
and the observed external input bit; its head predicts two classes. Circuit loss
and accuracy exclude external inputs. IDs, frame indices, split labels, generator
seeds, oracle values, and targets are never embedded. Node permutations jointly
reorder features, relations, masks, and latents while preserving node identities.

`F` has two distinct transformer layers reused together at every call, followed
by LayerNorm. Each layer uses pre-LayerNorm attention, a residual update, and a
pre-LayerNorm GELU feedforward of width `4D` with another residual update.
LayerNorm uses PyTorch's default epsilon `1e-5`. There is no dropout, rotary
embedding, special token, learned halting, or sparse execution.

Attention adds a learned per-head relation bias. Maze relations are `0=absent`,
`1=north`, `2=east`, `3=south`, `4=west`, indexed by source/query and target/key.
Circuit relations are `0=absent`, `1=forward wire`, `2=reverse relation`; all five
embedding rows remain allocated. The default `masked_neighbor` mode permits
observed neighbors and self. The `dense` ablation permits every valid node pair.
Both execute dense quadratic matrix operations. Padding keys are masked before
softmax and padded outputs are zeroed after layers, normalization, encoding, and
decoding. Finite padded latent values cannot affect valid outputs.

Fresh vectors and relation embeddings use normal initialization with standard
deviation `0.02`; Linear and LayerNorm use PyTorch defaults. Fresh and returned
states have independent storage. Supplied states must match the observation's
frame, episode, node identities, mask, dimensions, dtype, and device. Default
forward starts fresh; same-frame continuation accumulates the state's budget.
An adapter constructs the current-frame state at each edit boundary.

All requested cycles remain differentiable, including with frozen parameters.
Adapters detach previous-version state before initialization; the current
adapter-to-loss path stays on the autograd graph. `K=0` still executes encoding,
initialization or cloning, and decoding, but no solver block calls. Its fresh
prediction depends only on the learned initializer and decoder.

## Parameters and work

At `D=64`, two attention heads:

| Component | Maze parameters | Circuit parameters |
|---|---:|---:|
| Encoder | 320 | 448 |
| Fresh a/z | 128 | 128 |
| Shared block | 100,116 | 100,116 |
| Head | 390 | 130 |
| Total | 100,954 | 100,822 |

`K` outer cycles with `L` inner updates execute `(L+1)K` shared-block calls and
`2(L+1)K` transformer layers. Parameters are independent of `K`. The scaled maze
configuration uses `L=2`: three block calls and six layer executions per cycle.
Forward-hook tests independently check these counts; they are not FLOP or latency
measurements.

## Adapters and controls

The registry contains `restart`, `carry`, `spatial_gate`, `global_gate`,
`gru_adapter`, `residual_adapter`, `local_reset`, `random_reset`, `noisy_carry`,
`shuffled_gate`, and `answer_only`. The last five have `is_control=True`.
Evaluator impact-mask resets live separately in
[`eval/interventions.py`](../src/state_repair/eval/interventions.py), emit
`privileged=true`, and cannot be selected through `make_adapter`. They are not
optimal-repair bounds.

The four learned principal adapters have separate trainable copies of the same
observed-context architecture. It projects old/new features, feature deltas,
incoming/outgoing changed-edge indicators, old/new relation degrees, and detached
old `a/z` to context width `C`. Degrees exclude absent edges and divide by valid
node count. Input width is `I=3F+2+2R+2D`: 150 for mazes and 152 for circuits at
`D=64`. Old wiring enters through degree summaries and incident edits; new wiring
also controls context attention. No IDs or evaluator metadata are features.

The context block has two masked-neighbor transformer layers, one attention head,
and final LayerNorm. Newly injected information has a two-hop receptive field;
old latents can already summarize more distant nodes. Global gating additionally
pools all valid context nodes.

Spatial gating predicts separate sigmoid retain scalars for `a` and `z` at each
node; global gating predicts one scalar per latent per example. Both initialize
`r * old.detach() + (1-r) * fresh` once per edit. Frame zero bypasses learned
adapter work. Fixed zero/one gates match restart/carry. `retain_override` is a
constant control, not a fitted gate. Residual adaptation adds an unrestricted
linear `C -> 2D` delta to detached old latents.

GRU adaptation uses `torch.nn.GRUCell(C, 2D)` with `h=[a,z]` and `x=context`:

```text
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
u = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
h_new = (1-u)*n + u*h
```

The candidate applies reset after the hidden affine operation, as in PyTorch.

Local reset expands changed-feature nodes and changed-edge endpoints through the
union of old/new adjacency at radius 1, 2, or 3. Random reset selects exactly
`round(rate * valid_nodes)` nodes without replacement. Noisy carry adds independent
normal noise, with default scale `0.01`. Shuffled gating permutes paired `a/z`
retention values over valid nodes and preserves their multiset. Random controls
use PyTorch RNG; padding consumes no random draws. Equal seeds under node
permutations need not produce equivalent random placements.

Answer-only reuse projects the previous model prediction's detached softmax
probabilities to `2D` latents. It neither reads nor decodes old latents. Changing
old latents while holding predictions fixed leaves this initialization unchanged.

The following implementation counts use `D=64`, `C=16`, `B=1`, `N=4`, without
padding. Regenerate them with
[`scripts/check_adapter_accounting.py`](../scripts/check_adapter_accounting.py)
using an explicit `--output` path. Its records are marked `synthetic=true` and its
single cold timings are not latency benchmarks.

| Adapter | Parameters maze / circuit | Dense MACs per edit maze / circuit | Linear / GRU calls | Context block / layers |
|---|---:|---:|---:|---:|
| restart, carry | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| spatial_gate | 9,052 / 9,084 | 35,328 / 35,456 | 10 / 0 | 1 / 2 |
| global_gate | 9,052 / 9,084 | 35,232 / 35,360 | 10 / 0 | 1 / 2 |
| gru_adapter | 65,082 / 65,114 | 256,384 / 256,512 | 9 / 1 | 1 / 2 |
| residual_adapter | 11,194 / 11,226 | 43,392 / 43,520 | 10 / 0 | 1 / 2 |
| local_reset, random_reset, noisy_carry | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| shuffled_gate | 9,052 / 9,084 | 35,328 / 35,456 | 10 / 0 | 1 / 2 |
| answer_only | 896 / 384 | 3,072 / 1,024 | 1 / 0 | 0 / 0 |

Context MACs are `BNIC + 24BNC² + 4BN²C`. Add `2BNC` for spatial gating,
`2BC` for global gating, `2BNDC` for residual adaptation, or
`3BN(2D)(C+2D)` for GRU adaptation. Core MACs are
`(L+1)K * (24BND² + 4BN²D)`. The GRU has substantially more capacity and compute
than the scalar gates; identical inputs do not imply capacity matching.

MAC counts cover dense linear and attention products, excluding bias additions,
normalization, nonlinearities, softmax, masking, reductions, validation, edit
preparation, and copies. Zero MACs does not imply zero cost. Partial counters
also record gate mixing, residual/noise operations, neighborhood expansion, and
random draws. Timings include executed work in the encoder, fresh initialization,
adapter, core, decoder, correspondence checks, and storage copies. Device timing
synchronizes CUDA/MPS when used. Retaining more state does not itself save work.

## Stream ownership and verification

`FixedBudgetPolicy` owns detached previous observations, predictions, and state.
Each instance has one budget and adapter. Explicit continuation checks the
per-frame budget and policy/stream token, including for restart. Frame zero
clears internal state and rejects supplied history. Use separate instances for
other budgets, checkpoints, or independently evaluated policies, and reset the
wrapper after weight changes. The token is in-memory provenance, not a signed
persistent cache. Mixed fresh/edited examples require separate batches. Include
the initial solve in amortized cost.

Tests in [`test_model.py`](../tests/test_model.py),
[`test_adapters.py`](../tests/test_adapters.py),
[`test_policy.py`](../tests/test_policy.py), and
[`test_circuit_model.py`](../tests/test_circuit_model.py) cover block counts,
`K=0`, direction and permutation behavior, padding, independent storage, stream
budget ownership, adapter gradients with frozen backbones, the broken-detach
negative control, gate extremes, context reach, output-only isolation, and
operation accounting. These synthetic checks do not establish learned accuracy.

Saved capacity records include a width-128 maze failure at `K=16` under a 6 GiB
allocator cap (`runs/scaled_cuda_profile_w128`); the width-64/two-head fallback
retains 32 channels per head (`runs/scaled_cuda_profile_w64`). Width 64/four heads
was not separately profiled. Circuit training used width 128/four heads and batch
128 after a batch-256 capacity failure. These failures are not omitted learning
arms. The 72-run maze adapter pilot failed its declared 32-edit research gate;
see [the pilot report](PILOT_REPORT.md) and [saved results](pilot_results.json).

Cross-runtime bitwise inference is not guaranteed. The saved 5,016-prediction CPU
spot replay differed on three route decisions after long streams. Replay of the
discrepant cases in the original CUDA runtime reproduced 25,344 predictions
exactly. Small numerical differences can amplify across recurrent streams;
these diagnostics do not replace the original results.

## Source lineage

The recurrence is inspired by [Less is More: Recursive Reasoning with Tiny
Networks](https://arxiv.org/html/2510.04871v1) and the
[official implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
No upstream TRM, RSM, or adapter code was copied, so no reused-code commit applies.
The upstream TRM repository's MIT license does not make its mutable branch a
pinned dependency. PyTorch is used as a dependency under its installed license,
including its bundled third-party notices.

Departures from TRM include graph-relation biases, domain-specific node features
and heads, LayerNorm/GELU blocks, learned fresh vectors, fully differentiable
requested recurrence, no ACT, and smaller widths. These differences preclude
claims of an exact upstream reproduction. Reproduction commands and artifact
compatibility requirements are in [REPRODUCING.md](../REPRODUCING.md).
