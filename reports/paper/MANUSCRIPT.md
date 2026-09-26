# When Does Reusing Computation Help Small Recursive Solvers on Changing Problems?

**State reuse in editable mazes and Boolean circuits under a protocol fixed before testing**

Cleo Spiers.

## Abstract

A small recursive solver can use its previous computation to solve an edited maze more accurately when it has few refinement cycles. With more cycles, that advantage can shrink or reverse. We compare retaining latent state, reusing the previous output and restarting across five training seeds and 256 held-out roots per suite. At one cycle, every tested maze reuse policy exceeds restart by 27 to 51 percentage points. At eight cycles, restart and output reuse have higher mean accuracy than every policy that retains latents. Output reuse and the spatial gate also improve on restart under changes to circuit structure. Updating corrects some old answers: at one cycle, output reuse raises maze accuracy from 72.5% to 77.1%, and ordinary circuit accuracy from 32.5% to 99.3%. None of the three primary suites meets the prespecified target of preserving accuracy while reducing cost by 25%. We examine this outcome through timing of complete streams, intervals that include training seed variation, and correction diagnostics. Each policy is trained jointly with its solver, so these comparisons cannot isolate the effect of its initializer.

## 1. Introduction {#sec:introduction}

When a problem changes, a solver has a choice: start again or use what it has already computed. A small edit may leave an answer mostly intact. It may also change which answers are correct far from the edit itself. Reusing the old state can help with the next solve, but it also changes the starting point for every subsequent refinement. We study this choice in small recursive neural solvers that repeatedly update an answer latent and a work latent [TRM].

Mazes and Boolean circuits let us check exactly what an edit changes and whether the solver's new answer is correct. They also give us practical cost references. D* Lite and event-driven circuit evaluation are exact and cheaper on these tasks; their timings show how much the neural solvers cost relative to established methods. Within the neural comparisons, each policy has its own jointly trained backbone. A difference between policies can therefore reflect the initializer, the learned solver weights, or their interaction.

The available refinement budget changes which form of reuse works best. We measure this change across tasks and training seeds, and examine cases where adding cycles reduces accuracy. Some failures begin on the initial problem, before an edit or reuse operation can occur. Others appear as the solver processes the stream. Keeping both parts of the trajectory is necessary to tell them apart.

We make three contributions:

1. **We measure how the ordering changes with budget.** On 12x12 mazes, every tested reuse policy exceeds restart at one or two cycles. At one cycle, carry, the spatial and global gates, and the GRU adapter also exceed output reuse; the residual adapter falls below it. By eight cycles, restart and output reuse are near ceiling and have higher mean accuracy than all latent reuse policies. The global gate's gap from output reuse remains uncertain when training seeds are resampled. Section \ref{sec:budget-crossover} reports all principal policies, including the variation across seeds and tasks.
2. **We evaluate each policy over its own sequence of predictions.** A policy at a fixed budget carries only the state that it produced at that budget. Timing includes the initial solve and adapter overhead. Ordinary edits are sampled without rejecting them based on their consequences, then grouped by consequence for analysis. We record deployed streams, interventions with a shared prior state, classical references and privileged diagnostics separately. The confirmatory comparison and 2,176 test identities were committed before test generation. Section \ref{sec:protocol} describes the protocol.
3. **We check what updating corrects and what previous answers contribute.** The correction diagnostic scores each policy's old prediction on the edited problem before comparing it with the new prediction. We also hold the supplied prior state fixed across initializers and remove previous output content within a trained system, using matched seeds. On mazes, previous answer content supports the output reuse gain at small budgets. Under changes to circuit structure, some advantage remains after that content is removed. These interventions leave the effects of backbone training and initialization partly unresolved (Sections \ref{sec:previous-prediction}, \ref{sec:common-prior} and \ref{sec:reuse-gains}).

The frozen deployment test asks whether the output reuse policy selected on validation can remain no more than one percentage point below restart's accuracy while reducing amortized cost by at least 25% on all three primary suites. No suite meets the full target under either original timing mode. The accuracy gains at equal budgets therefore need to be read alongside the result at the selected operating points. Sections \ref{sec:frozen-comparison} and \ref{sec:cost} report that test and its cost components. The appendix also retains the development comparison of dense attention and attention restricted to neighbors, which used one seed (Section \ref{sec:backbone}).

For someone building a solver of this kind, output reuse is an inexpensive baseline in parameter count: it adds 896 parameters on mazes and improves K=1 accuracy by about 40 percentage points. Its measured latency still includes projection, copying and transfer. At larger budgets on ordinary mazes, restart matches or exceeds latent reuse, so an adapter needs to justify its extra work against that baseline. The circuit results under structural shift show why this choice needs to be checked on the intended task.

We retain every principal policy in the applicable full results tables and account for failed and interrupted attempts. The frozen decision uses intervals conditional on the observed training seeds, obtained by resampling roots. Exploratory comparisons emphasize crossed intervals that resample both training seeds and complete root episodes. The principal comparisons use five seeds; the lesions use three matched seeds. We also retain frame zero and seed identity so that a poor initial solve can be distinguished from deterioration after edits (Section \ref{sec:frame-zero}). Finally, mean accuracy can conceal differences in completing an entire stream: across 128 maze edits, restart and output reuse have nearly equal mean accuracy but whole-stream success of 96.2% and 91.6%, respectively. That reliability comparison is descriptive.

## 2. Empirical findings {#sec:results}

**How to read the results.** Each stream starts with a fresh problem, then changes one maze passage or one circuit input/operator at a time. A policy has K outer cycles to refine its answer at each frame, and each cycle makes three calls to the shared neural block. We evaluate budgets of K=1, 2, 4, 8 and 16. Each budget uses only the history it produced when processing the next edit. A maze answer is exact if it gives a shortest valid route from the designated start or correctly declares the goal unreachable. A circuit answer is exact if every value at a non-input node is correct.

**Policies.** Restart begins every frame with fresh answer and work latents; carry keeps both. Output reuse (answer_only in the saved records) initializes both latents from an affine projection of the previous soft output distribution, discarding the old latents. The spatial gate mixes old and fresh latents at each node, while the global gate sets one scalar for each latent tensor. The GRU adapter uses an ordinary recurrent update. The residual adapter adds an unconstrained learned correction. We compare all seven policies on mazes. The circuit comparisons include restart, carry, output reuse and the spatial gate. Section \ref{sec:solver} gives their equations and parameter counts.

**Suites and development history.** The primary suites contain 12x12 mazes, ordinary circuits with 32 nodes, and structurally shifted circuits with 48 nodes evaluated using the model trained on 32 nodes. We also train secondary models on 16x16 mazes and circuits with 64 nodes, then test them on shifted suites of 20x20 mazes and circuits with 96 nodes. Each circuit node count includes eight inputs. We use pilot to refer to the earlier studies restricted to validation, which froze the backbone or trained on streams and used three adaptation seeds with one shared backbone per family. The pilot selected output reuse for the frozen deployment comparison. In the present study, each size has five independently initialized backbones, and we jointly train each policy's solver and initializer.

The principal accuracy curves use five seeds and 256 held-out roots, with 32 edits per episode. Longer streams have 128 edits. These comparisons are exploratory; Section \ref{sec:frozen-comparison} reports the frozen deployment decision separately. To form crossed intervals, we resample training seeds and whole roots, keeping all edits and paired policies together within each draw. Lesions use three matched seeds. The cost figures identify their smaller matched samples for timing.

### 2.1 The maze budget crossover (exploratory) {#sec:budget-crossover}

Every reuse policy beats restart on 12x12 mazes at K≤2. At K=1, carry and three learned latent adapters also beat output reuse; the residual adapter does not. That advantage mostly disappears or reverses by K=4. At K=8, all policies that reuse latents have lower mean accuracy than output reuse, although some small differences remain unresolved across seeds. In particular, the crossed interval for the global gate spans zero. Several learned adapters lose accuracy at K=16, which is beyond the budgets used in joint training. Table \ref{tab:maze-accuracy} reports the accuracy curves. Selected crossed intervals and paired seed ranges appear in Table \ref{tab:key-contrasts}, with all contrasts conditional on the trained seeds in Table \ref{tab:maze-contrasts}. Figure \ref{fig:maze-crossover} shows the variation across training seeds.

<!-- figure:maze-crossover -->

<!-- figure:maze-differences -->

The ordering at small budgets is the reverse of the pilot result. Two parts of the confirmatory recipe changed: the backbones were trained independently, and joint training used more updates. The comparison cannot tell us which change caused the reversal. The larger maze suite also crosses over, but its policy ordering differs and depends more strongly on the seed.

<!-- table:maze-accuracy -->
| 12x12 mazes, 32 edits | K1 | K2 | K4 | K8 | K16 |
|---|---:|---:|---:|---:|---:|
| restart | 37.17 | 74.28 | 93.44 | 99.83 | 99.83 |
| carry | 85.26 | 92.50 | 96.30 | 97.69 | 97.72 |
| spatial_gate | 85.41 | 93.88 | 98.52 | 99.53 | 97.04 |
| global_gate | 86.17 | 94.89 | 98.72 | 99.70 | 98.63 |
| gru_adapter | 88.12 | 93.85 | 98.26 | 99.44 | 98.71 |
| residual_adapter | 64.65 | 89.35 | 97.53 | 98.67 | 95.94 |
| answer_only | 77.08 | 92.39 | 98.65 | 99.90 | 99.90 |



On 12x12 mazes, success on the whole stream requires all 33 frames to be exact. At K=8, this score is 99.0 for restart, 98.3 for output reuse, 94.9 for the spatial gate and 84.5 for carry. A high mean accuracy after edits can therefore coexist with substantially lower success across a complete stream. The two aggregate measures alone do not establish whether errors persist.

**Ordinary circuits.**

Ordinary circuits with 32 and 64 nodes reach near ceiling above K=1. On the 32-node suite, restart reaches 99.97 at K=2. At K=1, output reuse and the spatial gate improve on restart's 98.87 by +0.46 pp [+0.13, +0.83] and +0.49 pp [+0.16, +0.87], respectively, while carry collapses to 62.1.

<!-- table:key-contrasts -->
<!-- BEGIN key-contrasts -->
| Maze comparison | K | Δ (pp) | Crossed 95% CI | Seed Δ range |
|---|---|---|---|---|
| answer_only − restart | 1 | +39.91 | [+34.93, +44.86] | [+37.46, +46.20] |
| answer_only − restart | 8 | +0.07 | [-0.10, +0.39] | [-0.07, +0.39] |
| gru_adapter − answer_only | 1 | +11.04 | [+4.94, +16.37] | [+0.65, +17.08] |
| carry − answer_only | 8 | -2.21 | [-3.56, -1.19] | [-4.02, -1.07] |
| spatial_gate − answer_only | 8 | -0.37 | [-0.87, -0.07] | [-0.92, -0.02] |
| global_gate − answer_only | 8 | -0.20 | [-0.65, +0.07] | [-0.70, +0.12] |
| gru_adapter − answer_only | 8 | -0.46 | [-1.04, -0.11] | [-1.03, -0.13] |
| residual_adapter − answer_only | 8 | -1.23 | [-1.84, -0.72] | [-1.44, -0.72] |
| spatial_gate − answer_only | 16 | -2.85 | [-6.96, -0.54] | [-10.34, -0.56] |
<!-- END key-contrasts -->

### 2.2 Size and structural shifts {#sec:distribution-shift}

**Size shift, 20x20 mazes.** For the model trained on 12x12 mazes, reuse helps at K ≤ 4 (carry 64.5 versus restart 19.6 at K=1). At K=8 it hurts: restart scores 93.1, output reuse 92.4, the GRU adapter 86.7, the spatial gate 81.1 and carry 64.1. For the model trained on 16x16 mazes, output reuse is slightly above restart at K=8 (98.2 versus 98.0), and both are above the spatial gate (95.5). This restart/output ordering differs from the 12x12 model. Carry changes little as K increases under size shift (64.5 to 62.7). This is the clearest case in which more cycles fail to improve the carried state.

**Combined size, depth and wiring shift, circuits with 48 nodes evaluated using the model trained on 32 nodes.** Every reuse policy has higher mean accuracy than restart on the suite with 48 nodes at K ≥ 2. At K=8, the spatial gate scores 81.8, output reuse 78.9, carry 61.3 and restart 59.8. The spatial gate minus restart difference is +21.9 pp [+20.4, +23.5]; output reuse minus restart is +19.1 pp [+17.6, +20.6]. The corresponding crossed intervals are [+14.1, +30.2] and [+12.2, +27.1], reflecting differences of up to ten points among the five backbones. The spatial gate's advantage over output reuse is smaller and uncertain when seeds are resampled: +2.9 pp [+2.3, +3.5] at K=8 when resampling roots, but [−0.2, +6.0] under crossed resampling. Its advantage is +4.4 [+3.6, +5.3] at K=2 and −3.4 [−4.1, −2.7] at K=1. At 96 nodes, the difference is +1.5 [+0.8, +2.2] at K=8, with a crossed interval of [−9.0, +11.7]. Carry beats output reuse at K=1 under both shifts, then loses that relative advantage as the budget increases. Carry's accuracy on circuit48 still rises with K; it does not collapse in absolute terms. At 96 nodes, using the model trained on 64 nodes, output reuse and the spatial gate again improve on restart. Carry does not improve on restart at K8. The spatial gate scores 78.6, output reuse 77.1 and restart 45.7 at K=8, giving gains of +32.9 pp [+31.2, +34.5] and +31.4 pp [+29.8, +33.0]. Returning to the 48-node suite, the frozen comparison of K1 with K2 fails because restart at K=2 (23.4) already exceeds output reuse at K=1 (13.7). With equal K, output reuse leads restart by 22.0 pp at K=2 and 20.3 pp at K=4.

<!-- figure:circuit-depth-shift -->

### 2.3 Does updating improve the previous prediction? {#sec:previous-prediction}

If an edit leaves the previous answer valid, retaining it may be enough. We test how often updating improves that answer by scoring each policy's previous argmax prediction on the current problem before refinement, then scoring its updated output. For mazes, we follow the old action map from the designated start. It must still give a valid shortest route or correctly declare the goal unreachable. A changed target elsewhere does not, by itself, make that route wrong. For circuits, every value at a non-input node must match the current circuit. Both scores use the same current problems and the history produced by that policy at its own fixed budget.

Table \ref{tab:previous-prediction} gives accuracy before and after updating, the fraction of wrong predictions repaired, and the fraction of correct predictions broken. Gain intervals are crossed 95% intervals over five seeds and 256 complete root episodes. The supplement includes all 75 policy/budget points and the four possible before/after outcomes counted by episode. Here we report output reuse and spatial gating at K1 and K8 on each of the three primary suites.

On mazes, previous predictions are often still useful, and refinement corrects additional cases. Output reuse at K1 increases accuracy from 72.46% to 77.08%, a gain of +4.62 pp [4.13, 5.13]. It repairs 20.27% of previously wrong cases and breaks 1.32% of previously correct ones. At K8, accuracy rises from 94.59% to 99.90%, with 98.24% of invalid predictions repaired. Carry also changes its answers: at K1, its previous prediction scores 79.66% and its updated prediction 85.26% (+5.60 pp [5.10, 6.12]). Each comparison depends on that policy's learned history. These differences therefore cannot causally divide up the advantage one policy has over restart.

Ordinary circuits require much more correction. At K1, output reuse improves accuracy from 32.49% for the previous prediction to 99.33% after updating. On structurally shifted circuit48 at K8, output reuse raises accuracy from 33.92% to 78.91%; the spatial gate raises it from 35.21% to 81.77%. Updating makes a substantial contribution beyond keeping discrete predictions that are already correct. These results do not evaluate a policy that only caches its answers. The previous predictions came from histories that were actively updated, and output reuse receives soft distributions, not just argmax values. The diagnostic also provides no cost comparison for a policy that only caches answers.

The ordinary edit streams and enriched challenge branches have very different consequence frequencies. On maze12, 95.12% of ordinary edits have low consequence and 1.28% have high consequence. The reserved challenge instead has 64 low and 64 high branches. For ordinary circuit32, the high consequence fraction is 3.30%; it rises to 9.88% under the circuit48 structural shift. Section \ref{sec:consequence-detail} describes the challenge construction and gives frequencies for all six suites. These strata measure changes in target sets. They do not measure route validity or identify invalid latent state.

<!-- table:previous-prediction -->
<!-- BEGIN previous-prediction -->
| Suite / policy | K | Old % | New % | Repair % | Break % | Gain pp [95% CI] |
|---|---|---|---|---|---|---|
| maze12, answer_only | 1 | 72.5 | 77.1 | 20.3 | 1.3 | +4.62 [+4.13, +5.13] |
| maze12, answer_only | 8 | 94.6 | 99.9 | 98.2 | 0.0 | +5.31 [+4.74, +5.92] |
| maze12, spatial_gate | 1 | 80.1 | 85.4 | 34.6 | 2.0 | +5.26 [+4.66, +5.91] |
| maze12, spatial_gate | 8 | 94.3 | 99.5 | 92.2 | 0.0 | +5.27 [+4.69, +5.86] |
| circuit32, answer_only | 1 | 32.5 | 99.3 | 99.1 | 0.1 | +66.84 [+65.55, +68.17] |
| circuit32, answer_only | 8 | 32.7 | 100.0 | 100.0 | 0.0 | +67.28 [+66.01, +68.57] |
| circuit32, spatial_gate | 1 | 32.5 | 99.4 | 99.1 | 0.1 | +66.84 [+65.55, +68.18] |
| circuit32, spatial_gate | 8 | 32.7 | 100.0 | 100.0 | 0.0 | +67.27 [+66.00, +68.56] |
| circuit48, answer_only | 1 | 6.0 | 13.7 | 9.2 | 16.0 | +7.72 [+5.87, +9.63] |
| circuit48, answer_only | 8 | 33.9 | 78.9 | 68.8 | 1.4 | +44.99 [+41.51, +48.22] |
| circuit48, spatial_gate | 1 | 4.5 | 10.3 | 6.9 | 17.4 | +5.79 [+3.96, +7.64] |
| circuit48, spatial_gate | 8 | 35.2 | 81.8 | 72.3 | 0.7 | +46.56 [+43.21, +49.72] |
<!-- END previous-prediction -->

### 2.4 Interventions from a common prior and the dynamics of carried state {#sec:common-prior}

For the frozen prior forks, we first let the spatial gate's backbone solve the problem at K=8, then apply one edit. Every initializer starts from that same state and uses that backbone's weights. There are three seeds, with 384 branches on maze12 and 370 on maze16, including the reserved challenge pairs, and 256 branches per circuit suite. This comparison is interpretable for initializers that use the backbone's own parameters: restart, carry, the spatial gate, its shuffled, local, random and noisy controls, and the privileged oracle reset. The mechanism jobs also applied the global, GRU, residual and output reuse projections after transferring them from their own checkpoints to this backbone. Those rows depend on that transfer and do not describe the deployed policies. We retain them in the supplementary mechanism tables and label the transfer explicitly. Table \ref{tab:common-prior} reports the maze interventions from a common prior using the backbone's own parameters.

<!-- table:common-prior -->
| Initializer (K=1 / K=8) | Low impact (n=313) | High impact (n=66) |
|---|---:|---:|
| restart | 35.6 / 99.4 | 16.2 / 95.0 |
| carry | 96.0 / 94.3 | 27.3 / 31.3 |
| spatial_gate | 73.2 / 99.8 | 24.2 / 90.9 |
| shuffled_gate | 57.1 / 86.7 | 11.6 / 60.1 |
| local_reset_1 | 82.4 / 79.8 | 26.8 / 35.4 |
| impact_mask (privileged oracle) | 95.6 / 95.0 | 29.3 / 78.3 |

With one cycle, carried state answers most edits of low impact correctly. Eight cycles still leave substantial error on edits of high impact (low 94.3, high 31.3), and sixteen cycles make carry worse on 16x16 mazes (low 60.8). The gate gives up most of carry's advantage after one cycle on edits of low impact (73.2). In return, after eight cycles it approaches restart's accuracy on edits of high impact (90.9 versus restart's 95.0). The privileged oracle reset clears exactly the nodes whose targets changed, yet reaches only 78.3 on edits of high impact at K=8. Clearing those nodes is insufficient in this intervention. A mask of changed outputs does not tell us which latents remain valid, so this result neither locates stale information nor explains the reset's failure. We also examine carried state by extending a deployed carry policy's own history of four edits at K=8, giving it extra cycles on the same observation. The state is not at an exact fixed point. On 12x12 mazes, at least one node action changes in 9.6% of rows with low impact at K=1 and 39.2% at K=16. The mean fraction of changed actions is only 0.10% to 0.44%, however, compared with 6.0% to 14.8% on the five branches with high impact. Under structural shift, the carried state moves considerably more: 59% to 72% of rows with low impact change, with a mean changed fraction of 2.8% to 5.3%. Exact correctness rises from 58.2% at K=1 to 66.9% at K=8. Latent distance gives another descriptive measure on mazes. At K=1, carry moves the latents by 5 to 15 units, while restart moves them by 100.

The frozen prior forks give different results on circuits under structural shift. At K=8, the spatial gate (88.3 low / 36.8 high) and shuffled gate (88.5 / 33.3) have similar point estimates; restart scores 77.5 / 68.4 and carry scores 50.0 / 0.0. Relative to restart, the gate helps on edits of low impact and hurts on edits of high impact. Shuffling provides no clear evidence here that the placement of retention values explains the advantage. We did not test equivalence. These interventions also cannot separate backbone effects from initializer effects in deployed streams.

### 2.5 Interventions on output content and associations with training {#sec:reuse-gains}

**Answer content.** The lesions use seeds 29, 43 and 71. We restrict intact output reuse and separately trained restart to those same seeds and roots in the table below. Entries give mean exact accuracy after edits (%). The intervened versions were not trained as replacement policies.

<!-- table:answer-lesions -->
| Suite and budget | Restart | Intact answer-only | Uniform answer | Shuffled nodes |
|---|---:|---:|---:|---:|
| maze12, K1 | 36.38 | 77.35 | 34.02 | 24.46 |
| maze12, K2 | 73.84 | 92.18 | 66.79 | 46.37 |
| maze12, K4 | 93.07 | 98.69 | 95.56 | 62.45 |
| circuit32, K1 | 98.78 | 99.36 | 99.03 | 98.66 |
| circuit48, K8 | 63.08 | 82.03 | 75.72 | 70.61 |
| circuit96, K8 | 52.08 | 79.84 | 61.78 | 57.57 |

Removing answer content erases the advantage over restart at small budgets on mazes. At K1, the uniform intervention minus intact output reuse difference is −43.33 pp (95% CI from paired roots [−47.35, −39.35]); at K2 it is −25.39 pp [−28.94, −21.97]. Some advantage remains at K4: uniform exceeds restart, evaluated on the same seeds, by 2.49 pp. Joint backbone training could account for this residual, as could applying the learned projection to a constant input. The lesion cannot separate those explanations.

Under structural shift, removing content lowers output reuse accuracy by 6.31 pp [5.19, 7.48] at 48 nodes and 18.06 pp [16.43, 19.73] at 96 nodes. Even with uniform input, the policy remains 12.64 and 9.69 pp above restart evaluated on the same seeds, respectively; these are descriptive differences. Previous answer content contributes to accuracy, but some advantage survives its removal. The residual advantage is larger than the content loss at 48 nodes and smaller at 96 nodes. Neither result establishes that the backbone alone causes the residual. We did not run comparable content lesions for the latent adapters. On ordinary circuit32, uniform is slightly below intact output reuse at every K evaluated on the matched seeds, including 99.03 versus 99.36 at K1. Scores this close to ceiling do not establish that previous answer content has no value.

**Stream exposure.** We compare training on single edits with training on streams while matching forward calls. Stream training is strongly associated with the spatial gate's accuracy at small budgets. On 12x12 mazes at K=1, the spatial gate trained on single edits scores 46.6, compared with 85.4 after stream training. The corresponding scores are 79.4 and 85.3 for carry, 69.9 and 77.1 for output reuse, and 36.2 and 37.2 for restart. On 16x16 mazes, carry trained on single edits declines as K rises (65.0 at K=1 to 53.5 at K=16), repeating the pilot's collapse. Carry trained on streams instead rises to 96.5. This comparison screens for training associations; it cannot identify their causes, since the recipes differ in more than stream length.

**Gate values.** Shuffling preserves the distribution of learned retention values but assigns them to different nodes. On 12x12 mazes, the shuffled gate scores 38.2 at K=1 and 60.0 at K=8, compared with 85.4 and 99.5 for the spatial gate. Placement matters on this suite. Resets within one to three hops of the edit score 28.5, 25.2 and 23.1 at K=1, while random resets with a matched rate score 8.0. All fall far below every learned policy. Noisy carry scores 51.7, below carry. These reset and noise controls use the spatial gate's backbone, while the reported carry policy uses its own adapted backbone, so the comparison does not isolate the effect of resetting or noise. On circuits under structural shift, the shuffled gate has similar point estimates to the spatial gate in the frozen prior forks (Section \ref{sec:common-prior}). Evidence about placement in that setting is limited; similar estimates do not establish that the pattern has no effect.

### 2.6 Reliability over complete streams and longer horizons {#sec:long-streams}

Across 128 edits on 12x12 mazes, output reuse at K=1 scores 73.8, down from 77.1 over 32 edits. Restart scores 37.6, carry 85.8, the spatial gate 86.9 and the GRU adapter 88.5. At K=8, output reuse and restart both score 99.8. The frozen operating point is unchanged (+0.00 pp, [−0.12, +0.13]), but success over a complete stream of 128 edits at K=8 is 96.2 for restart and 91.6 for output reuse. For streams of 32 edits at K1, restart accumulates 20.1 errors over 32 edits, output reuse 7.3, carry 4.7 and the spatial gate 4.7. The edit process also changes the problem distribution over time. On 12x12 mazes, restart's accuracy at K=1 falls from 39.1% over frames 1 to 8 to 35.7% over frames 25 to 32. At K=8, it falls from 99.94% to 99.78%. Uniform passage toggles gradually move the mazes away from the spanning tree distribution. Over the same frames, output reuse at K=1 falls from 80.4% to 73.1%, carry from 86.3% to 84.2%, and the spatial gate from 88.4% to 83.8%. Relative to restart, output reuse therefore loses accuracy a little faster than latent reuse (supplementary curves by frame).

The similar mean frame scores conceal a 4.6 pp descriptive difference in the chance of finishing a stream of 128 edits without error. Accuracy on individual frames is insufficient to describe reliability over a complete episode.

### 2.7 Accuracy versus measured cost {#sec:cost-curves}

Using the same K matches calls to the recursive core, while each learned adapter adds its own cost. Figure \ref{fig:accuracy-latency} plots every principal system at all five budgets against the original measured amortized latency at batch size one, including frame zero. Both accuracy and latency use the same eight timing roots and checkpoints from seed 29. This small sample provides a separate view of the five seed accuracy results. Figure \ref{fig:accuracy-cost} in the appendix gives the corresponding throughput costs measured with five seeds, 256 roots and batch size 64. Section \ref{sec:cost} retains the original cost breakdown by stage and the classical reference implementations.

In the frozen maze comparison, restart and output reuse both use K8. Output reuse adds a projection and saves no recursive cycles. Failure to save 25% at this selected point has a limited interpretation; the budget crossover is supported by the full curves. Section \ref{sec:frozen-comparison} retains the frozen comparison unchanged.

We also check how measurement overhead affects latency by comparing the original instrumented implementation with a lean inference path, using the same existing checkpoints. The lean path stores only tensors needed for the next prediction and removes explicit stage clocks and copies of the audit history. We time complete streams between synchronization boundaries. Both modes include input transfer and CPU output availability, while observation construction and oracle scoring occur outside the timed region. Tensor validation remains enabled and may synchronize internally. Restart keeps no prior tensors, output reuse keeps logits, and latent reuse keeps the prior observation and state. This comparison measures sensitivity to a specific implementation change; it does not estimate an ideal latency.

This exploratory check uses seed 29 and the first two stored roots from each primary suite, with 32 edits plus frame zero. It covers all principal policies and five budgets, with three repetitions in shuffled, interleaved blocks. All actions agree across paths and repetitions. Table \ref{tab:timing-sensitivity} reports timing at the frozen budget choices. Figure \ref{fig:lean-latency} in the appendix gives every accuracy and latency point from the matched sample. The Original column uses two roots and a clock around the complete stream, whereas the frozen latency in Table \ref{tab:primary-cost} averages individual frame timings over eight roots; their output reuse/restart ratios therefore need not match.

For this small timing sample, the lean path's output reuse/restart cost ratios at the frozen budgets are 1.098, 0.745 and 0.653 on maze12, circuit32 and circuit48, respectively. Both circuit point ratios fall below 0.75, showing sensitivity to implementation overhead. This sample cannot establish a 25% saving that preserves accuracy across training runs, and the original accuracy failure under circuit structural shift remains.

<!-- figure:accuracy-latency -->

<!-- table:timing-sensitivity -->
<!-- BEGIN timing-sensitivity -->
| Suite | Policy | K | Original ms | Lean ms | Lean / orig. |
|---|---|---|---|---|---|
| maze12 | restart | 8 | 100.93 | 84.53 | 0.837 |
| maze12 | answer_only | 8 | 106.47 | 92.83 | 0.872 |
| circuit32 | restart | 2 | 42.94 | 27.16 | 0.633 |
| circuit32 | answer_only | 1 | 35.97 | 20.22 | 0.562 |
| circuit48 | restart | 2 | 43.95 | 29.29 | 0.666 |
| circuit48 | answer_only | 1 | 34.09 | 19.14 | 0.561 |
<!-- END timing-sensitivity -->

### 2.8 Uncertainty across roots and training seeds {#sec:variability}

Variation among examples and variation among training runs answer different questions. Their raw standard deviations do not tell us how much each contributes to uncertainty in the mean, especially when there are 256 roots and only five seeds. For exploratory policy rankings, we use crossed intervals; the frozen decision retains intervals conditional on the trained seeds. At the maze operating point, the root SD of the paired delta is 1.18 pp and the training seed SD is 0.20 pp. The ordinary circuit point gives 2.31 and 0.17 pp, and the point under structural shift gives 9.93 and 6.05 pp. Under structural shift, deltas for individual seeds range from −4.7 to −18.6 pp. Resampling seeds as well as roots widens the circuit interval to [−1.01, −0.36] and the structural shift interval to [−14.95, −5.15]. Restart's five backbones give similar results (37.2 ± 1.9 at K=1 on 12x12 mazes); the GRU adapter's K=1 accuracy ranges from 76.9 to 92.6 across seeds. Training five independent backbones per size allows these statements about seed variation. The pilot used one shared backbone.

## 3. Related work {#sec:related-work}

**Reuse across related inputs.** Several methods already reuse computation when an input changes. HRM-Agent compares carrying and resetting hierarchical latents in dynamic mazes trained through reinforcement learning, without distinguishing changes that affect the solution from those that do not [HRM-Agent]. StreamDEQ carries representations between video frames [StreamDEQ]. Learned warm starts train an initializer through a fixed-point solver [Warm]. CoFRe mixes old and fresh token states between steps of masked generation [CoFRe], while Fixed Point Diffusion Models reuse fixed points between denoising steps [FPDM]. Pointer Graph Networks learn which node pointers to retain after dynamic operations [PGN]. These precedents make it useful to measure both the size of an observed edit and how much it changes the required answer.

**Recurrent solvers and adaptation.** Our solver follows the recursive architectures of TRM and the Recursive Stem Model [TRM, RSM]. GRU gating provides a generic learned alternative to a constrained mixture of old and fresh state [GRU]. Skip RNN learns when to skip state updates [Skip], and Probabilistic TRM perturbs latents to explore alternative trajectories [PTRM]. Neural cellular automata also learn recurrent local dynamics, including regeneration and pathfinding [NCA, PathNCA]. We compare spatial retention with carry, a global gate, a GRU and a residual adapter. The learned adapters receive the same observable information and are compared under an explicit budget of neural block calls.

**Incremental computation.** Incremental GNN systems maintain embeddings as graphs change [NeutronRT, RIPPLE++]. D* Lite retains search information when costs change [D* Lite]. These methods follow dependencies and motivate measuring how far an edit's consequences extend beyond the edited location. We examine that distinction in neural solvers across two task families and several computation budgets. Recurrence, forgetting gates, warm starts and incremental planning are all established ideas.

**Specific comparison with the closest work.** HRM-Agent studies carrying and resetting state for dynamic maze navigation learned through reinforcement learning. It explicitly notes that its changed and unchanged conditions do not distinguish consequential edits from irrelevant ones [HRM-Agent]. StreamDEQ studies recycling at small iteration counts, trajectories across frames and some degradation when more steps are unrolled [StreamDEQ]. We add a comparison between compact distributions over previous outputs and retained latents. We measure how their ordering changes across budgets and maze or circuit structures, whether updating corrects an old prediction, and how an edit's consequences relate to correction. We also measure cost and reliability over complete streams. The study asks which jointly trained reuse system helps under a given set of constraints; state reuse itself is established.

## 4. Problem setting and protocol {#sec:protocol}

**Streams.** Each episode starts with a base problem `x_0`. A sequence of `T` observable edits produces `x_1, …, x_T`. The policy receives the current and previous problem descriptions, the explicit edit derived from them, and its own previous state and prediction. Its weights stay fixed during evaluation. It runs exactly `K` outer cycles at every frame before emitting a prediction, and every policy starts fresh at frame zero. The primary streams use a fixed budget throughout `T = 32` edits; streams with `T = 128` test longer horizons.

**Tasks.** Mazes are grids with four neighbors per node and a fixed start and goal. Each node predicts north, east, south, west, goal or unreachable. A frame is exact if following the argmax actions from the start gives a shortest route or correctly reports that no route exists. We generate each base using randomized Kruskal to form a spanning tree, then add extra passages independently with probability 0.15. Every fourth base has a vertical cut through the middle to include unreachable cases. Each edit toggles a potential passage chosen uniformly at random. Repeated edits can undo earlier changes, disconnections remain in the data, and no edits are rejected. Circuits are random directed acyclic graphs over INPUT, AND, OR, XOR and NOT, with 8 inputs. The policy predicts every gate value; a frame is exact if every non-input node is correct. An edit is equally likely to flip an input bit or substitute a binary operator. Node presentation order is randomly permuted once per root and then held fixed. Observations never contain internal gate values. For the suites with structural shift, every gate takes the preceding gate as one parent. The wiring therefore gives chain depth `nodes − inputs`.

**Sizes and splits.** We train on 12x12 and 16x16 mazes, each with 1,024 training and 256 validation bases, and on circuits with 32 and 64 nodes, each with 2,048 and 256 bases. Tests of structural shift use 20x20 mazes for both maze models, and deep circuits with 48 and 96 nodes for the models trained on 32 and 64 nodes. These shifted suites are used only for evaluation. Base problems are assigned to splits before any descendants are generated. Every frame derived from a base stays in that split, and generation stops if an exact duplicate crosses splits. Each test suite has 256 ordinary bases, with another 128 reserved maze challenge identities per size. Their identity hashes were committed with the frozen protocol before generation.

**Consequence strata.** For each edit, the evaluator records the fraction of scored nodes whose target changes: the optimal action set in mazes or the gate value in circuits. The strata are low (≤ 0.1), middle and high (≥ 0.4). We use these strata to summarize interventions from a frozen prior state and the new analysis of frequencies in ordinary streams. They never enter the policy. Ordinary streams are sampled without conditioning on consequences. Changes in the required output do not identify which latent coordinates are stale.

**Record modes.** We keep separate records for deployed policies with fixed budgets, interventions from a common prior state, privileged diagnostics and classical references. The cost comparison between learned systems includes only deployed learned policies. Section \ref{sec:cost} gives the implementation details and audit accounting.

**Cost.** Before each measured stream, we run a warm-up and discard its state. We then time the stream's frames with device synchronization. Timing covers nine stages: observed-edit computation and validation, encoder, fresh initialization, adapter, recursive core, decoder, state and history copies, and input/output transfer. Amortized cost is `(initial solve + Σ edit costs)/(T+1)`. Accuracy jobs use batch 64 on the GPU, and we retain their timings. Dedicated latency jobs use a batch of one, one checkpoint per policy, eight test roots and three repetitions. We report dense multiply-accumulate counts together with elapsed time. Soft gates still execute dense work. Retaining state alone saves no computation; a saving requires fewer cycles.

**Ordinary circuit construction.** Circuit size includes every node, including eight INPUT nodes whose bits are independent fair binary draws. We add the remaining nodes in topological order, drawing each operator uniformly from AND, OR, XOR and NOT. A NOT gate takes one earlier node sampled uniformly. A binary operator takes two distinct earlier nodes sampled uniformly. The graph is randomly relabeled, and a separate permutation sets the presentation order for the whole episode. Each edit chooses between an input flip and a binary operator substitution with equal probability, then samples an eligible node uniformly. A substitution chooses one of the two other binary operators. If the circuit has no binary gate, the edit flips an input instead. Under structural shift, every successive non-input node must take its immediate predecessor as a parent; a binary gate's second parent is sampled from the other earlier nodes. This generator changes the number of nodes and the wiring distribution as well as depth, so the experiment cannot isolate the effect of depth.

## 5. Solver, reuse policies and training {#sec:solver}

**Backbone.** We use a solver with two latents, inspired by TRM; it is not a TRM reproduction. An encoder maps the observed node features to `e(x)`. For mazes these features are normalized coordinates and start/goal flags; for circuits they are a one-hot operator encoding and the input bit. Learned vectors `a_0, z_0` are broadcast to every node. Each outer cycle makes three calls to a shared transformer block `F` with two layers and pre-norm: `z ← F(e(x)+a+z)` twice, then `a ← F(a+z)`. A linear head reads `a`. Attention is restricted to observed neighbors and the node itself. Maze neighbors are connected by passages; circuit neighbors are connected by wires in either direction, with distinct learned biases. A learned relation bias for each head encodes direction. The maze solver has width 64 and two heads (100,954 parameters), and the circuit solver has width 128 and four heads (398,250 parameters). A budget of K always executes `3K` block calls and `6K` transformer layers. There is no halting mechanism.

**Reuse policies.** An adapter runs once at each edit, then supplies the state to the solver. All policies use the same solver architecture, but each deployed policy has its own jointly trained backbone. The output reuse policy retains confidence and probability mass outside the argmax. For each valid node $v$ at an edit frame $t>0$, its initialization is $[a_t^{\mathrm{init}}(v);z_t^{\mathrm{init}}(v)]=W\,\operatorname{softmax}(\operatorname{stopgrad}(\ell_{t-1}(v)))+b$, where $\ell_{t-1}$ is the previous output logit vector and the two halves of the affine projection replace the learned fresh latents. At frame zero it uses $a_0,z_0$; padding remains zero.

<!-- table:policies -->
| Policy | Initialization | Extra parameters (maze / circuit) |
|---|---|---:|
| restart | fresh `a_0, z_0` | 0 |
| carry | detached previous `a, z` | 0 |
| spatial_gate | per-node sigmoid `r_a, r_z` from a two-layer observed-context block; `r·old + (1−r)·fresh` | 9,052 / 11,132 |
| global_gate | same context, one scalar per latent per example | 9,052 / 11,132 |
| gru_adapter | `GRUCell` over concatenated old latents with the context as input | 65,082 / 221,530 |
| residual_adapter | unconstrained linear delta from the context added to old latents | 11,194 / 15,450 |
| answer_only | affine projection of the detached previous softmax output replaces `a, z` at edits; never reads old latents | 896 / 768 |

The four learned latent adapters use the same context architecture over the new observation. It receives old and new features, their difference, indicators of changed edges, old and new relation degrees, and the detached old latents. Two transformer layers with attention masked to neighbors process this context at width 16. On circuits we compare restart, carry, spatial_gate and answer_only. Auxiliary controls use the spatial gate's backbone: resets within radius 1, 2 or 3 of the edit; random resets at the rate calibrated on validation; carry with Gaussian noise (σ = 0.01); and a shuffled gate. Shuffling permutes the learned retention values across nodes without changing their multiset. No policy receives oracle information.

**Gradient contract.** The model generates its own previous states, which are detached at each version boundary. The adapter and the current rollout remain connected to the current loss through autograd. Tests check that gate gradients are finite and nonzero, that an optimizer step changes the gate, and that gradients do not cross version boundaries. A negative control deliberately detaches the state after the adapter and must fail the gradient check.

**Supervised objective.** For maze frame $t$, let $V$ contain every node that is not padding and let $A_t(v)$ contain all valid optimal actions at node $v$, including GOAL or UNREACHABLE where appropriate. We use the loss $L_t=-|V|^{-1}\sum_{v\in V}\log\sum_{a\in A_t(v)}p_t(a\mid v)$. Every valid optimal action contributes probability mass, so the loss does not choose arbitrarily among tied shortest paths. Nodes away from the designated start also contribute. Circuit loss is the mean binary-class cross-entropy over nodes that are neither inputs nor padding; input nodes have no supervised loss. Static training supervises only the final output after the sampled K cycles. Stream training averages the five terminal frame losses, covering frame zero and four edits, and accumulates their gradients before one optimizer update. States from the previous frame are detached, but the current rollout of K cycles remains differentiable. There is no auxiliary loss between cycles. The control with one edit averages the losses from its fresh and edited frames. Examples of equal size therefore receive equal weight within a batch.

**Training.** Each static backbone receives 6,000 AdamW updates. The learning rate starts at 0.001 and falls to 0.0003 after 75% of training, with no weight decay, gradient clip 1, and batch 64 for mazes or 128 for circuits. Each step samples K from {1,2,4,8,16}. We train five independent seeds (29, 43, 71, 101, 137) per size. Every backbone must pass a frozen criterion: exact accuracy on ordinary validation data must be nondecreasing across K and reach at least 0.85 at K=16. All twenty passed. For each size and seed, all policy arms branch from separate copies of the same static pretrained checkpoint; each copy then receives 1,024 joint updates together with its own adapter, where applicable. Each stream has one fresh solve and four edits, with batch 64 and K sampled from {1,2,4,8} for mazes or {1,2} for circuits. We try two pairs of backbone/adapter learning rates, 0.0001/0.0003 and 0.0003/0.001. Restart receives the same stream training, so every subsequent reference to restart means the policy trained on streams. For each policy and size, validation selects the learning rate pair with the highest mean exact accuracy over four edits, seeds and K; ties favor the lower pair. The higher pair won for every adapter and the lower pair for restart at every size. Controls with equal training compute train restart, carry, spatial_gate and answer_only on examples with a single edit for 2,576 updates. Their forward calls match within 2% per seed.

**Maze and circuit lineage from the pilot.** Before the confirmatory run, we ran a pilot with a frozen backbone and a pilot trained on streams. Both used validation data only, with three adaptation seeds on one shared backbone per family. The pilot did not meet its preregistered criterion for latent reuse to improve on output reuse. On 12x12 mazes, spatial gate minus answer-only was −7.9 pp [−11.5, −4.4] at K=1 and −5.4 pp [−6.8, −4.1] at K=8. No latent policy had a positive lower bound at any K. Output reuse exceeded restart by +32.5, +12.1 and +3.9 pp at K=1, 2, 4, then fell below it by 0.2 pp at K=8. These validation results selected answer-only for the confirmatory comparison.

A difference between deployed policies therefore includes several effects: initialization, the weights learned with that initializer, the selected learning rate and their interaction during training. Interventions from a common prior state and lesions of the previous output test narrower questions within a trained system. Neither establishes a general causal advantage for the selective gate.

## 6. Frozen protocol, primary endpoint and outcome {#sec:frozen-comparison}

We committed the primary comparison before generating any test root. Table \ref{tab:frozen-protocol} lists the frozen choices. Other comparisons are exploratory. The committed execution matrix contains 1,314 jobs and their configuration hash.

<!-- table:frozen-protocol -->
| Frozen item | Specification |
|---|---|
| Outcome | Mean post-edit exact accuracy over 32 edits; each base root has equal weight. |
| Co-primary suites | 12x12 mazes, ordinary 32-node circuits, and reserved 48-node structural-shift circuits. |
| Reuse policy | Answer-only, selected from development evidence. |
| Operating points | Maze: answer-only K8 versus restart K8. Both circuit suites: answer-only K1 versus restart K2, unchanged under structural shift. |
| Validation rule | Highest-accuracy restart point; then the least-cost answer-only point within one percentage point of that accuracy. |
| Target | At most one percentage point lower accuracy and at least 25% lower amortized cost. |
| Decision rule | Noninferiority: accuracy-difference lower 95% bound above −1 pp. Saving: cost-ratio upper bound below 1; 25% target: upper bound at most 0.75. All three primary suites must pass. |
| Statistics | 10,000 paired-root bootstrap draws, seed 64037; training seeds averaged within roots. Crossed intervals also resample training seeds. |

Validation had already shown a loss of approximately 10.8 points at the frozen point under structural shift. Development cost ratios were 0.91 for mazes and 0.77 for circuits. We kept the selected points and recorded this unfavorable power assessment before seeing the test results. The supplement includes the full planning calculations.

**Outcome.** No primary suite reached the 25% cost target. The suite with structural shift also failed noninferiority by about ten points, as anticipated by the protocol's power table.

<!-- table:primary-accuracy -->
| Primary suite | Δ accuracy (pp) | Root 95% CI | Crossed 95% CI | Noninferior (root) |
|---|---:|---:|---:|---|
| maze12 | +0.07 | [−0.07, +0.23] | [−0.10, +0.39] | yes |
| circuit32 | −0.64 | [−0.94, −0.39] | [−1.01, −0.36] | yes |
| circuit48 | −9.74 | [−10.97, −8.54] | [−14.95, −5.15] | no |

<!-- table:primary-cost -->
| Primary suite | Batch-64 cost ratio | Root 95% CI | Batch-one cost ratio | Root 95% CI |
|---|---:|---:|---:|---:|
| maze12 | 1.0037 | [1.0029, 1.0045] | 1.0059 | [1.0052, 1.0067] |
| circuit32 | 0.9931 | [0.9917, 0.9945] | 0.7723 | [0.7685, 0.7774] |
| circuit48 | 0.9428 | [0.9422, 0.9434] | 0.7753 | [0.7691, 0.7797] |

Neither timing mode in Table \ref{tab:primary-cost} meets the 25% target. The accuracy failure under structural shift holds regardless of timing. Secondary results also miss the target. At the frozen points, accuracy differences from restart are +0.00 pp on maze16, −1.11 pp on circuit64 and −3.39 pp on circuit96. Over 128 edits, the differences are +0.00, +0.04, −0.78 and −1.08 pp on maze12, maze16, circuit32 and circuit64, respectively. The supplementary tables give the full estimates at these operating points. Sections \ref{sec:long-streams} and \ref{sec:variability} report horizon effects and variation across seeds separately.

Section \ref{sec:cost} breaks down measured costs at the frozen operating points. Those costs do not set a lower bound for other implementations or operating points. Curves across fixed budgets and the shifted suites were prespecified secondary analyses. Comparisons among reuse policies and the added diagnostic and timing analyses are exploratory.

## 7. Limitations {#sec:limitations}

The exploratory findings apply to the solvers and tasks measured here: 100k to 400k parameters, width 64 or 128, three block calls per cycle, two synthetic task families with exact evaluators, edits to one passage or one gate, and streams of 32 or 128 edits. Reuse policies were trained on streams of four edits, with K ≤ 8 for mazes and K ≤ 2 for circuits. Their degradation at K=16 could therefore reflect the budgets used in training. We cannot attribute it to reuse in general.

The timing results come from one GPU under a Windows display driver, with a 10 GiB allocator cap, two CPU threads and TF32 off. Dedicated batch-one latency uses one checkpoint and eight roots. It assigns a different value to a saved cycle than batch-64 throughput does. The protocol did not specify which measurement governed the cost target, so we report both; neither meets it. The classical references run in Python on the host CPU, which prevents a direct comparison of their stages with those of the GPU policies.

Joint training limits what the interventions can establish. The equal-compute comparison of stream exposure is an initial screen. Lesions isolate the content of the previous answer, while leaving changes to the jointly trained backbone in place. That confound applies to every jointly trained arm, and we could not make comparable lesions for latent adapters. Before the run, predicted power was low for the primary circuit and structural-shift contrasts, and the protocol proceeded anyway. Validation had also shown a large accuracy loss at the frozen point under structural shift, so that test failure was expected.

The study has no third task family, learned halting, sparse execution or external replication. Development and test streams share the stream seed constant. Each edit sequence also depends on its root's identity, so the sequences differ across the two sets. Neither the seed nor the root identity is a policy feature, though the shared constant remains a limitation. Consequence strata describe evaluator judgments and cannot identify invalid latent state. Reserved challenge pairs were chosen to meet consequence conditions, so their frequencies cannot estimate prevalence.

Five seeds give limited precision about variability across retraining, even when the resampling includes seeds. The principal deployed comparisons lack controls that reuse hard outputs or hold a shared backbone frozen across policies, which limits the causal claims.

## 8. Conclusion {#sec:conclusion}

Reuse gives a substantial accuracy gain at small budgets in the measured maze setting. By eight cycles, restart and output reuse have higher mean accuracy than the policies that retain latents, although the crossed interval leaves the global gate's difference from output reuse unresolved. Task, budget and training seed all affect the ordering. The residual adapter falls below output reuse at one cycle, while output reuse and the spatial gate retain large gains at equal budgets under circuit structural shift. When interventions supply the same prior state, carry remains inaccurate after edits with large consequences. Lesions with matched seeds show that previous-answer content contributes to the gain. They also leave a residual advantage whose source within the jointly trained system remains unresolved.

The frozen comparison of output reuse failed to achieve accuracy-preserving 25% cost savings on all three primary suites. Both timing modes give that decision, despite their substantially different circuit cost ratios. For solvers in the measured maze regime, compact output reuse provides a useful baseline at small budgets, and restart provides a strong reference at larger ones. A deployment claim requires complete streams evaluated against a declared accuracy target. Keep frame zero and training-seed identity in the records, and measure the overhead of reuse before treating fewer cycles as a saving.

## Reproducibility and artifacts

<!-- materials-access -->

The supplement contains enough evidence to recompute the primary comparisons. Its 7,680 derived episode records cover both policies, five seeds and 256 roots per suite at the three primary operating points. Each record retains accuracy after edits, the full sequence of edit errors and amortized timing. These records support recomputation of the primary paired-root and crossed intervals. The ZIP also includes the implementation and tests, frozen protocol and scientific configuration, saved numerical summaries for all suites, four sealed report directories, and the four sealed raw evaluation jobs used to examine frame zero. The supplement also includes diagnostic counts of the four possible outcomes in 96,000 records, each covering a complete episode for one seed, root and budget, 1,125 audit examples of raw transitions with problem descriptions, ordinary consequence frequencies, and all 900 timing-block records with their predictions. The included checker recomputes the diagnostic tables and verifies the audit sample from the package alone. Figures and tables regenerate from these saved inputs.

Replaying the full audit requires more than this package. The ZIP omits the 46.7 million raw actions covered by the complete audit, the full set of 1,314 job directories, all trained checkpoints and the full ledger. The complete original run archive is approximately 18 GB. It will be deposited in a public archive at publication; until then, it is available from the authors. The included aggregates reproduce the stated primary and diagnostic statistics, but each omitted raw action still needs its original record to be checked. Extracting the new diagnostic in full also requires the original raw jobs. The package supplies its bounded audit sample and complete episode aggregates for review.

<!-- appendix -->

## A. Complete maze contrasts {#sec:complete-contrasts}

<!-- table:maze-contrasts -->
<!-- BEGIN GENERATED MAZE CONTRASTS -->
| Policy | K | Δ vs restart (pp), 95% CI | Δ vs answer-only (pp), 95% CI |
|---|---:|---:|---:|
| restart | 1 | reference | -39.91 [-44.05, -35.83] |
|  | 2 | reference | -18.11 [-21.28, -15.05] |
|  | 4 | reference | -5.21 [-6.90, -3.69] |
|  | 8 | reference | -0.07 [-0.23, +0.07] |
|  | 16 | reference | -0.07 [-0.21, +0.06] |
| carry | 1 | +48.09 [+43.76, +52.47] | +8.18 [+5.48, +11.01] |
|  | 2 | +18.23 [+14.68, +21.85] | +0.11 [-1.29, +1.66] |
|  | 4 | +2.86 [+1.22, +4.59] | -2.35 [-2.91, -1.82] |
|  | 8 | -2.14 [-2.77, -1.56] | -2.21 [-2.82, -1.65] |
|  | 16 | -2.11 [-2.78, -1.49] | -2.18 [-2.82, -1.58] |
| spatial_gate | 1 | +48.24 [+43.95, +52.56] | +8.33 [+5.72, +11.09] |
|  | 2 | +19.60 [+16.27, +23.07] | +1.49 [+0.24, +2.85] |
|  | 4 | +5.08 [+3.51, +6.81] | -0.13 [-0.42, +0.19] |
|  | 8 | -0.31 [-0.60, -0.05] | -0.37 [-0.63, -0.16] |
|  | 16 | -2.79 [-3.57, -2.05] | -2.85 [-3.64, -2.13] |
| global_gate | 1 | +49.00 [+44.65, +53.41] | +9.09 [+6.60, +11.74] |
|  | 2 | +20.61 [+17.11, +24.26] | +2.50 [+1.32, +3.85] |
|  | 4 | +5.28 [+3.71, +7.01] | +0.07 [-0.18, +0.32] |
|  | 8 | -0.13 [-0.39, +0.08] | -0.20 [-0.42, -0.03] |
|  | 16 | -1.20 [-1.71, -0.75] | -1.27 [-1.76, -0.85] |
| gru_adapter | 1 | +50.95 [+46.42, +55.52] | +11.04 [+8.21, +13.98] |
|  | 2 | +19.57 [+15.83, +23.41] | +1.46 [-0.01, +3.01] |
|  | 4 | +4.82 [+3.14, +6.65] | -0.39 [-0.86, +0.09] |
|  | 8 | -0.39 [-0.72, -0.11] | -0.46 [-0.76, -0.22] |
|  | 16 | -1.12 [-1.61, -0.67] | -1.18 [-1.66, -0.77] |
| residual_adapter | 1 | +27.48 [+22.86, +32.28] | -12.43 [-15.99, -8.66] |
|  | 2 | +15.07 [+11.59, +18.69] | -3.04 [-4.56, -1.46] |
|  | 4 | +4.08 [+2.47, +5.86] | -1.12 [-1.56, -0.69] |
|  | 8 | -1.16 [-1.58, -0.79] | -1.23 [-1.65, -0.85] |
|  | 16 | -3.89 [-4.70, -3.15] | -3.96 [-4.75, -3.23] |
| answer_only | 1 | +39.91 [+35.83, +44.05] | reference |
|  | 2 | +18.11 [+15.05, +21.28] | reference |
|  | 4 | +5.21 [+3.69, +6.90] | reference |
|  | 8 | +0.07 [-0.07, +0.23] | reference |
|  | 16 | +0.07 [-0.06, +0.21] | reference |
<!-- END GENERATED MAZE CONTRASTS -->

## B. Frame-zero case studies {#sec:frame-zero}

A low average stream score can have two different sources: the solver may begin with a wrong answer, or it may lose accuracy after edits. To distinguish them, evaluation must retain the fresh solve at frame zero, the subsequent trajectory and training-seed identity. The cases below were selected after inspecting results. They illustrate how to diagnose a failure, without establishing that a checkpoint's behavior generalizes across solvers.

Circuit joint training used K in {1,2}; static backbone training used K in {1,2,4,8,16}. K=16 therefore tests a budget outside the joint-training range, despite its inclusion in static pretraining.

On ordinary circuit32, seed 137's output reuse policy scores 83.0% mean exact accuracy after edits at K=16, compared with essentially 100% at K=8. The initial solve already fails on 50 of 256 roots at K=16 and on none at K=8. Every policy starts fresh at frame zero, so these errors arise before an adapter acts at an edit boundary or a previous answer is reused. Accuracy then fluctuates around the lower level across frames. It does not progressively collapse. The spatial gate for the same seed scores 84.7% after edits at K=16; the separately stream-trained restart policy remains at 100%. Figure \ref{fig:budget-extrapolation} traces these checkpoints from frame zero through all 32 edits.

Part of this failure under budget extrapolation is already present at the initial solve. Joint training need not produce that failure: the other circuit32 seeds remain much more accurate, and circuit64 does not show the same collapse. Means over five seeds conceal the difference, which makes intervals that include seeds and the full curves for each seed essential to the diagnosis. These cases remain exploratory because they were chosen after inspecting the budget curves; they do not replace the primary endpoint.

The maze16 spatial-gate checkpoint at seed 71 shows both sources of failure. Correct fresh solves fall from 255/256 roots at K=8 to 224/256 at K=16, and accuracy then declines within the stream. Together, the cases show why the record needs frame zero, inference budget and training-seed identity. A low stream score by itself cannot identify stale state accumulating across edits as the cause.

<!-- figure:budget-extrapolation -->

## C. Consequence frequencies and reserved challenges {#sec:consequence-detail}

Table \ref{tab:consequence-frequencies} reports descriptive frequencies over 32 ordinary edits on each of 256 roots. Roots are the independent sampling units. All policies and training seeds see the same observations, so they add no observations to that denominator.

Before test generation, we reserved 128 challenge roots for each maze size. Sixty-four were constructed to produce addition candidates. Each has two disconnected rectangular spanning trees separated by a horizontal cut. The goal lies in the top tree, and the other tree holds at least half the nodes. A random vertical reflection prevents the goal's side from becoming a fixed cue. Adding a passage within a tree can leave every unreachable target unchanged; reconnecting the trees across the cut changes many targets. The other 64 roots come from the ordinary maze generator. All these candidates are separate from the 256 ordinary roots.

We enumerate every possible toggle of one passage for each candidate and classify it by the fraction of optimal-action sets that change. After shuffling roots, we choose at most one matched low/high pair per root, with quotas of 32 addition and 32 removal pairs. Constructed candidates qualify only for addition. If both edit types qualify, selection favors the less-filled quota and breaks ties randomly. One branch is sampled uniformly from each eligible low/high pool. The thresholds stay fixed at at most 0.1 and at least 0.4. If a quota cannot be filled, we report the gap without resampling roots or changing thresholds. Maze12 yields 64 pairs and maze16 yields 57 pairs. This selection deliberately enriches and balances the branches, so their frequencies cannot estimate those of ordinary edits.

<!-- table:consequence-frequencies -->
<!-- BEGIN consequence-frequencies -->
| Suite | Low % | Middle % | High % | Challenge n | Challenge L/M/H |
|---|---|---|---|---|---|
| maze12 | 95.12 | 3.60 | 1.28 | 128 | 64/0/64 |
| circuit32 | 64.82 | 31.88 | 3.30 | 0 | 0/0/0 |
| circuit48 | 65.47 | 24.66 | 9.88 | 0 | 0/0/0 |
| maze16 | 97.57 | 1.77 | 0.66 | 114 | 57/0/57 |
| circuit64 | 78.61 | 21.20 | 0.18 | 0 | 0/0/0 |
| circuit96 | 73.58 | 23.33 | 3.09 | 0 | 0/0/0 |
<!-- END consequence-frequencies -->

## D. Backbone development finding {#sec:backbone}

Masked-neighbor attention entered the study through a development comparison on 8x8 mazes with 512 training and 128 validation roots. With identical parameters, data, minibatch order and sampled budgets, dense relation-biased attention solved 30, 31, 26, 29 and 25 of 128 validation routes at K=1 to 16. It made 50 illegal moves at K=16. Masked-neighbor attention solved 60, 81, 105, 120 and 120 routes. The predeclared gate required nondecreasing solved routes and at least 50% at K=16. Dense attention failed; masked attention passed and was adopted only after that result. This comparison uses one seed and exploratory validation. We report it because every subsequent result depends on the choice.

## E. Original timing attribution and execution accounting {#sec:cost}

**Cost at the frozen operating points.** Output reuse saves no cycle at the maze point: both policies run eight, and the projection adds work. At the circuit point it saves one cycle, but the core occupies only 9% of the restart K1 frame at batch 64 on 32-node circuits (0.17 ms of 1.86 ms). State and history copies account for 57%, and transfer between host and device for 28%. Moving from K1 to K2 changes total cost by 7%. At batch one, the core accounts for 31% of the K1 frame and 78% of the K8 frame, making the same cycle saving worth 23%. On 12x12 mazes at batch 64, the core accounts for 33% of the K1 frame and 80% at K8. The measured overhead limits the value of a saved cycle at the selected circuit point. Savings could differ with another accuracy target or implementation, or under structural shift, where restart remains below ceiling even at high K.

Above K=2, the core dominates batch-one latency on the RTX 5070. A 12x12 maze restart frame costs 26.0, 35.1, 52.9, 88.6 and 160.3 ms at K=1 to 16. These totals include 7.7 ms for state and history copies, 3.4 ms for transfer and 1.7 ms for edit validation per frame, regardless of K. The adapter stage costs 1.85 ms for restart bookkeeping, 2.31 ms for the output reuse projection and 5.59 ms for the spatial gate's context block. The accuracy jobs record batch-64 throughput timing, which also supplies the frozen ratios of cost at the selected operating points. It divides the fixed stages across 64 examples, leaving the core a smaller share at low K. Because the two timing measures assign different values to a saved cycle (Section \ref{sec:frozen-comparison}), we report both wherever we claim a cost.

The classical references give an absolute cost reference for the learned systems. Python implementations of D* Lite and an event-driven circuit evaluator solve the same streams exactly on the host CPU. Their amortized times per frame are 0.25 ms for 12x12 mazes, 0.35 ms for 16x16, 0.54 ms for 20x20, and 0.015 to 0.035 ms for circuits. Even the cheapest neural point (batch 64, K=1) is 2 times slower on 12x12 mazes and 125 times slower on 32-node circuits, with far more errors. These references have separately labeled rows, and their stages cannot be compared directly with GPU policy stages. Their timings put the absolute costs of the learned policies in context, as introduced in Section \ref{sec:introduction}.

Two interrupted attempts resumed exactly: a training job saved a graceful-shutdown checkpoint at 372 s, and an evaluation job recorded a 0.76 s lower bound before a reset. Their quarantined directories remain preserved and contribute to no table.

**Record modes.** Records distinguish deployed policies (`fixed_budget_stream`), initializers forked from the same saved prior (`frozen_state_intervention`), classical references (`reference_solver`: D* Lite for mazes and an event-driven circuit evaluator for circuits, both on the host CPU), and an oracle impact-mask reset (`privileged_diagnostic`, `privileged=true`). Only `fixed_budget_stream` enters comparisons of learned-policy accuracy and cost; classical references occupy separately labeled rows. Sealed report jobs verify their inputs before reporting. Their recorded checks cover 46,719,642 raw actions across 766 checker rows. Execution seals and dependency checks cover the full 1,314-job matrix.

The completed matrix consumed 239.8 GPU-hours and 13.7 CPU-job hours on one RTX 5070 host across 1,335 attempts. There were no external charges, and electricity was not measured. An independent implementation audit reproduced the primary summaries and found no mismatch when rescoring 7,128 raw actions. It checks the recorded experiment; external replication remains absent. The separate exploratory timing check adds 0.70 GPU-job hours without new training. Coding-agent assistance supported implementation, analysis tooling and manuscript preparation.

<!-- figure:accuracy-cost -->

<!-- figure:lean-latency -->

## References

- [HRM-Agent] L. H. Dang, D. Rawlinson. HRM-Agent: Training a recurrent reasoning model in dynamic environments using reinforcement learning. arXiv:2510.22832, 2025.
- [TRM] A. Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny Networks. arXiv:2510.04871, 2025.
- [StreamDEQ] C. U. Ertenli, R. G. Cinbis, E. Akbas. Representation Recycling for Streaming Video Analysis. arXiv:2204.13492 (v5, 2026); earlier as Streaming Multiscale Deep Equilibrium Models, ECCV 2022.
- [GRU] K. Cho et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014, arXiv:1406.1078.
- [Warm] R. Sambharya, G. Hall, B. Amos, B. Stellato. Learning to Warm-Start Fixed-Point Optimization Algorithms. JMLR 25(166), 2024, arXiv:2309.07835.
- [Skip] V. Campos et al. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks. ICLR 2018, arXiv:1708.06834.
- [NCA] A. Mordvintsev, E. Randazzo, E. Niklasson, M. Levin. Growing Neural Cellular Automata. Distill, 2020.
- [PathNCA] S. Earle, O. Yildiz, J. Togelius, C. Hegde. Pathfinding Neural Cellular Automata. arXiv:2301.06820, 2023.
- [NeutronRT] Q. Wang et al. Incremental GNN Embedding Computation on Streaming Graphs. ICDE 2026, arXiv:2603.20622.
- [RIPPLE++] P. Naman, P. Agarwal, H. Haritas, Y. Simmhan. RIPPLE++: An Incremental Framework for Efficient GNN Inference on Evolving Graphs. arXiv:2601.12347, 2026.
- [RSM] N. Hakimi. Form Follows Function: Recursive Stem Model. arXiv:2603.15641, 2026.
- [PTRM] A. Sghaier, A. Parviz, A. Jolicoeur-Martineau. Probabilistic Tiny Recursive Model. arXiv:2605.19943, 2026.
- [CoFRe] A. Miele, Y. Qin, A. Carballo-Castro, J. Deschenaux, P. Frossard. Fixed-Point Masked Generative Modeling. arXiv:2605.31215, 2026.
- [FPDM] X. Bai, L. Melas-Kyriazi. Fixed Point Diffusion Models. CVPR 2024, arXiv:2401.08741.
- [PGN] P. Veličković et al. Pointer Graph Networks. NeurIPS 2020, arXiv:2006.06380.
- [D* Lite] S. Koenig, M. Likhachev. D* Lite. AAAI 2002, pp. 476–483.
