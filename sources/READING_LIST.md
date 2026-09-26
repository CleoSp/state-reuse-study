# Prioritized reading and audit scope

Reviewed September 8, 2026. This list records useful primary sources and the scope of the initial screening. It is not an exhaustive systematic review. Some entries were screened through their abstract or author documentation rather than a complete methods audit.

## Read first

1. **HRM-Agent — Long H Dang and David Rawlinson (2025).** The closest precedent. Read §§2.2.2 and 3.2 for carry/reset and changed/unchanged conditions. Initial screening: full HTML methods and analysis inspected. https://arxiv.org/html/2510.22832v1
2. **TRM — Alexia Jolicoeur-Martineau (2025).** Read the two-state recurrence and training pseudocode; inspect the implementation before porting. Initial screening: paper HTML and repository README inspected. https://arxiv.org/abs/2510.04871 ; https://github.com/SamsungSAILMontreal/TinyRecursiveModels
3. **GRU — Cho et al. (2014).** Read the reset/update equations. This prevents attributing ordinary gated recurrence to the new method. Initial screening: relevant full-text equations inspected. https://arxiv.org/html/1406.1078
4. **StreamDEQ — Ertenli, Cinbis, and Akbas (initially 2022).** Read how previous-frame representations initialize inference on new frames. Initial screening: version-5 HTML methods inspected. https://arxiv.org/html/2204.13492v5 ; https://ufukertenli.github.io/streamdeq/
5. **Learning to Warm-Start Fixed-Point Optimization Algorithms — Sambharya et al. (2023).** Read how the initializer is trained through a solver. Initial screening: primary abstract; full methods review was not part of this screening. https://arxiv.org/abs/2309.07835

## Read for competing explanations

6. **RSM — Navid Hakimi (2026).** Inspect the gradient contract, not just headline runtime. Initial screening: PDF methods and selected figures inspected; author repository README checked. https://arxiv.org/pdf/2603.15641 ; https://github.com/navidivan/rsm
7. **Skip RNN — Campos et al. (2017 preprint / ICLR 2018).** Existing learned compute-skipping precedent. Initial screening: primary abstract and author project/repository. https://arxiv.org/abs/1708.06834 ; https://github.com/imatge-upc/skiprnn-2017-telecombcn
8. **Growing Neural Cellular Automata — Mordvintsev et al. (2020).** Existing neural regeneration precedent. Initial screening: original Distill publication. https://distill.pub/2020/growing-ca/
9. **Pathfinding Neural Cellular Automata — Earle et al. (2023).** Iterative neural algorithms on mazes. Initial screening: primary abstract and publication entry; deeper method comparison remains. https://arxiv.org/abs/2301.06820
10. **Incremental GNN Embedding Computation on Streaming Graphs — Wang et al. (2026).** Distinguish semantic latent repair from exact update of stored embeddings. Initial screening: HTML method/equivalence sections inspected. https://arxiv.org/html/2603.20622v1
11. **Probabilistic Tiny Recursive Model — Sghaier, Parviz, and Jolicoeur-Martineau (2026).** A reason to include noise and alternative-initialization controls. Initial screening: primary abstract; the full implementation should be reviewed before reproducing its method. https://arxiv.org/abs/2605.19943
12. **D* Lite — Koenig and Likhachev (2002).** Classical incremental path-planning reference. Initial screening: author publication page. https://idm-lab.org/bib/abstracts/Koen02e.html

## Implementation and cost sources

13. **PyTorch official installer.** Consult again when installing; do not assume a historical version is current. https://pytorch.org/get-started/locally/
14. **PyTorch 2.7 release notes.** Documents introduction of Blackwell support and CUDA 12.8 wheels. https://pytorch.org/blog/pytorch-2-7/
15. **Runpod RTX 4090 listing.** Listed starting price $0.74/hour on the reviewed page, updated August 27, 2026. Actual offers and extra charges must be rechecked. https://www.runpod.io/gpu-models/rtx-4090
16. **Runpod A40 listing.** Listed starting price $0.49/hour on the reviewed page, updated August 27, 2026. https://www.runpod.io/gpu-models/a40

## Unresolved novelty checks

Search descendants of HRM-Agent and TRM, dynamic neural algorithm execution, learned selective state reset, graph-state repair, incremental neural program/circuit execution, learned initialization using previous solution and parameter changes, and neural cache invalidation. Separate inference-time activation updates from weight updates in continual learning. Check both papers and code; do not use search-result snippets to infer a missing method.

A positive novelty audit should identify the exact residual contribution. It cannot prove universal absence of prior art. All final related-work claims should be revisited immediately before submission.
