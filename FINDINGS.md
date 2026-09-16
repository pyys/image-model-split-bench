# Findings — what splitting an image model across GPUs costs

Measured 2026-09-11 (P104) and 2026-09-14 to 09-15 (V100). The reproduction procedure
and the engine constraints are in [`METHOD.md`](METHOD.md).

> **The short version.** Using four cards to split the model is **slower than using
> one.** Running those same four cards as independent workers is **about 3.9x faster.**
> Splitting the model is not a performance technique; it is **the last resort when the
> model will not fit on one card at all.**
>
> **And this is not a quirk of old cards or slow links.** Two hardware generations
> whose interconnect bandwidth differs by **16x** produced **the same result.**

**On both platforms, and in both layer split and tensor (row) split, throughput fell
as the model was divided across more cards.** The effect was far stronger for tensor
split. The evidence is below. [Results](#3-results)

---

## 1. What we measured and why

There are three tiers to using N GPUs for a diffusion model.

| Tier | Minimum VRAM per card | What is divided |
|---|---|---|
| **The whole model on every GPU** (worker parallel) | **about 34GB** | Images |
| **Modules spread across GPUs** (module separation) | **about 24GB** | Component modules (DiT / VAE / T5 …) |
| **One module split across GPUs** (intra-module split) | **about 4GB** | Layer ranges / weight rows |

> Figures are for FLUX.1-dev in FP16. Module sizes are DiT 23.8GB, T5-XXL 9.5GB,
> CLIP-L 0.25GB, VAE 0.17GB. Worker parallel by definition puts **the whole model** on
> each GPU, hence 34GB; module separation only needs room for **the largest module
> (the DiT)**, hence 24GB.
>
> **Every measurement in this document keeps the text encoder on the CPU (`te=cpu`),
> which is itself module separation.** So this is not a clean separation of the three
> tiers — it is **intra-module split measured on top of module separation.**

There is a common assumption — **"module separation is fine for image models, but if
you have to split inside a module you are better off lowering the quantization or
changing the model."** The direction looked right, but **no measurement could be
found.** This repository tests it.

---

## 2. Measurement environment

**The same measurement was repeated on two very different generations and
interconnects.**

| Item | **P104-100 x4** | **V100-SXM2-16GB x4** |
|---|---|---|
| Architecture | Pascal GP104, sm_61 | Volta GV100, sm_70 |
| VRAM | 8,109 MiB per card (32GB total) | 16,384 MiB per card (64GB total) |
| FP32 | about 6.6 TFLOPS | 15.7 TFLOPS |
| FP16 | **1/64 of FP32 rate** | 31.4 TFLOPS (tensor cores 125) |
| Memory bandwidth | about 320 GB/s (GDDR5X) | **900 GB/s (HBM2)** |
| **Interconnect** | **PCIe Gen1 x4** (about 1.0 GB/s, 0.86 effective) | **PCIe Gen3 x16** (about 15.75 GB/s) |
| NVLink | none | **none** (SXM2 adapted to PCIe) |
| Power limit | 180W | 300W |
| Measured | 2026-09-11 / 09-14 | 2026-09-14 |

Common to both — AMD EPYC 7232P 8-core, Ubuntu 24.04.
Card specifications are published figures [Notes and Caveats 1)](#notes-and-caveats).

**The interconnect differs by about 16x.** That is the axis this document turns on.

### Build

| sd.cpp | ggml |
|---|---|
| `68f3d6d` (2026-07-04) | `eced84c` |

**Both platforms were measured from the same commit.** Only the V100 side was rebuilt
with `CMAKE_CUDA_ARCHITECTURES=61;70` to add sm_70. The only difference between cells
is card count and split mode. **There is a reason this commit was used instead of
current master** [Notes and Caveats 2)](#notes-and-caveats).

### Model

A FLUX.1-dev derived DiT **quantized to Q4_0**, converted from the FP16 original
(23GB) with `-M convert --type q4_0`.

| Module | Size |
|---|---|
| **DiT (Q4_0)** | **6,482 MB** (VRAM) |
| T5 (`t5xxl-q8_0`) | 4,826 MB (RAM) |
| CLIP-L | 235 MB (RAM) |
| VAE | 160 MB |

```
flux: depth = 19, depth_single_blocks = 38, hidden_size = 3072, num_heads = 24
```

> **Why Q4.** The P104 has 8GB per card, and the point of this experiment is to
> **use single-card loading as the baseline** and measure what layer and tensor
> splitting cost against it. **The FP16 original fits on a V100 16GB, but the same
> Q4_0 was used there to keep the comparison identical.**
> **Image quality was not measured.** What Q4 quantization costs is not what this
> experiment is about.

### Fixed conditions

| Item | Value |
|---|---|
| Resolution | 512 x 512 (4,096 tokens) |
| Steps | 15 |
| Sampler / CFG / guidance | euler / 1.0 / 3.5 |
| Seed | 42 |
| Prompt | `a weathered brass compass on a wooden desk, morning light through a window, shallow depth of field` |
| Flags | `--diffusion-fa` `--vae-tiling` |
| Module placement | `te=cpu, vae=cpu` (intra-module split runs) |

**Pinning CFG at 1.0 matters.** FLUX dev is distilled and does not use CFG. Leaving the
default of 7.0 runs the uncond path as well, **doubling the cost per step** and making
the numbers incomparable to anything else.

**`--diffusion-fa` costs nothing, so it was made a fixed condition.** A separate check
found it cut the compute buffer by **46%** (302 → 164 MB) while running **3% faster**
(3.13 → 3.04 s/step).

---

## 3. Results

### 3-1. The headline — what four cards buy you

512 x 512, 15 steps, Q4_0. Times are **steady-state sampling** (excluding model load,
prompt encoding and VAE decode).

**P104-100 x4**

| Configuration | Cards used | s/step | Per image | img/min | **Speed vs one card** |
|---|---|---|---|---|---|
| **One card alone** (three idle) | 1 | 2.69 | 40.4s | 1.49 | **100%** |
| Layer split, 4 cards | 4 | 3.21 | 48.2s | 1.25 | **84%** |
| Tensor (row) split, 4 cards | 4 | 16.87 | 253.1s | 0.24 | **16%** |
| **Worker parallel, 4 copies** | 4 | — | 46.9s (x4 at once) | **5.11** | **344%** |

**V100-SXM2-16GB x4**

| Configuration | Cards used | s/step | Per image | img/min | **Speed vs one card** |
|---|---|---|---|---|---|
| **One card alone** (three idle) | 1 | 0.470 | 7.05s | 8.52 | **100%** |
| Layer split, 4 cards | 4 | 0.538 | 8.06s | 7.44 | **87%** |
| Tensor (row) split, 4 cards | 4 | 3.03 | 45.5s | 1.32 | **15%** |
| **Worker parallel, 4 copies** | 4 | — | 7.26s (x4 at once) | **33.06** | **388%** |

**Three things to read off this.**

1. Layer split is slower than one card. It uses more cards to go slower.
2. Tensor split uses three more cards to deliver **15~16%**. **The same on both
   platforms.**
3. The same four cards as independent workers give **344~388%**.

> **Mind the baseline for worker parallel.** This table is against **one card from the
> 7-cell run**. The figures in 3-3 are against **one worker from the worker run**, so
> they differ slightly (P104 344% against 352%, V100 388% against 393%). The two
> baselines are not identical because the VAE sits in a different place — `vae=cpu` in
> the 7-cell run, `vae=cuda` in the worker run. **These are not typos.**

### 3-2. Intra-module split — seven cells, both platforms

The primary metric is **warm s/step**, taken from the last step in the progress log
[Notes and Caveats 3)](#notes-and-caveats).

| Cell | Cards | Mode | **P104 s/step** | **Speed vs 1** | **V100 s/step** | **Speed vs 1** |
|---|---|---|---|---|---|---|
| 1card | 1 | single | **2.69** | **100%** | **0.470** | **100%** |
| L2 | 2 | layer | 3.02 | **89%** | 0.500 | **94%** |
| L3 | 3 | layer | 3.17 | **85%** | 0.513 | **92%** |
| L4 | 4 | layer | 3.21 | **84%** | 0.538 | **87%** |
| R2 | 2 | row | 12.72 | **21%** | 1.74 | **27%** |
| R3 | 3 | row | 14.34 | **19%** | 2.35 | **20%** |
| R4 | 4 | row | 16.87 | **16%** | 3.03 | **15%** |

**Both modes, both platforms, slower with every card added.** Layer split gently, row
split steeply. **Neither gains anything.**

### 3-3. Worker parallel — one to four copies

`te=cpu`, **the VAE on each worker's own GPU**, `--batch-count 4` (prompt encoding runs
once per worker, verified via the `cond` column)
[Notes and Caveats 4)](#notes-and-caveats).

| Workers | P104 per image | P104 img/min | **vs 1 worker** | **V100 per image** | **V100 img/min** | **vs 1 worker** |
|---|---|---|---|---|---|---|
| 1 | 41.28s | 1.45 | **100%** | **7.13s** | 8.42 | **100%** |
| 2 | 41.47s | 2.89 | **199%** | 7.20s | 16.67 | **198%** |
| 3 | 46.90s | 3.84 | **264%** | 7.28s | 24.73 | **294%** |
| 4 | 46.94s | 5.11 | **352%** | 7.26s | **33.06** | **393%** |

**The V100 is nearly linear.** Time per image grows only from 7.13 to 7.26 seconds,
**+1.8%**. The P104 bends from the third worker, 41.3 to 46.9 seconds, **+13.7%**.
**The difference is cooling** — from three workers the P104 starts throttling
thermally and its SM clock drops 14.2%
[Notes and Caveats 4)](#notes-and-caveats).

**So the P104 figure of 352% is a floor imposed by cooling, and 393% is what appears
when there is thermal headroom.**

---

## 4. Why layer split buys nothing

**Because only one card computes at a time.** There are no micro-batches, so the
pipeline never fills.

Sampling GPU utilization during layer split, at one-second intervals
[Notes and Caveats 5)](#notes-and-caveats):

```
GPU3 100%  /  GPU0 0%   GPU1 0%   GPU2 0%
GPU2 100%  /  GPU0 0%   GPU1 0%   GPU3 4%
```

**One card at 100% and the rest at 0%, rotating.** The engine documentation says as
much [{ref 4.}](#references)

> Layer split disables single-device segmented execution and **next-segment prefetch.**

This is the degenerate, pipeline-less form of pipeline parallelism, usually called
naive model parallelism. **PipeFusion exists precisely to fill that empty pipeline**
[{ref 2.}](#references), reporting **2.01x / 1.48x / 1.10x** latency reductions at
1024 / 2048 / 8192px against existing parallel methods. It is not implemented in
sd.cpp.

**This behaviour is not specific to this engine.** ComfyUI-MultiGPU also places
components on chosen GPUs but states that **"workflow steps still execute
sequentially"** [{ref 8.}](#references)

So **"more cards does not make it faster" is not a discovery; it follows from the
definition of the method.** What this document measures is **not how much faster but
how much slower**, and the answer is **84% on P104 and 87% on V100** at four cards.

**The cost of layer split did respond to more bandwidth.** It crosses a boundary only
once per segment, so the traffic is small and the improvement showed up. **Row split
is the opposite** (5-4).

---

## 5. Why tensor (row) split is worse

### 5-1. The split itself worked

```
Diffusion model row split: 304 tensors (6348.4 MB) split across 4 devices (main CUDA0)
```

All 304 weight tensors were split by row across four cards. This is not a fallback.
**Zero fallbacks on both platforms.**

### 5-2. But the cards do not work evenly

Utilization during sampling [Notes and Caveats 5)](#notes-and-caveats):

| | Utilization | VRAM |
|---|---|---|
| **CUDA0 (main)** | **79~82%** | **2,345 MiB** |
| CUDA1 | 18~21% | 2,021 MiB |
| CUDA2 | 20~21% | 2,021 MiB |
| CUDA3 | 26~28% | 2,021 MiB |

This is clearly different from the layer-split pattern, and **all four cards do run at
once.** But **main sits at 80% and the rest at 20%**, and main also holds 324 MiB more
VRAM (presumably the un-split embeddings, norms and graph I/O)
[Notes and Caveats 6)](#notes-and-caveats).

### 5-3. Tensor parallelism is structurally unsuited to diffusion

**This is not a peculiarity of our hardware.** The xDiT paper, describing an inference
engine built specifically for DiTs, reaches the same conclusion on far better
equipment [{ref 1.}](#references) [{ref 10.}](#references)

> **TP exhibits poor scalability** … communication cost **proportional to sequence
> length**, resulting in poor scalability.

As a result, tensor parallelism **does not even appear** in their FLUX.1-dev 1024px
comparison.

| Environment | Tensor parallelism |
|---|---|
| 8x A100 (**NVLink**) | excluded from comparison |
| 8x L40 (PCIe) | excluded from comparison |
| SD3 evaluation | excluded for *"significant time and memory inefficiencies"* |

**It is not worthwhile even over NVLink.** Tensor parallelism's traffic is proportional
to sequence length, and **diffusion processes the whole latent every step, so the
sequence is long.** 512px alone is 4,096 tokens.

For the same reason **LLMs reach the opposite conclusion.** LLM token generation
handles one token per step, so the traffic is small. On **the same cards and the same
link, tensor parallelism gave the fastest token generation** — 2.3x over layer split
on P104 and 1.7x on V100 → [Related documents](#related-documents)

### 5-4. The interconnect is not the cause — measured

**Bandwidth went up 16x, which is 2.8x more than compute and 5.7x more than memory
bandwidth, and the tensor-split penalty barely moved.**

| | P104 (Gen1 x4) | V100 (Gen3 x16) |
|---|---|---|
| R2 | 21% | 27% |
| R3 | 19% | 20% |
| **R4** | **16%** | **15%** |

**Working the overhead backwards, it shrank in step with compute (5.7x), not with
PCIe (16x)** [Notes and Caveats 7)](#notes-and-caveats). **So interconnect bandwidth
is not the governing factor.** That is the settled result of this measurement.

**A consistent hypothesis — the inefficient GEMMs that row split creates.** Row split
turns one matrix multiply into N thin ones. The total arithmetic is unchanged, but
**arithmetic intensity falls and per-card efficiency with it, and because that loss
scales with GPU compute performance, the ratio survives a faster card.** It also
explains why R4 on the V100 got marginally slower — **tensor cores gain most on large
matrices, so a matrix cut into pieces loses more against theoretical peak.** The
utilization pattern in 5-2 (main 80%, the rest 20%) points the same way. **If
communication were the bottleneck all four cards would be waiting.**

---

## 6. Relation to prior work

Separating what this measurement newly answers from what was already answered.

| Claim | Prior source |
|---|---|
| Tensor parallelism scales poorly for diffusion | **exists** [{ref 1.}](#references) |
| Naive layer split leaves the pipeline empty | **exists** [{ref 2.}](#references) |
| Mainstream tooling also runs components sequentially | **exists** [{ref 8.}](#references) |
| Multi-GPU can be slower than single GPU (training) | **exists** [{ref 9.}](#references) |
| **Single GPU vs naive layer split inference cost (84~87%)** | **not found** |
| **Row split delivering 15~16% of one card** | **not found** |
| **That penalty not responding to a 16x interconnect improvement** | **not found** |

Prior work mostly **compares advanced techniques against each other on A100 or L40
class hardware.** Nothing was found that measures what actually works, and what does
not, in a real consumer setup.

---

## 7. Limits

| # | Item |
|---|---|
| 1 | **One run per cell.** Only the P104 seven cells were **measured twice, three days apart, agreeing within ±0.4%**. Worker parallel has three steady-state samples |
| 2 | **Only 512 x 512 was measured.** No resolution sweep |
| 3 | **Image quality was not measured.** What Q4_0 costs is out of scope |
| 4 | Row split was measured **only on the `68f3d6d` build**. It cannot be reproduced on current master |
| 5 | The main-card concentration in row split is **inferred from utilization and VRAM patterns** |
| 6 | **The GEMM inefficiency in 5-4 is a hypothesis.** No kernel profiling was done |
| 7 | Between the two platforms, **card, link, VRAM and power limit all changed at once.** The factors were not decomposed |
| 8 | Only one model family (a FLUX-class DiT) was measured. **UNet architectures such as SDXL are structurally different** and may suit layer split differently |
| 9 | Sequence and patch parallelism (USP, DistriFusion [{ref 3.}](#references), PipeFusion [{ref 2.}](#references)) **could not be used in this environment and were left out.** The reasons are in the companion document |
| 10 | The V100s are **SXM2 modules adapted to PCIe.** Power delivery and cooling differ from factory PCIe cards |
| 11 | **The power limit engaged differently per cell.** Below |

**On the power limit.** On the P104, `SwPowerCap` samples ranged from 56% of the run
on 1card down to 6% on R4. The card's power limit is **the default 180W** (217W
maximum) and was not lowered — it is the standard operating condition for a P104-100.
One card running flat out hits that limit; spread across several cards, each is loaded
less and hits it less often.

**So this is not a measurement defect but what actually happens when you drive one of
these cards fully.** Still, because the baseline is the cell most often against the
limit, **the P104 layer-split cost (84%) may be somewhat understated.** On the V100,
power throttling stayed between 0 and 8% across all cells.

---

## Related documents

| Document | Contents |
|---|---|
| **[METHOD.md](METHOD.md)** (this repository) | Reproduction and caveats — how row split broke on current master, the 2 GiB per-card reserve, the engine choosing the card count for you, a PyTorch requirements comparison, build and run procedure, throttle criteria |
| **[`layer-tensor-parallel-bench`](https://github.com/pyys/layer-tensor-parallel-bench)** (separate repository) | The same cards and the same link measuring an **LLM (27B)**. **It reaches the opposite conclusion** — tensor parallelism gave the fastest token generation (1.7~2.3x depending on the platform) |

---

## Notes and Caveats

**1)** The card specifications in section 2 (TFLOPS, memory bandwidth, PCIe generation)
are **published figures, not measurements.** The only measurements here are this
document's results. The interconnect was checked against the real link state via
`nvidia-smi` `pcie.link.gen.current` / `width.current`, and the P104-100 is **limited to
Gen1 x4 by the card itself** (not by a riser). The 0.86 GB/s effective bandwidth is a
separate measurement.

**2)** There is a reason this commit (`68f3d6d` plus ggml `eced84c`) was used.
**Tensor (row) split does not work on current master** — the underlying library lost a
required capability. The account and the reproduction steps are in the companion
document → [METHOD.md](METHOD.md)

**3)** The primary metric is **warm s/step, not total sampling time.** The seven cells
produce one image each, so **total sampling time includes the first-image warm-up.**
On the V100 1card, total sampling time is 12.96 seconds while warm s/step works out to
7.05. The same holds on P104 (48.90 against 40.4). **Warm s/step excludes that overhead
and applies identically to both platforms.** There is also a cross-check that warm
s/step is the right metric — steady state for a single worker on the V100 is **7.13
seconds**, matching the 7.05 derived from the seven cells.

Separately, **absolute single-card performance improved 5.7x** (2.69 → 0.470 s/step).
That figure feeds the derivation in 5-4.

**There was no thermal effect.** Across all seven cells, thermal throttle samples were
**0%**, with peaks of 60 °C on P104 and 59 °C on V100. **The intra-module split
measurements are not contaminated by temperature.**

**4)** What separates P104 from V100 in worker parallel is **cooling.**

| | P104 | V100 |
|---|---|---|
| Thermal throttle, 3 workers | **11%** | **0%** |
| Thermal throttle, 4 workers | **24%** | **0%** |
| Peak temperature, 4 workers | **88 °C** | 72 °C |
| SM clock (early → late) | 1,873 → **1,607 MHz** (−14.2%) | unchanged |

The P104's **−14.2%** clock drop matches its **+13.7%** increase in time per image
almost exactly. **The P104 figure of 352% is a floor imposed by cooling.**

**Why the VAE moved onto the GPU for worker parallel only** — four workers running a
CPU VAE at once serialize on eight cores, so the measurement would be **CPU contention
rather than GPU throughput.** What is directly comparable to the seven cells is
therefore **sampling time**; total time is under different conditions and is not
comparable.

**5)** The **GPU utilization figures in sections 4 and 5-2 were all measured on P104.**
The same table could not be produced on the V100 — at one-second sampling, GPU0's
utilization **never once** exceeded 50% under three- and four-card layer split, because
each card's share had become that short. **That points the same way as the explanation
in section 4**, but one-second sampling is too coarse for this purpose. **It is not
used for more than illustration.**

**6)** The main-card concentration in 5-2 is **an observed pattern; the implementation
was not verified against the source.** That the auxiliary cards spend most of their
time waiting, and that it worsens as cards are added (21% → 19% → 16%), is
**consistent with aggregation concentrating on main** — but it is not asserted.

**7)** The derivation behind 5-4. Writing the split cost as
`T_split = T_compute + T_overhead`, the slowdown against one card is `1 + O/C`.

```
P104   6.27 = 1 + O/C   ->   O/C = 5.27      (16% speed = 6.27x the time)
V100   6.45 = 1 + O/C   ->   O/C = 5.45      (15% speed = 6.45x the time)
```

**O/C moved by only +3.4%.** Compute C got 5.7x faster while the ratio held, which
means **the overhead O also shrank by roughly 5.5x.**

| Factor | P104 → V100 |
|---|---|
| **PCIe bandwidth** | **16x** |
| Memory bandwidth | 2.8x |
| FP16 compute | about 4.7x |
| **Measured single-card performance** | **5.7x** |
| **Overhead reduction, derived** | **about 5.5x** |

**The overhead did not follow PCIe (16x); it followed compute (5.7x).** It does not
match memory bandwidth (2.8x) either.

⚠️ **No kernel-level profiling was done.** The GEMM hypothesis in 5-4 is consistent
with the numbers, but confirming it would mean profiling a single step and comparing
GEMM time against peer-to-peer transfer time.

**A contrary report.** sd.cpp PR #1640, which proposes `--dit-split`, reports **row
split running 11% faster on 4x GTX 1080 Ti** (161 frames in 168s → 150s, LTX distilled
Q4_K_M) [{ref 5.}](#references). That is the opposite direction on the same Pascal
generation. But **the link configuration is not stated and the model differs** (LTX
video), so it is recorded as a case rather than compared against this work.

---

## References

{ref 1.} *xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive
Parallelism*. https://arxiv.org/html/2411.01738 (accessed 2026-09-11)

{ref 2.} *PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion
Transformer Models*. https://arxiv.org/html/2405.14430v1 (accessed 2026-09-11)

{ref 3.} *DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion
Models*. https://arxiv.org/html/2402.19481v3 (accessed 2026-09-11)

{ref 4.} *stable-diffusion.cpp — Backend and Multi-GPU Configuration* (`docs/backend.md`).
https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/backend.md (accessed 2026-09-11)

{ref 5.} *stable-diffusion.cpp Pull Request #1640 — feat: --dit-split for row-level and
layer-split tensor parallelism across GPUs* (unmerged).
https://github.com/leejet/stable-diffusion.cpp/pull/1640 (accessed 2026-09-11)

{ref 6.} *stable-diffusion.cpp Pull Request #1734 — feat: add multi-device layer split*
(merged 2026-07-04). https://github.com/leejet/stable-diffusion.cpp/pull/1734 (accessed 2026-09-11)

{ref 7.} *stable-diffusion.cpp Pull Request #1735 — feat: support for cross-device row
split* (merged 2026-07-04). https://github.com/leejet/stable-diffusion.cpp/pull/1735
(accessed 2026-09-11)

{ref 8.} *ComfyUI-MultiGPU — extension documentation*.
https://comfy.icu/extension/pollockjj__ComfyUI-MultiGPU (accessed 2026-09-11)

{ref 9.} *diffusers Issue #5197 — Multi-GPU Training Runs Slower Than Single-GPU Training
for SD1.5 text-to-image finetuning*. https://github.com/huggingface/diffusers/issues/5197
(accessed 2026-09-11)

{ref 10.} *Hugging Face Diffusers — xDiT optimization documentation*.
https://huggingface.co/docs/diffusers/optimization/xdit (accessed 2026-09-11)
