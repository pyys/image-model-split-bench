# METHOD — reproduction and caveats

Measured 2026-09-11 (P104) and 2026-09-14 to 09-15 (V100).

> This document holds **the engine constraints hit during measurement, their causes,
> the traps we stepped in, and how to reproduce the work.** Results and conclusions
> are in [`FINDINGS.md`](FINDINGS.md).

---

## 1. Tensor (row) split does not work on current master

Running `--split-mode row` on master HEAD (`b68d586`, 2026-09-11) gives this.

```
[WARN] Diffusion model: row split unavailable (backend has no split buffer type);
       falling back to layer split
```

**It warns and drops to layer split.** And s/step then matches layer split to the
decimal (3.13 / 3.14). **Without reading the log you could believe you had measured
tensor parallelism.**

That is why FINDINGS was measured on the `68f3d6d` (2026-07-04) build.

| Tag | sd.cpp | ggml | `split_buffer_type` | Row split |
|---|---|---|---|---|
| **JUL** | `68f3d6d` (2026-07-04) | `eced84c` | **present** | **works** |
| **SEP** | `b68d586` (2026-09-11, master HEAD) | `e20c3a1` | **absent** | **falls back** |

### 1-1. The cause — a capability disappeared from the underlying library

Confirmed against the source.

| Component | `split_buffer_type` |
|---|---|
| `ggml/include/ggml-backend.h` (interface) | **present** |
| `ggml/src/ggml-sycl/` | **present** |
| **`ggml/src/ggml-cuda/` (SEP, `e20c3a1`)** | **absent** |
| `ggml/src/ggml-cuda/` (JUL, `eced84c`) | **present** |
| sd.cpp's row path | **requests it** |

sd.cpp computes the per-device split ratios all the way through and then collapses at
the last step.

```c
split_buft = backend_manager.split_buffer_type(main_backend, tensor_split);
if (split_buft == nullptr) return fall_back_to_layer_split("backend has no split buffer type");
```

**Nothing is wrong with sd.cpp's code.** It requests through the public interface and
honestly warns that it fell back. **The CUDA backend lost its implementation of that
interface somewhere between `eced84c` and `e20c3a1`.**

> **The SYCL backend still has it.** Row split may work on Intel GPUs. Not verified.

### 1-2. An alternative path exists, but sd.cpp does not use it

The same ggml-cuda contains **an NCCL-based all-reduce, exposed as public API.**

```
ggml-cuda.h      ggml_backend_cuda_allreduce_tensor(backends, tensors, n_backends)
ggml-backend.h   typedef ... ggml_backend_comm_allreduce_tensor_t     (backend neutral)
```

**NCCL is already linked into our sd.cpp build** (`libnccl.so.2`,
`GGML_CUDA_NCCL=ON`). llama.cpp moved to this path to provide `--split-mode tensor`,
but **sd.cpp has not.**

> `allreduce_tensor` is one communication primitive. Row split needs weight
> distribution and partial-matmul plumbing on top of it. **The foundation is there and
> the plumbing is not** — we make no judgement about the difficulty.

**So this is "not ported", not "not implemented".** Row split was **merged** on
2026-07-04 and then broken by a change underneath it.

### 1-3. Upstream status (as of 2026-09-11)

| PR | State | Content |
|---|---|---|
| [#1735](https://github.com/leejet/stable-diffusion.cpp/pull/1735) | **merged 2026-07-04** | cross-device row split [{ref 7.}](FINDINGS.md#references) |
| [#1734](https://github.com/leejet/stable-diffusion.cpp/pull/1734) | **merged 2026-07-04** | multi-device layer split [{ref 6.}](FINDINGS.md#references). **The 2 GiB reserve came in here too** |
| [#1640](https://github.com/leejet/stable-diffusion.cpp/pull/1640) | unmerged | proposes `--dit-split`. Reports 11% on 4x 1080 Ti |
| [#1911](https://github.com/leejet/stable-diffusion.cpp/pull/1911) | open | makes the 2 GiB reserve tunable via `SD_COMPUTE_HEADROOM_MB` |
| [#1186](https://github.com/leejet/stable-diffusion.cpp/pull/1186) | open | flash attention on by default |
| [#1931](https://github.com/leejet/stable-diffusion.cpp/pull/1931) | open | drop NCCL from the CUDA docker image (*"not useful in the general case of local inference"*) [{ref 11.}](#references) |

---

## 2. The engine decides the card count, not you

Found while measuring with Q8 (12,125 MB), but **it applies to every sd.cpp multi-GPU
user regardless of model size.**

### 2-1. 2 GiB per card is reserved unconditionally

```c
// src/core/layer_split_partition.cpp:80
// src/pipeline/diffusion_engine.cpp:422
constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
```

```
free 8,023.19 MB  -  reserve 2,048.00 MB  =  usable 5,975.19 MB
```

This matches the error message to the decimal. **`--max-vram` cannot raise that
ceiling — only lower it.**

### 2-2. The consequence — with Q8 the card count was not adjustable (P104 8GB)

| Cards | Possible? | Why |
|---|---|---|
| 2 | **no** | 12,125 / 2 = 6,062 > 5,975 |
| **3** | **the only option** | chosen by automatic distribution |
| 4 | **no** | forcing it by budget always failed (nine configurations tried) |

And the three-card distribution came out as **5,944.2 / 5,857.6 / 323.5 MB** — the
third card carries 2.7%. **It is effectively two cards.** Supplying four made no
difference (CUDA3 got zero).

**Two of four cards sit idle and the user has no way to intervene.** The real headroom
is 2,047 MB per card, **4GB left unused in total.**

> **On a large card the constraint never surfaces.** A V100 16GB has about 14,336 MB
> usable per card, so the same Q8 model **fits on one.** The constraint has not gone
> away — it is **the kind that only shows up on small cards.** Because it subtracts a
> fixed constant, **the smaller the card the larger the share** — 25% at 8GB, 12.5% at
> 16GB.

> If #1911 is merged the constraint becomes tunable. Note that the PR only touches
> `layer_split_partition.cpp`; the same constant at `diffusion_engine.cpp:422` may
> remain.

### 2-3. Row split is not subject to this

On the row path the 2 GiB reserve is used **for the distribution ratio, not as an
acceptance limit.**

```c
int64_t usable_bytes = max(free - 2GiB, free / 8);
tensor_split[reg_index] = usable_bytes / MB;
```

With identical cards the ratio comes out even, and **it divides cleanly across one,
two, three or four cards.** That is why the R2 / R3 / R4 cells in FINDINGS worked
without constraint.

| | layer | row |
|---|---|---|
| Role of the reserve | **per-card acceptance limit** — exceed it and it fails | **distribution ratio** — not a failure mode |
| Two cards possible? | depends on model size | yes |
| Control over card count | effectively none | **exactly the device list** |

---

## 3. Side figures from the Q8 measurement

The main experiment is Q4_0, but the earlier Q8 measurements are recorded here.
**Three-card automatic distribution (5,944 / 5,858 / 324 MB), `--diffusion-fa`,
`vae=cpu`, 15 steps.**

| Resolution | Tokens | Warm s/step | vs 512 | CUDA0 buffer | CUDA1 buffer |
|---|---|---|---|---|---|
| 512 | 4,096 | **3.04** | 1.00 | 164 MB | about 250 MB |
| 768 | 9,216 | **6.69** | 2.20 | 342 MB | 527 MB |
| 1024 | 16,384 | **13.22** | **4.35** | 567 MB | 896 MB |

**The compute buffer is linear in token count, while s/step is superlinear by 1024**
(4x the tokens, 4.35x the time). Flash attention removes the quadratic term in
**memory** but leaves it in **compute**.

**With flash attention off, 1024 would not run at all** — the buffer grows to 2,240 MB
and it fails with `need 8,194 > available 8,023`.

---

## 4. Options that were out of reach

There are better ways to parallelize image generation across GPUs. The problem is that
they are **architecture dependent.**

### 4-1. What blocked them on the P104-100 (sm_61)

| Barrier | Detail |
|---|---|
| **PyTorch** | From 2.8 / cu128, **Pascal (sm_50, sm_60) support was dropped.** For sm_61, **2.7 is the last** official wheel. CUDA 13 builds target sm_75 and above [{ref 12.}](#references) [{ref 13.}](#references) |
| **flash-attn** | **Requires compute capability 8.0 (Ampere) or newer.** xDiT's primary method, USP, sits on top of it [{ref 14.}](#references) [{ref 15.}](#references) |
| **P104-100** | fp16 runs at **1/64** of fp32. PyTorch assumes fp16/bf16 |

**Three independent reasons, so routing around one leaves the others.** Dropping to
2.7 still leaves flash-attn, and working around that still leaves fp16 performance.

### 4-2. Requirements across the two cards

| Requirement | P104-100 (sm_61) | V100 (sm_70) |
|---|---|---|
| Official PyTorch wheel | **2.7 is the last** [{ref 12.}](#references) [{ref 13.}](#references) | **sm_70 is supported by current builds** |
| flash-attn | requires compute capability **8.0 or newer** [{ref 14.}](#references) [{ref 15.}](#references) | likewise requires **8.0 or newer** |
| fp16 | **1/64** of fp32 | **31.4 TFLOPS** (tensor cores 125) |

> ⚠️ **This compares requirements; nothing was run.** Every measurement in this work
> used sd.cpp, and moving to the PyTorch ecosystem was out of scope.

**On the P104, sd.cpp was not a compromise.** It was effectively the only path that ran
on Pascal with 8GB, and within it the only available intra-module split is **naive
layer split** (row is broken, and sequence and patch parallelism are not implemented).
**sd.cpp was kept on the V100 as well so that both cards could be compared through the
same engine.**

---

## 5. Reproduction

### 5-1. Build

```
git clone --recursive https://github.com/leejet/stable-diffusion.cpp /root/sdcpp-row
git -C /root/sdcpp-row checkout 68f3d6d
git -C /root/sdcpp-row submodule update --init --recursive
cmake -S /root/sdcpp-row -B /root/sdcpp-row/build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build /root/sdcpp-row/build --config Release -j 8
```

**Match the architecture to the card.** The P104-100 is `61` and the V100 is `70`. To
alternate between cards, **list both as `61;70`.** The V100 measurements in FINDINGS
were built that way.

```
grep -i cuda_architectures /root/sdcpp-row/build/CMakeCache.txt
```

**Check that the split buffer is actually present before building.** Without it the
row split measurement does not exist.

```
grep -rn "split_buffer_type" /root/sdcpp-row/ggml/src/ggml-cuda/ | head
```

### 5-2. Quantization

```
sd-cli -M convert -m <FLUX.1-dev derived FP16 safetensors> --type q4_0 -o <output>.gguf
```

A 23GB original produces a 6,482 MB GGUF.

### 5-3. Running

The cell scripts, the full logs and the per-card utilization CSVs are published
alongside.

| File | Contents |
|---|---|
| `harness/flux_bench.sh` | the seven cells (1card / L2·L3·L4 / R2·R3·R4) in sequence |
| `harness/flux_worker_bench.sh` | one to four workers |
| `results/` | the per-cell result tables |

Both scripts record the following per cell at one-second intervals.

```
timestamp, index, utilization.gpu, memory.used, temperature.gpu,
power.draw, clocks.current.sm, clocks_throttle_reasons.active
```

### 5-4. Detecting the fallback — always check

Whether `--split-mode row` actually took effect has to be **cross-checked two ways, by
log string and by card usage pattern.**

| Signal | Row working | Fallback |
|---|---|---|
| Log | `row split: N tensors (X MB) split across N devices` | `row split unavailable` |
| A `layer split:` line | **absent** | present |
| Card utilization | all cards at once (main-heavy) | one card at 100%, sequential |
| s/step | **different** from layer | **identical** to layer |

**If s/step matches layer split to the decimal, it is a fallback.** That is not
coincidence. The FINDINGS measurements had **zero fallbacks on both platforms.**

### 5-5. Judging throttling — a trap we stepped in

**`clocks_throttle_reasons.active` is a bitmask and `0x1` is GpuIdle. That is not
throttling.** Every idle card sets it, so counting "non-zero means throttled"
over-reports badly.

The bits that actually hold the clock down are these.

```
therm = 0x08 HwSlowdown | 0x20 SwThermal | 0x40 HwThermal
pcap  = 0x04 SwPowerCap | 0x80 HwPowerBrake
```

**Aggregate thermal and power separately or the cause is indistinguishable.**

> **We actually stepped in this.** The first aggregation counted idle as throttling and
> produced values like "1card 399, L4 177", and only **the nonsensical pattern of
> throttling falling as cards were added** revealed the error. Layer split runs one
> card at a time, so **more cards means more idle samples.** Those were being counted
> as throttling.

To check an aggregation, read the bitmask distribution straight from the raw CSV.

```
awk -F, '{gsub(/ /,"",$8); c[$8]++} END {for (k in c) print k, c[k]}' gpu_L4.csv
```

### 5-6. Warm-up — one image per cell mixes it in

**The first image carries roughly 36% of warm-up overhead.** In the worker run, time
per image comes out as `9.73 / 7.10 / 7.13 / 7.13` — only the first differs.

The seven-cell run produces one image per cell, so **that overhead lands directly in
the total sampling time.** This is why FINDINGS uses **the last step rate from the
progress log (warm s/step) as its primary metric.**

```
grep -aoE "15/15 - [0-9.]+ ?(it/s|s/it)" r_1card.log | tail -1
```

> **There is no space before `it/s`.** It prints as `2.13it/s`, so a regex that requires
> a space matches nothing. We stepped in this one too.

If you want to use total sampling time, **run with `--batch-count 2` or more and take
the last image.**

---

## References

Shared with FINDINGS. Only the additional citations from this document are listed.

{ref 11.} *stable-diffusion.cpp Pull Request #1931 — chore: Create cuda runtime docker
from scratch and drop NCCL* (open).
https://github.com/leejet/stable-diffusion.cpp/pull/1931 (accessed 2026-09-11)

{ref 12.} *PyTorch Versions and Supported NVIDIA GPU Compute Capability Levels*
(Qualiteg Journal, 2026-06).
https://journal.qualiteg.com/pytorch_and_supported_gpu_version/ (accessed 2026-09-11)

{ref 13.} *pytorch Issue #53164 — why sm_61 was dropped from the nightly build?*
https://github.com/pytorch/pytorch/issues/53164 (accessed 2026-09-11)

{ref 14.} *flash-attention — requirements (compute capability >= 8.0)*.
https://github.com/dao-ailab/flash-attention (accessed 2026-09-11)

{ref 15.} *transformers Issue #28188 — RuntimeError: FlashAttention only supports Ampere
GPUs or newer*. https://github.com/huggingface/transformers/issues/28188 (accessed 2026-09-11)
