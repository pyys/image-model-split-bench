# image-model-split-bench

**What it costs to split an image generation model across GPUs — measured.**

When an image generation model does not fit in VRAM, the usual answer is a lower
quantization or a smaller model rather than more GPUs. That is because multi-GPU is
assumed to be inefficient for image generation. **This repository tests that
assumption by measurement.**

The largest module in an image model is the DiT. We loaded it across several GPUs
using **layer split and tensor (row) split**, and compared that side by side against
**running the same cards as independent workers.** A FLUX.1-dev derived DiT quantized
to Q4_0, fixed at 512x512 and 15 steps. The platforms are P104-100 x4 (PCIe Gen1 x4)
and V100 16GB x4 (PCIe Gen3 x16).

**Splitting the DiT across GPUs made it slower.** Four cards deliver **84~87%** of a
single card under layer split and **15~16%** under tensor split; the penalty is far
larger for tensor split. Running the same four cards as independent workers instead
gives **344~388%**. The two platforms differ by **16x** in interconnect bandwidth and
the tensor-split penalty did not shrink.

**So the penalty does not come from interconnect bandwidth, and the conventional
wisdom applies only to splitting a module.** If you want multi-GPU to raise image
throughput, design around **replicating the model rather than splitting it.**

English is authoritative; Korean lives alongside each file as `*.ko.md`.
**이 문서의 한국어판: [README.ko.md](README.ko.md)**
Code and raw data: MIT. Documents: CC BY 4.0.

| | |
|---|---|
| [FINDINGS.md](FINDINGS.md) | What we found, and what we did not |
| [METHOD.md](METHOD.md) | Reproduction and caveats — engine limits, traps, procedure |
| [harness/](harness/) | The measurement scripts |
| [results/](results/) | Per-cell result tables |

**Start with [FINDINGS.md](FINDINGS.md).** It carries the conclusions, the numbers
behind them and the limits on each one. METHOD covers the constraints those numbers
were produced under and how to reproduce them.

> A sister repository measures **LLMs** with the same approach —
> [`layer-tensor-parallel-bench`](https://github.com/pyys/layer-tensor-parallel-bench).
> **It reaches the opposite conclusion.** There, tensor parallelism gave the fastest
> token generation.
