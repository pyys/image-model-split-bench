# METHOD — 재현 및 관련 주의사항

> ⚠️ **영문 [`METHOD.md`](METHOD.md) 이 정본이다.** 어긋나면 영문이 맞다.

측정 2026-09-11(P104), 2026-09-14 ~ 09-15(V100).

> 여기에는 **측정 과정에서 부딪힌 엔진 제약, 그 원인, 밟은 함정, 그리고 재현 방법**을
> 남긴다. 결과와 결론은 [FINDINGS.ko.md](FINDINGS.ko.md) 에 있다.

---

## 1. 현재 master 에서는 텐서(row) 분할이 동작하지 않는다

`--split-mode row` 를 master HEAD(`b68d586`, 2026-09-11)에서 실행하면 이렇게 된다.

```
[WARN] Diffusion model: row split unavailable (backend has no split buffer type);
       falling back to layer split
```

**경고를 내고 레이어 분할로 내려간다.** 그리고 s/step 이 레이어 분할과 소수점까지
같게 나온다(3.13 / 3.14). **로그를 읽지 않으면 "텐서 병렬을 측정했다"고 착각할
여지가 있다.**

FINDINGS 의 측정이 `68f3d6d`(2026-07-04) 빌드로 수행된 이유가 이것이다.

| 태그 | sd.cpp | ggml | `split_buffer_type` | row 분할 |
|---|---|---|---|---|
| **JUL** | `68f3d6d` (2026-07-04) | `eced84c` | **있음** | **동작** |
| **SEP** | `b68d586` (2026-09-11, master HEAD) | `e20c3a1` | **없음** | **폴백** |

### 1-1. 원인 — 하부 라이브러리에서 기능이 빠졌다

소스로 확인한 사실이다.

| 구성요소 | `split_buffer_type` |
|---|---|
| `ggml/include/ggml-backend.h` (인터페이스) | **있음** |
| `ggml/src/ggml-sycl/` | **있음** |
| **`ggml/src/ggml-cuda/` (SEP, `e20c3a1`)** | **없음** |
| `ggml/src/ggml-cuda/` (JUL, `eced84c`) | **있음** |
| sd.cpp 의 row 경로 | **요청함** |

sd.cpp 는 장치별 배분 비율까지 다 계산해 놓고 마지막에 무너진다.

```c
split_buft = backend_manager.split_buffer_type(main_backend, tensor_split);
if (split_buft == nullptr) return fall_back_to_layer_split("backend has no split buffer type");
```

**sd.cpp 코드에 잘못이 없다.** 공용 인터페이스대로 요청하고, 폴백 사실을 정직하게
경고한다. **`eced84c` → `e20c3a1` 사이에 CUDA 백엔드가 이 인터페이스 구현을 잃었다.**

> **SYCL 백엔드에는 남아 있다.** Intel GPU 에서는 row 분할이 동작할 가능성이 있다.
> 확인하지 않았다.

### 1-2. 대체 경로는 있으나 sd.cpp 가 쓰지 않는다

같은 ggml-cuda 안에 **NCCL 기반 all-reduce 가 있고, 공개 API 로 노출돼 있다.**

```
ggml-cuda.h      ggml_backend_cuda_allreduce_tensor(backends, tensors, n_backends)
ggml-backend.h   typedef ... ggml_backend_comm_allreduce_tensor_t     (백엔드 중립)
```

우리 sd.cpp 빌드에 **NCCL 이 이미 링크돼 있다**(`libnccl.so.2`, `GGML_CUDA_NCCL=ON`).
llama.cpp 는 이 경로로 옮겨 `--split-mode tensor` 를 제공하지만, **sd.cpp 는 아직
옮기지 않았다.**

> `allreduce_tensor` 는 통신 원시 연산 하나다. row 분할을 하려면 그 위에 가중치 분배와
> 부분 matmul 배관이 필요하다. **기반은 있고 배관이 없다** — 난이도는 판단하지 않는다.

**즉 "미구현"이 아니라 "미이전"이다.** row 분할은 2026-07-04 에 **병합됐다가**
하부 변경으로 끊겼다.

### 1-3. 관련 상류 현황 (2026-09-11 기준)

| PR | 상태 | 내용 |
|---|---|---|
| [#1735](https://github.com/leejet/stable-diffusion.cpp/pull/1735) | **병합 2026-07-04** | cross-device row split [{ref 7.}](FINDINGS.ko.md#참고문헌) |
| [#1734](https://github.com/leejet/stable-diffusion.cpp/pull/1734) | **병합 2026-07-04** | multi-device layer split [{ref 6.}](FINDINGS.ko.md#참고문헌). **2 GiB 예비분도 여기서 도입** |
| [#1640](https://github.com/leejet/stable-diffusion.cpp/pull/1640) | 미병합 | `--dit-split` 제안. 1080 Ti 4장 11% 보고 |
| [#1911](https://github.com/leejet/stable-diffusion.cpp/pull/1911) | 열림 | 2 GiB 예비분을 `SD_COMPUTE_HEADROOM_MB` 로 조절 가능하게 |
| [#1186](https://github.com/leejet/stable-diffusion.cpp/pull/1186) | 열림 | flash attention 기본값화 |
| [#1931](https://github.com/leejet/stable-diffusion.cpp/pull/1931) | 열림 | CUDA 도커에서 NCCL 제거 (*"not useful in the general case of local inference"*) [{ref 11.}](#참고문헌) |

---

## 2. 엔진이 카드 수를 정한다 — 사용자가 아니라

Q8(12,125 MB)로 측정하던 중 발견한 것으로, **모델 크기와 무관하게 sd.cpp 멀티 GPU
사용자 전부에게 해당한다.**

### 2-1. 카드당 2 GiB 가 무조건 예비로 빠진다

```c
// src/core/layer_split_partition.cpp:80
// src/pipeline/diffusion_engine.cpp:422
constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
```

```
가용(free)  8,023.19 MB  −  예비 2,048.00 MB  =  사용 가능 5,975.19 MB
```

에러 메시지의 값과 소수점까지 일치한다. **`--max-vram` 으로 이 상한을 올릴 수 없다
— 내리기만 한다.**

### 2-2. 결과 — Q8 에서 카드 수는 조절 불가능했다 (P104 8GB)

| 장수 | 가능? | 이유 |
|---|---|---|
| 2장 | **불가** | 12,125 / 2 = 6,062 > 5,975 |
| **3장** | **유일하게 가능** | 자동 배분이 선택 |
| 4장 | **불가** | 예산으로 강제 시 항상 실패 (9개 구성 시도) |

그리고 3장 배분이 **5,944.2 / 5,857.6 / 323.5 MB** 였다 — 세 번째 카드가 2.7%만
든다. **실질 2장이다.** 4장을 제공해도 결과가 동일했다(CUDA3 에 0 할당).

**카드 4장 중 2장이 놀고 있는데도 사용자가 개입할 수단이 없다.** 실제 여유는 카드당
2,047 MB 씩, 합쳐서 **4GB 가 사용되지 못한 채 남는다.**

> **VRAM 이 크면 이 제약은 드러나지 않는다.** V100 16GB 에서는 카드당 사용 가능이
> 약 14,336 MB 라 같은 Q8 모델이 **한 장에 들어간다.** 제약이 사라진 것이 아니라
> **작은 카드에서만 표면화되는 종류의 제약**이다. 고정 상수를 빼는 방식이기 때문에
> **카드가 작을수록 비중이 커진다** — 8GB 에서는 25%, 16GB 에서는 12.5% 다.

> #1911 이 병합되면 이 제약은 조절 가능해진다. 다만 그 PR 은 `layer_split_partition.cpp`
> 쪽만 다루며 `diffusion_engine.cpp:422` 의 같은 상수는 남을 수 있다.

### 2-3. row 분할에는 이 제약이 적용되지 않는다

row 경로에서 2 GiB 예비분은 **수용 한도가 아니라 배분 비율 계산에만** 쓰인다.

```c
int64_t usable_bytes = max(free - 2GiB, free / 8);
tensor_split[reg_index] = usable_bytes / MB;
```

카드가 동일하면 균등 비율이 되고, **1/2/3/4장 어디에든 고르게 나뉜다.** FINDINGS 의
R2/R3/R4 셀이 제약 없이 성립한 이유다.

| | layer | row |
|---|---|---|
| 예비분의 역할 | **카드별 수용 한도** — 넘으면 실패 | **배분 비율** — 실패 요인 아님 |
| 2장 가능? | 모델 크기에 따라 불가 | 가능 |
| 카드 수 제어 | 사실상 불가 | **장치 목록 그대로** |

---

## 3. Q8 측정에서 얻은 부수 수치

주 실험은 Q4_0 이지만, Q8 로 먼저 측정한 값도 기록해 둔다.
**3장 자동 배분(5,944 / 5,858 / 324 MB), `--diffusion-fa`, `vae=cpu`, 15스텝.**

| 해상도 | 토큰 | 웜 s/step | 512 대비 | CUDA0 버퍼 | CUDA1 버퍼 |
|---|---|---|---|---|---|
| 512 | 4,096 | **3.04** | 1.00 | 164 MB | 약 250 MB |
| 768 | 9,216 | **6.69** | 2.20 | 342 MB | 527 MB |
| 1024 | 16,384 | **13.22** | **4.35** | 567 MB | 896 MB |

**연산 버퍼는 토큰 수에 선형이고, s/step 은 1024 에서 초선형이다**(토큰 4배에 시간
4.35배). flash attention 이 **메모리**의 제곱항은 없애지만 **연산**의 제곱항은
남기기 때문이다.

**flash attention 을 끄면 1024 가 아예 실행되지 않았다** — 버퍼가 2,240 MB 로
불어나 `need 8,194 > available 8,023` 으로 실패한다.

---

## 4. 접근할 수 없었던 선택지들

이미지 생성의 다중 GPU 병렬화에는 더 나은 기법들이 있다. 문제는 그 기법들이
**하드웨어 아키텍처 의존적**이라는 것이다.

### 4-1. P104-100 (sm_61) 에서 막힌 것

| 장벽 | 내용 |
|---|---|
| **PyTorch** | 2.8 / cu128 부터 **Pascal(sm_50·sm_60) 지원 중단.** sm_61 은 **2.7 이 마지막** 공식 휠. CUDA 13 빌드는 sm_75 이상 [{ref 12.}](#참고문헌) [{ref 13.}](#참고문헌) |
| **flash-attn** | **compute capability 8.0(Ampere) 이상 필수.** xDiT 의 주력인 USP 가 여기 얹혀 있다 [{ref 14.}](#참고문헌) [{ref 15.}](#참고문헌) |
| **P104-100** | fp16 이 fp32 의 **1/64**. PyTorch 는 fp16/bf16 전제 |

**독립적인 이유 셋이라 하나를 우회해도 나머지가 남는다.** 2.7 로 내려도 flash-attn 이
막고, 그걸 우회해도 fp16 성능이 막는다.

### 4-2. 두 카드의 요구 사항 대조

| 요구 사항 | P104-100 (sm_61) | V100 (sm_70) |
|---|---|---|
| PyTorch 공식 휠 | **2.7 이 마지막** [{ref 12.}](#참고문헌) [{ref 13.}](#참고문헌) | **sm_70 을 현행 빌드가 지원** |
| flash-attn | compute capability **8.0 이상** 요구 [{ref 14.}](#참고문헌) [{ref 15.}](#참고문헌) | 동일하게 **8.0 이상** 요구 |
| fp16 | fp32 의 **1/64** | **31.4 TFLOPS** (텐서코어 125) |

> ⚠️ **요구 사항을 대조한 것이지 실행해 본 것이 아니다.** 이 문서의 측정은 전부
> sd.cpp 로 했고, PyTorch 생태계로 넘어가는 것은 범위 밖이었다.

**P104 에서 sd.cpp 는 타협이 아니었다.** Pascal + 8GB 에서 사실상 유일하게 돌아가는
경로였고, 그 안에서 쓸 수 있는 모듈 내부 분할은 **naive 레이어 분할 하나**뿐이다
(row 는 끊겼고, 시퀀스/패치 병렬은 미구현). **두 카드를 같은 엔진으로 비교하기 위해
V100 에서도 sd.cpp 를 유지했다.**

---

## 5. 재현

### 5-1. 빌드

```
git clone --recursive https://github.com/leejet/stable-diffusion.cpp /root/sdcpp-row
git -C /root/sdcpp-row checkout 68f3d6d
git -C /root/sdcpp-row submodule update --init --recursive
cmake -S /root/sdcpp-row -B /root/sdcpp-row/build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build /root/sdcpp-row/build --config Release -j 8
```

**아키텍처를 카드에 맞출 것.** P104-100 은 `61`, V100 은 `70` 이다. 두 카드를
번갈아 쓰려면 **`61;70` 으로 함께 넣는다.** FINDINGS 의 V100 측정이 그렇게 빌드됐다.

```
grep -i cuda_architectures /root/sdcpp-row/build/CMakeCache.txt
```

**빌드 전에 split buffer 가 실제로 있는지 확인할 것.** 없으면 row 분할 측정이
성립하지 않는다.

```
grep -rn "split_buffer_type" /root/sdcpp-row/ggml/src/ggml-cuda/ | head
```

### 5-2. 양자화

```
sd-cli -M convert -m <FLUX.1-dev 파생 FP16 safetensors> --type q4_0 -o <출력>.gguf
```

23GB 원본에서 6,482 MB GGUF 가 나온다.

### 5-3. 실행

셀 실행 스크립트와 셀별 결과 표를 저장소에 함께 공개한다.

| 파일 | 내용 |
|---|---|
| `harness/flux_bench.sh` | 7셀(1card / L2·L3·L4 / R2·R3·R4) 순차 실행 |
| `harness/flux_worker_bench.sh` | 워커 1~4벌 |
| `results/` | 셀별 결과 표 |

두 스크립트 모두 셀마다 다음을 1초 간격으로 기록한다.

```
timestamp, index, utilization.gpu, memory.used, temperature.gpu,
power.draw, clocks.current.sm, clocks_throttle_reasons.active
```

### 5-4. 폴백 검출 — 반드시 확인할 것

`--split-mode row` 가 실제로 동작했는지는 **로그 문자열과 카드 사용 패턴 둘로**
교차 확인해야 한다.

| 신호 | row 정상 | 폴백 |
|---|---|---|
| 로그 | `row split: N tensors (X MB) split across N devices` | `row split unavailable` |
| `layer split:` 줄 | **없음** | 있음 |
| 카드 사용률 | 전 장이 동시에 (main 편중) | 한 장만 100%, 순차 |
| s/step | layer 와 **다름** | layer 와 **동일** |

**s/step 이 layer 와 소수점까지 같으면 폴백이다.** 우연이 아니다.
FINDINGS 의 측정은 **두 하드웨어 모두 폴백 0건**이었다.

### 5-5. 스로틀 판정 — 밟았던 함정

**`clocks_throttle_reasons.active` 는 비트마스크이고 `0x1` 은 GpuIdle 이다.
스로틀링이 아니다.** 노는 카드가 전부 이 비트를 켜므로 "0이 아니면 스로틀"로 세면
크게 과대 보고된다.

실제로 클럭을 누르는 비트는 이것들이다.

```
therm = 0x08 HwSlowdown | 0x20 SwThermal | 0x40 HwThermal
pcap  = 0x04 SwPowerCap | 0x80 HwPowerBrake
```

**열과 전력을 분리해 집계해야 원인이 구분된다.**

> **이 함정은 측정 중 실제로 밟았다.** 처음 집계에서 유휴를 스로틀로 세어
> "1card 399, L4 177" 같은 값이 나왔고, **카드를 더 쓸수록 스로틀이 줄어드는 이상한
> 패턴**을 보고서야 오류를 알아챘다. 레이어 분할은 한 번에 한 장만 돌므로 **카드가
> 많을수록 유휴 샘플이 많아진다.** 그것을 스로틀로 세고 있었다.

집계가 옳은지는 원시 CSV 의 비트마스크 분포를 직접 확인하면 된다.

```
awk -F, '{gsub(/ /,"",$8); c[$8]++} END {for (k in c) print k, c[k]}' gpu_L4.csv
```

### 5-6. 워밍업 — 셀당 1장만 뽑으면 섞인다

**첫 장에는 약 36% 의 워밍업 오버헤드가 붙는다.** 워커 벤치에서 장당 시간이
`9.73 / 7.10 / 7.13 / 7.13` 으로 나온다 — 첫 장만 다르다.

7셀 벤치는 셀당 1장만 뽑으므로 **샘플링 총시간에 이 오버헤드가 그대로 들어간다.**
그래서 FINDINGS 는 **진행 로그의 마지막 스텝 속도(웜 s/step)를 주 지표로 쓴다.**

```
grep -aoE "15/15 - [0-9.]+ ?(it/s|s/it)" r_1card.log | tail -1
```

> **`it/s` 앞에 공백이 없다.** `2.13it/s` 형태로 출력되므로 공백을 요구하는 정규식은
> 아무것도 잡지 못한다. 이것도 측정 중 밟았다.

샘플링 총시간을 쓰고 싶다면 **`--batch-count 2` 이상으로 돌리고 마지막 장을 취해야
한다.**

---

## 참고문헌

FINDINGS 와 공유한다. 이 문서에서 추가로 인용한 것만 적는다.

{ref 11.} *stable-diffusion.cpp Pull Request #1931 — chore: Create cuda runtime docker
from scratch and drop NCCL* (열림).
https://github.com/leejet/stable-diffusion.cpp/pull/1931 (접근 2026-09-11)

{ref 12.} *PyTorch Versions and Supported NVIDIA GPU Compute Capability Levels*
(Qualiteg Journal, 2026-06).
https://journal.qualiteg.com/pytorch_and_supported_gpu_version/ (접근 2026-09-11)

{ref 13.} *pytorch Issue #53164 — why sm_61 was dropped from the nightly build?*
https://github.com/pytorch/pytorch/issues/53164 (접근 2026-09-11)

{ref 14.} *flash-attention — requirements (compute capability >= 8.0)*.
https://github.com/dao-ailab/flash-attention (접근 2026-09-11)

{ref 15.} *transformers Issue #28188 — RuntimeError: FlashAttention only supports Ampere
GPUs or newer*. https://github.com/huggingface/transformers/issues/28188 (접근 2026-09-11)

---
