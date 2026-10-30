# vLLM on RTX 2080 Ti (SM 7.5) via Docker Desktop / WSL2 — verification record

Produced by web research on 2026-10-29 (vLLM source at tag v0.32.0, docs, GitHub issues/PRs,
PyTorch and NVIDIA release notes). Each claim carries a level: **VERIFIED** (read in a primary
source), **LIKELY** (secondary source), **UNKNOWN**. This is a desk check, not a test; the M1
smoke test on the actual card is the confirmation. Summary and consequences are in
[DESIGN.md §2.1](DESIGN.md).

## 1. Latest release / toolchain

- vLLM **v0.32.0, released 18 Oct 2026** (v0.33.0rc0 is a pre-release). VERIFIED — https://github.com/vllm-project/vllm/releases/tag/v0.32.0
- Wheels and `vllm/vllm-openai:v0.32.0` default to **CUDA 13.0**; a `-cu129` tag/wheel is also published. Python 3.11–3.14 (3.10 dropped). VERIFIED — same URL. The docs install page still says "compiled with CUDA 12.9" (stale): https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- `requirements/cuda.txt` at v0.32.0: **torch==2.13.0**, flashinfer-python==0.7.0.post1; Dockerfile: `CUDA_VERSION=13.0.3`, Python 3.12, Ubuntu 24.04. VERIFIED — https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/requirements/cuda.txt , https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docker/Dockerfile

## 2. Is compute capability 7.5 still supported?

- Docs: "GPU: compute capability **7.5 or higher** (e.g., T4, RTX20xx, A100…)". Volta (7.0) is no longer listed. VERIFIED — https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- Dockerfile v0.32.0: `torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` → the published image contains sm_75 kernels. VERIFIED — Dockerfile URL above.
- CMakeLists v0.32.0: `CUDA_SUPPORTED_ARCHS` for CUDA ≥ 13.0 = `7.5;8.0;…;12.0` (7.0 only survives in the pre-12.8 fallback list; comment "marlin has limited support for turing"). VERIFIED — https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/CMakeLists.txt
- PyTorch 2.13.0 cu130/cu129/cu132 x86_64 wheels build `{75, 80, 86, 90, 100, 120}` — sm_75 included, sm_70 dropped. VERIFIED — https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/__init__.py ; CUDA 13.0 removed Maxwell/Pascal/Volta, not Turing. VERIFIED — https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html ; PyTorch 2.15 RFC: "Turing and newer (sm_75+) are unaffected" (opened Jul 2026). VERIFIED — https://github.com/pytorch/pytorch/issues/190385
- No vLLM deprecation notice for Turing found as of v0.32.0. VERIFIED (absence) — release notes above.

## 3. V1 attention backend on CC 7.5

- Selection order (non-MLA): `FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION`. FLASH_ATTN requires capability ≥ 8.0; FLASHINFER has a *temporary* floor of 8.0 ("FlashInfer supports SM75+, but is currently broken on SM75… revert to (7,5) once fix lands"); TRITON_ATTN `supports_compute_capability` returns True. **So Turing auto-selects TRITON_ATTN; no env var needed.** A forced `--attention-backend FLASHINFER` raises ValueError on 7.5. VERIFIED — https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/platforms/cuda.py , `vllm/v1/attention/backends/flash_attn.py`, `flashinfer.py`, `triton_attn.py` at the same tag.
- Known problem: since Triton 3.3, Triton MMA does not use tensor cores on SM < 80 → on a T4, TRITON_ATTN prefill is **6–15× slower** than FlashInfer (2k-token prefill 0.68 s vs 0.12 s), decode 1.2–3× slower; FLEX_ATTENTION is slower still. Issue open (9 Oct 2026); fix PR #55380 (restore FlashInfer on SM75) **not merged** as of mid-Oct 2026. VERIFIED — https://github.com/vllm-project/vllm/issues/60821 , https://github.com/vllm-project/vllm/pull/55380 , warning-only PR https://github.com/vllm-project/vllm/pull/61015
- TRITON_ATTN: head sizes ≥ 32, block size multiple of 16; rejects bf16 KV on < SM80 and FP8 KV on < SM89. Prefix caching and chunked prefill are generic V1 features ("enabled by default whenever possible"); no Turing exclusion found. LIKELY — https://docs.vllm.ai/en/latest/usage/v1_guide/

## 4. Quantization on sm_75

- Docs hardware table: AWQ ✅ Turing, GPTQ ✅, Marlin (GPTQ/AWQ/FP8/FP4) ✅* Turing (*no MXFP4), INT8 W8A8 ✅, FP8 W8A8 ❌. VERIFIED — https://docs.vllm.ai/en/latest/features/quantization/
- Marlin sm75 kernel merged 16 Dec 2025 (PR #29901: GPTQ/AWQ/FP8/NVFP4, fp16 activations; caveats: fp16 accumulator, slower than dequant + cuBLAS at large batch). VERIFIED — https://github.com/vllm-project/vllm/pull/29901
- v0.32.0 code: `AutoAWQConfig.get_min_capability() = 75`; Marlin allowed when `device_capability >= 75` (`marlin_utils.query_marlin_supported_quant_types`); `MarlinLinearKernel.get_min_capability() = 75`, `ExllamaLinearKernel = 60` (fp16 activations only). So AWQ-GEMM checkpoints run via Marlin on the 2080 Ti, with the "unoptimized AWQ kernels" as fallback. VERIFIED — `vllm/model_executor/layers/quantization/auto_awq.py`, `utils/marlin_utils.py`, `kernels/linear/mixed_precision/{marlin,exllama}.py` at v0.32.0.
- 2025–26 Turing breakage reports: a Qwen3-Next AWQ (MoE) failure on T4-class under torch.compile (bf16) in the #29901 thread; nothing for dense Qwen2.5 AWQ/GPTQ. LIKELY.
- **FP8 KV cache: not usable on Turing.** TRITON_ATTN raises "FP8 KV cache is not supported… native FP8 (fp8e4nv) requires SM89+"; FlexAttention lists no fp8; FlashInfer (which did work on SM75 with fp8 KV in v0.23) is gated off. VERIFIED — `triton_attn.py` above; https://github.com/vllm-project/vllm/issues/47549

## 5. CUDA graphs / torch.compile / flags

- No capability gate on CUDA graphs or torch.compile in `cuda.py` (`support_static_graph_mode` → True). VERIFIED. Field reports: a T4 benchmark (Mar 2026) used `--enforce-eager` "required on T4" without evidence of failure; the PR #55380 author ran 2× 2080 Ti end-to-end on v0.28/0.29; the #60821 benchmarks ran default mode on a T4 (v0.31.1). Default should work; keep `--enforce-eager` as the first fallback. LIKELY — https://gist.github.com/Nishant0905/d1c5ff79a4622f8d0db747db3e759e84
- bf16 models: `dtype=auto` logs "Your device … doesn't support bfloat16. Falling back to float16"; an explicit `--dtype bfloat16` is not auto-corrected. VERIFIED — `vllm/config/model.py` at v0.32.0. Qwen2.5 AWQ checkpoints are already fp16, so `--dtype half` is just explicit.
- Defaults: `--gpu-memory-utilization 0.92`, `--block-size` auto (16 is fine), `--max-cudagraph-capture-size ≤ 512`. VERIFIED — https://docs.vllm.ai/en/latest/configuration/engine_args/ . V1 aborts if free VRAM at start < utilization × total (relevant when the 2080 Ti also drives the Windows desktop) → use 0.80–0.85. VERIFIED — https://github.com/vllm-project/vllm/issues/25837

## 6. Docker Desktop + WSL2 GPU

- Requirements: Windows 10/11 up to date, WSL2 backend, latest WSL kernel (`wsl --update`), NVIDIA Windows driver with WSL GPU-PV; run with `--gpus=all`. No Docker Desktop version floor stated. VERIFIED — https://docs.docker.com/desktop/features/gpu/
- Only the Windows driver is installed ("Do not install any Linux display driver in WSL"); the NVIDIA Container Toolkit is needed only for Docker Engine inside WSL, not with Docker Desktop. VERIFIED — https://docs.nvidia.com/cuda/wsl-user-guide/index.html , https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
- CUDA 13 images need an **R580+ host driver** (CUDA 13.x family ≥ 580). Use the `-cu129` tag if the Windows driver is 57x. VERIFIED — https://docs.vllm.ai/en/latest/deployment/docker.html , CUDA 13.0 release notes above. A 2080 Ti report hit "Error 803 unsupported display driver/cuda driver combination" from the image's compat libcuda on an older driver. VERIFIED — https://github.com/vllm-project/vllm/issues/19445 . *This machine has driver 610.88, so the default CUDA 13 image applies.*
- Gotchas: `--ipc=host` or `--shm-size` (docs); `.wslconfig` `memory` default = 50% of host RAM; WDDM caps pinned memory and vLLM disables pin_memory on WSL by default (`VLLM_WSL2_ENABLE_PIN_MEMORY=1` to enable); HF cache volume `/root/.cache/huggingface`. VERIFIED — `cuda.py`; https://learn.microsoft.com/en-us/windows/wsl/wsl-config
- Reports of vLLM 2025–26 on Docker Desktop + WSL2 **with a Turing card: none found → UNKNOWN.** (docker/model-runner #1018 is Model Runner's CPU-only torch bug, not the vllm-openai image.)
- **Measured on this machine (2026-10-29/30):** Docker Desktop 4.2.0 (Nov 2021) passes the GPU through (`nvidia-smi` in a CUDA 13.0.1 container shows the 2080 Ti) but `cuInit` fails inside containers with `Error 500: named symbol not found` for both torch 2.13+cu130 and a raw ctypes call. Cause: Windows drivers 555+ need `libnvdxgdmal.so.1` from `/usr/lib/wsl/drivers/nv_dispi*/`, which NVIDIA container toolkits before 1.14.4 do not mount. VERIFIED — https://github.com/NVIDIA/nvidia-container-toolkit/issues/520 , https://forums.developer.nvidia.com/t/cuda-initialization-fails-on-docker-containers-with-wsl2/294326 . Workaround in this repo: bind-mount `/usr/lib/wsl/lib` and `/usr/lib/wsl/drivers` and prepend them to `LD_LIBRARY_PATH` (`scripts/vllm-entrypoint.sh`); torch then reports the GPU with ~10 GB free.

## 7. Qwen2.5 AWQ models

- Both exist, ungated. 7B: Apache-2.0, 5.57 GB safetensors; 3B: **qwen-research licence**, 2.69 GB. Both AWQ 4-bit, group 128, GEMM, zero_point, fp16; 28 L / 4 KV heads / head 128 (7B) and 36 L / 2 KV heads (3B) → ~56 KiB and ~36 KiB KV per token (fp16). VERIFIED — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ , https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ
- On a T4 (sm_75, 15 GB) the 7B-AWQ served at ~22.6 tok/s single-stream with `--max-model-len 2048 --gpu-memory-utilization 0.85 --enforce-eager`. LIKELY — gist above. No 11 GB report found; estimate: 7B ≈ 5.2 GiB weights + ~1 GiB runtime → ~2.5–3 GiB KV ≈ 45–55k tokens at 0.85 utilization; 3B leaves ~6 GiB KV. UNVERIFIED estimate.

## Recommended smoke-test command (PowerShell, Docker Desktop)

```
docker run --rm --gpus all --ipc=host -p 8000:8000 `
  -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
  vllm/vllm-openai:v0.32.0 `
  --model Qwen/Qwen2.5-3B-Instruct-AWQ --dtype half `
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.82
```

Omit `--runtime nvidia` (Docker Desktop registers `--gpus` itself). If the driver were < 580, use
`:v0.32.0-cu129`. If startup fails in cudagraph capture, add `--enforce-eager`. Do not pass
`--kv-cache-dtype fp8`.

## Verdict

**Viable with specific flags, with a performance caveat.** v0.32.0's image and PyTorch 2.13/cu130
both ship sm_75 kernels, docs list CC 7.5 as the minimum, AWQ/GPTQ run via the new Marlin-sm75 path,
and WSL2/Docker Desktop GPU passthrough needs only a ≥ 580 Windows driver. The costs: TRITON_ATTN is
the only auto-selectable backend and lacks tensor-core MMA on Turing (prefill up to ~15× slower than
FlashInfer; decode 1.2–3× slower), no fp8 KV cache, no bf16. If prefill latency matters, wait for
PR #55380 or pin `v0.23.0` (last image where FlashInfer + fp8 KV worked on SM75 per #47549). Nobody
has publicly confirmed this exact Turing + Docker Desktop combination on v0.32.0, so budget a smoke test.
