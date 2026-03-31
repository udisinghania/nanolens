# NanoLens 🔬
A lightweight, memory-safe PyTorch circuit breaker to detect inter-sample variance collapse and halt silent representation death in continuous world models.

### The Problem
When training continuous latent models (like JEPAs or VLAs), representations often silently collapse into zero-variance states. The cluster keeps running, the loss looks stable, but the model is mathematically dead. You just burned thousands of dollars of H100 compute.

### The Solution
`nanoLens` tracks latent variance (`dim=0`) across the residual stream in real-time. It catches the exact layer where the latent space calcifies, dumps a VRAM-safe telemetry snapshot, and kills the run before wasting compute budget.

### Installation & Usage

Install directly via pip:
```bash
pip install git+[https://github.com/udisinghania/nanolens.git](https://github.com/udisinghania/nanolens.git)
```
Attach it to your model before the training loop:
```python
import torch
from nanolens import attach_nanolens

# Attach to your Vision Transformer or VLA

# Strong collapse detection (default) — inter-sample spread, dim=0
handles = attach_nanolens(model, threshold=1e-4)

# Lower overhead — intra-token spread, dim=-1
handles = attach_nanolens(model, threshold=1e-4, variance_dim=-1)

# Run your standard distributed training loop...
```

### ⚠️ Distributed Training (DDP/FSDP) Architecture
DDP-safe shutdown requires a `should_stop` scalar tensor broadcast via `dist.all_reduce(MAX)` across the process group. This ensures all ranks exit cleanly after the current step completes, rather than raising mid-forward and deadlocking ranks waiting on the next collective. The DDP-safe API (`check_nanolens()`) is under active development.

### ⚡ Profiling & Hardware Notes
On CPU, `var(dim=0)` on a (batch, seq, dim) tensor allocates a (seq, dim) intermediate per hook call — at GPT-2 scale, causing a **~350% overhead** across 48 hooks with `batch=4`. 

On A100/H100 at realistic training batch sizes (≥32), matrix multiplications dominate and the hook overhead is amortized to **<3%**. Use `variance_dim=-1` for lower overhead at the cost of a weaker collapse signal (intra-token spread rather than inter-sample spread).

### Checksums (SHA-256):
* **Primary_Architecture_Draft_v1.0:** 61046D654DD61040C7142CA3D488B1C01A46C035DEB178F76C01B0211D978072
* **Primary_Architecture_Draft_v1.1 (Notation Patch):** 5D7D3BEAACDB62F278E5517CBA6BB286C5A25FBC85373EF93FF0D459F4A1B74B

### License
MIT License

Copyright (c) 2026 Udit Singhania

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
