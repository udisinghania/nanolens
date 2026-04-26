# NanoLens 🔬
A lightweight PyTorch telemetry tool to track layer-wise variance, detect early representation collapse, and optionally halt silent representation death in continuous world models.

### The Problem
When training continuous latent models (like JEPAs or VLAs), representations often silently collapse into zero-variance states. The cluster keeps running, the loss looks stable, but the model is mathematically dead. You just burned thousands of dollars of H100 compute.

### The Evidence: Layer-wise Pathology
Standard global loss metrics fail to capture the structural reality of representation death. 

![Layer-wise Variance Collapse Propagation](collapse_heatmap.jpg)
*NanoLens capturing variance collapse propagating through a JEPA. When variance regularization is removed (white dotted line), the global loss continues to drop smoothly, but internal telemetry reveals a structural decoupling: the encoder layers collapse to near-zero variance, while the predictor layers simultaneously spike as they thrash against a dead encoder.*

### The Solution
`nanoLens` tracks latent variance (`dim=0`) across the residual stream in real-time. It catches the exact layer where the latent space calcifies, dumps a VRAM-safe telemetry snapshot, and coordinates a safe distributed cluster shutdown before wasting compute budget.

### Installation & Usage

Install directly via pip:
```bash
pip install git+[https://github.com/udisinghania/nanolens.git](https://github.com/udisinghania/nanolens.git)
```
Attach it to your model to begin layer-wise telemetry and DDP-safe anomaly detection:

```python
import torch
from nanolens import attach_nanolens, check_nanolens

# Step 1: Attach to model (choose variance_dim based on your overhead tolerance)
# threshold_box allows for dynamic/warmup calibration mid-training
threshold_box = [0.0] 
handles, should_stop = attach_nanolens(
    model, 
    threshold_box=threshold_box, 
    variance_dim=0 # inter-sample spread — strong signal, higher CPU cost
)

# Step 2: Distributed Training Loop
for step, batch in enumerate(dataloader):
    
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # DDP-safe circuit breaker — call after optimizer.step(), every rank
    check_nanolens(should_stop) 

# Step 3: Cleanup
for h in handles:
    h.remove()
```

### ⚠️ Distributed Training (DDP/FSDP) Architecture
DDP-safe shutdown requires a `should_stop` scalar tensor broadcast via `dist.all_reduce(MAX)` across the process group. This ensures all ranks exit cleanly after the current step completes, rather than raising mid-forward and deadlocking ranks waiting on the next collective. `nanoLens` implements this natively via the `check_nanolens()` API.

### ⚡ Profiling & Hardware Notes
On CPU, `var(dim=0)` on a (batch, seq, dim) tensor allocates a (seq, dim) intermediate per hook call — at GPT-2 scale, causing a **~350% overhead** across 48 hooks with `batch=4`. 

On A100/H100 GPUs at realistic training batch sizes (≥32), matrix multiplications dominate and the hook overhead is amortized to **<3%**. Use `variance_dim=-1` for lower overhead at the cost of a weaker collapse signal (intra-token spread rather than inter-sample spread).

### License
MIT License

Copyright (c) 2026 Udit Singhania

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
