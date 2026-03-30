# NanoLens 🔬
A lightweight, fault-tolerant PyTorch hook to detect variance collapse and prevent silent representation death in continuous physical world models.

### The Problem
When training continuous latent models (like JEPAs or VLAs), representations often silently collapse into zero-variance states. The cluster keeps running, the loss looks stable, but the model is mathematically dead. You just burned thousands of dollars of H100 compute.

### The Solution
`nanoLens` tracks latent variance (`dim=-1`) across the residual stream in real-time. It catches the exact layer where the latent space calcifies, dumps a VRAM-safe telemetry snapshot, and kills the run before wasting compute budget.

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
# Automatically monitors linear layers for variance drops below 1e-4
handles = attach_nanolens(model, threshold=1e-4)

# Run your standard distributed training loop...
```

### Checksums (SHA-256):
* **Primary_Architecture_Draft_v1.0:** 61046D654DD61040C7142CA3D488B1C01A46C035DEB178F76C01B0211D978072
* **Primary_Architecture_Draft_v1.1 (Notation Patch):** 5D7D3BEAACDB62F278E5517CBA6BB286C5A25FBC85373EF93FF0D459F4A1B74B

### License
MIT License

Copyright (c) 2026 Udit Singhania

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
