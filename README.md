# cgpu

Quick CUDA/GPU status summary for ML engineers. One import, one call, all the info you need.

## Installation

**Full ML stack** (recommended for new projects):
```bash
pip install cgpu-info[full]
```

This installs: torch, torchvision, torchaudio, numpy, pandas, matplotlib, seaborn, scikit-learn

**Other options:**
```bash
# Just PyTorch stack
pip install cgpu-info[torch]

# Just data science packages (no torch)
pip install cgpu-info[science]

# Minimal - just cgpu (if you already have torch)
pip install cgpu-info
```

Works with uv too:
```bash
uv pip install cgpu-info[full]
```

### Installing PyTorch with specific CUDA version

For specific CUDA versions, use the built-in install helper:
```bash
# Install torch with CUDA 12.1
cgpu install --cuda 12.1

# Install torch with CUDA 12.4
cgpu install --cuda 12.4

# Install torch with CUDA 11.8
cgpu install --cuda 11.8

# Install CPU-only torch
cgpu install --cuda cpu
``` 

## Usage

### Python
```python
from cgpu import cgpu

device = cgpu()
# Now use `device` in your code
model.to(device)
```

### CLI
```bash
# Show GPU status
cgpu

# Show version
cgpu --version
```

That's it! You'll see a colorful summary like:

```
═══════════════════════════════════════
          GPU Status Summary
═══════════════════════════════════════
✓ CUDA Available
  Device: cuda
  GPU Count: 1
  [0] NVIDIA GeForce RTX 4090
      VRAM: 24.0 GB
      Allocated: 0.00 GB
      Reserved: 0.00 GB
      Temp: 42°C
      GPU Util: 0%
      Mem Util: 0%
  CUDA Version: 12.1
  cuDNN Version: 8902
  PyTorch: 2.1.0
═══════════════════════════════════════
```

## What it shows

- CUDA availability status
- Device string (`cuda` or `cpu`)
- GPU name and count
- VRAM total and usage
- Temperature (color-coded: green < 50°C, yellow < 70°C, red >= 70°C)
- GPU/Memory utilization
- CUDA, cuDNN, and PyTorch versions

## License

MIT
