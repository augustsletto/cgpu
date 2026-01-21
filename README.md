# cgpu

Quick CUDA/GPU status summary for ML engineers. One import, one call, all the info you need.

## Installation

```bash
pip install cgpu
```

or with uv:

```bash
uv pip install cgpu
```

**Note:** You need PyTorch installed separately. For temperature/utilization info, also install `pynvml`:

```bash
pip install torch pynvml
```

## Usage

```python
from cgpu import cgpu

device = cgpu()
# Now use `device` in your code
model.to(device)
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
