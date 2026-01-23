"""ctqdm - tqdm progress bar with real-time GPU metrics."""

import time
from tqdm import tqdm


def _get_gpu_stats(device_index=0):
    """Get GPU stats. Returns dict with available metrics."""
    stats = {}

    try:
        import torch
        if not torch.cuda.is_available():
            return stats

        mem_allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        stats["vram"] = f"{mem_allocated:.1f}/{mem_total:.1f}GB"
    except (ImportError, RuntimeError):
        return stats

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["temp"] = f"{temp}\u00b0C"
        stats["util"] = f"GPU:{util.gpu}%"
    except (ImportError, Exception):
        pass

    return stats


def _format_stats(stats):
    """Format GPU stats into a compact string."""
    parts = []
    if "temp" in stats:
        parts.append(stats["temp"])
    if "vram" in stats:
        parts.append(stats["vram"])
    if "util" in stats:
        parts.append(stats["util"])
    return " | ".join(parts)


class ctqdm(tqdm):
    """tqdm progress bar with real-time GPU metrics.

    Usage:
        from cgpu import ctqdm

        for batch in ctqdm(dataloader):
            ...
    """

    def __init__(self, *args, gpu_update_interval=0.5, gpu_device=0, **kwargs):
        """
        Args:
            gpu_update_interval: Seconds between GPU stat refreshes (default 0.5).
            gpu_device: CUDA device index to monitor (default 0).
            *args, **kwargs: Passed to tqdm.
        """
        self._gpu_update_interval = gpu_update_interval
        self._gpu_device = gpu_device
        self._last_gpu_update = 0
        self._gpu_stats_str = ""
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        now = time.time()
        if now - self._last_gpu_update >= self._gpu_update_interval:
            stats = _get_gpu_stats(self._gpu_device)
            self._gpu_stats_str = _format_stats(stats)
            self._last_gpu_update = now

        if self._gpu_stats_str:
            self.set_postfix_str(self._gpu_stats_str, refresh=False)

        super().display(msg, pos)
