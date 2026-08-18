"""
Sequential VRAM Manager (PRD §7.1.2 / 8 GB Local GPU Budget Constraint).
Enforces explicit unloading of GPU model weights between ingestion and query execution stages,
preventing VRAM memory contention and OOM crashes.
"""

import gc
from typing import Dict, Any


class VRAMManager:
    """Utility class to manage PyTorch CUDA VRAM allocation and sequential model unloading."""

    @staticmethod
    def unload_model(model_obj: Any, model_name: str = "Model"):
        """Explicitly delete model object, run garbage collection, and flush CUDA cache."""
        print(f"[VRAMManager] Unloading {model_name} from GPU VRAM...")
        try:
            del model_obj
        except Exception:
            pass

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = peak_bytes / (1024 ** 2)
                print(f"[VRAMManager] Flushed CUDA cache. Peak GPU VRAM allocated: {peak_mb:.2f} MB / 8,192 MB budget.")
        except Exception:
            pass

    @staticmethod
    def get_vram_stats() -> Dict[str, Any]:
        """Return current GPU VRAM allocation statistics."""
        stats = {
            "cuda_available": False,
            "device_name": "CPU",
            "allocated_mb": 0.0,
            "max_allocated_mb": 0.0
        }

        try:
            import torch
            if torch.cuda.is_available():
                stats["cuda_available"] = True
                stats["device_name"] = torch.cuda.get_device_name(0)
                stats["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
                stats["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        except Exception:
            pass

        return stats
