import torch
import os
import logging

logger = logging.getLogger(__name__)

class Optimizer:
    def __init__(self):
        self.device = self._get_device()
        self.dtype = self._get_dtype()
        self.use_compile = self._check_compile_support()
        
    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            logger.info(f"Using GPU: {props.name} with {props.total_memory / 1024**3:.2f} GB VRAM")
            return device
        else:
            logger.warning("No GPU found, falling back to CPU. Performance will be slow.")
            return torch.device("cpu")

    def _get_dtype(self):
        if self.device.type == "cuda":
            # Check for Ampere or newer for BF16, otherwise FP16
            props = torch.cuda.get_device_properties(self.device)
            if props.major >= 8:
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _check_compile_support(self):
        # Torch compile is supported on Linux/WSL, might be flaky on Windows native
        # We'll enable it if available and not explicitly disabled
        if os.name == 'nt':
            return False # Often problematic on Windows
        return hasattr(torch, 'compile')

    def optimize_model(self, model):
        model = model.to(self.device)
        
        # Apply compilation if supported
        if self.use_compile:
            try:
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
        
        return model

    def get_autocast_context(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

# Global optimizer instance
optimizer = Optimizer()
