import logging
import sys
import torch

def setup_logger(name: str = "ecash_llm") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_device_info():
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    else:
        return "CPU"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
