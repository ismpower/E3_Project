# davp_utils.py
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

def compute_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Compute hash of file for integrity tracking"""
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def generate_processing_id() -> str:
    """Generate unique processing ID"""
    return str(uuid.uuid4())

def create_timestamp() -> str:
    """Create ISO format timestamp"""
    return datetime.utcnow().isoformat() + 'Z'

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def safe_float_convert(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with default"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_convert(value: str, default: int = 0) -> int:
    """Safely convert string to int with default"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
