from datetime import datetime
from pathlib import Path

def timestamp() -> str: 
    """ Current time as 'DDMMYY-HHMMSS' , eg 190426-140352"""
    return datetime.now().strftime("%d%m%y_%H%M%S")

def stamped_path(path: str | Path) -> Path:
    """
    Insert a timestamp before the file extension.
    'figures/dynamics.png' → 'figures/dynamics_190426_143052.png'
    """
    p = Path(path)
    return p.with_name(f"{p.stem}_{timestamp()}{p.suffix}")
