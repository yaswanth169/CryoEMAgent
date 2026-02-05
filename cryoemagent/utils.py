"""Utility functions for CryoEMAgent."""

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Configure logging for the agent."""
    logger = logging.getLogger("cryoemagent")
    logger.setLevel(getattr(logging, level.upper()))
    
    if use_rich:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    return logger


def validate_movie_path(path: str) -> tuple[bool, str]:
    """Validate movie file path or glob pattern."""
    from glob import glob
    
    if "*" in path:
        files = glob(path)
        if not files:
            return False, f"No files found matching pattern: {path}"
        return True, f"Found {len(files)} files"
    
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in [".mrc", ".mrcs", ".tif", ".tiff", ".eer"]:
            return True, "Valid movie file"
        return False, f"Unsupported file format: {p.suffix}"
    
    if p.is_dir():
        return False, "Path is a directory, please provide file path or glob pattern"
    
    return False, f"Path does not exist: {path}"


def format_resolution(resolution: Optional[float]) -> str:
    """Format resolution value for display."""
    if resolution is None:
        return "N/A"
    return f"{resolution:.2f} Å"


def format_particle_count(count: int) -> str:
    """Format particle count with thousands separator."""
    return f"{count:,}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
