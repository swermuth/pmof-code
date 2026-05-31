"""
src package initialization
Provides shared utilities like the global logger.
"""

from .utils import logger, DATA_BASE_DIR

#from .data import read_annotation, imgid_to_annpath
#from .visualization import FrameVisualizer, build_legend
#from .visualization import VIZ_PARAMS

__all__ = [
    "logger",
    "DATA_BASE_DIR"
]