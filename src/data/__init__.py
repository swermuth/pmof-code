"""
Data utilities: reading annotations, dataset management, etc.
"""

from .annotation_utils import read_annotation, xywhr_to_xyxyxyxy
from .file_utils import imgid_to_annpath, imgid_to_imgpath, imgid_to_recid, recid_to_annpath, get_image_size, imgid_to_txtpath
from .dataset_utils import json_image_match, list_record_ids, recordid_to_imageids
from .obbsize_analysis import OBBSizeAnalyzer, MultiDatasetOBBSizeAnalyzer

__all__ = ["read_annotation", "xywhr_to_xyxyxyxy", "imgid_to_annpath", "imgid_to_imgpath", "imgid_to_recid", "recid_to_annpath", "json_image_match", "list_record_ids", "recordid_to_imageids"]
