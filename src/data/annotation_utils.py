import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from src.utils import logger

@dataclass
class Annotation:
    """Container for for a single annotation."""
    image_id: str
    image_width: int
    image_height: int
    category_id: int
    category_name: str
    bbox: List[float] #tlx, tly, w, h
    bbox_area: float # w*h
    rotation: float
    xyxyxyxy : List[float] #x1, y1, x2, y2, x3, y3, x4, y4
    cxcywhr: List[float] #cx, cy, w, h, rotation
    occluded: bool
    track_id: int
    action: Optional[str] = None

def xywhr_to_area(bbox):
    """
    Returns area of bounding box.
    
    Parameters
    -------
    bbox: list
        bounding box coordinates in COCO format (top left x, top left y, width, height)

    Returns
    float: area of bounding box
    """
    _, _, w, h = bbox #top left x, top left y, width of bbox, height of bbox
    return w * h

def xywhr_to_xyxyxyxy(bbox, rotation):
    """
    Returns x1, y1, x2, y2, x3, y3, x4, y4 coordinates.
    
    Parameters
    -------
    bbox: list
        bounding box coordinates in COCO format (top left x, top left y, width, height)
    rotation: float
        rotation of bounding box in COCO format

    Returns
    List: x1, y1, x2, y2, x3, y3, x4, y4
    """
    tlx, tly, w, h = bbox #top left x, top left y, width of bbox, height of bbox
    cx, cy = tlx + w / 2, tly + h / 2  # Box center

    # Define corners relative to box center
    corners = np.array([
        [-w / 2, -h / 2],
        [ w / 2, -h / 2],
        [ w / 2,  h / 2],
        [-w / 2,  h / 2]
    ])

    # Rotation matrix (convert rotation to radians)
    theta = np.radians(rotation)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate and translate corners
    xyxyxyxy = corners @ rot_mat.T + np.array([cx, cy])
    xyxyxyxy = xyxyxyxy.flatten().tolist()
    
    return xyxyxyxy

def xywhr_to_cxcywhr(bbox, rotation):
    """
    Returns center x, center y, width, height, rotation.
    
    Parameters
    -------
    bbox: list
        bounding box coordinates in COCO format (top left x, top left y, width, height)
    rotation: float
        rotation of bounding box in COCO format

    Returns
    List: center x, center y, width, height, rotation
    """
    tlx, tly, w, h = bbox #top left x, top left y, width of bbox, height of bbox
    cx, cy = tlx + w / 2, tly + h / 2  # Box center

    return [cx, cy, w, h, rotation]

def read_annotation(annotation_path, image_id):
    """
    Returns all annotation for given image_id
    
    Parameters
    -------
    annotation_path: str or Path
        Path to the JSON file.
    
    image_id: str
        image_id, e.g. rec1_00001
    
    Returns
    -------
    List[Annotation]
        A list of Annotation objects.

    Raises
    -------
    FileNotFound
        If the annotation file is not found.
    ValueError
        If the file doesn't list specified image_id.
    """    
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        annotations = None
        raise FileNotFoundError(f"Json file not found: {annotation_path}")

    with open(annotation_path, 'r') as f:
        all_annotations = json.load(f)

    ann_img_id = None
    for image in all_annotations.get("images", []):
        file_name = image.get("file_name", "")
        if Path(file_name).stem == image_id:
            ann_img_id = image['id']
            image_width = image['width']
            image_height = image['height']
            break

    if ann_img_id is None:
        raise ValueError(f"No image entry found for image_id {image_id}.")

    category_id_name_map = {category['id']: category['name'] for category in all_annotations.get("categories", [])}
    
    annotations: List[Annotation] = []
    for annotation in all_annotations.get("annotations", []):
        if annotation.get("image_id") == ann_img_id:
            attributes = annotation.get("attributes", {})
            ann = Annotation(
                image_id=image_id,
                image_width=image_width,
                image_height=image_height,
                category_id=annotation["category_id"],
                category_name = category_id_name_map[annotation["category_id"]],
                bbox=annotation["bbox"],
                bbox_area=xywhr_to_area(annotation["bbox"]),
                rotation=attributes["rotation"],
                xyxyxyxy=xywhr_to_xyxyxyxy(annotation["bbox"], attributes["rotation"]), #convert bbox coordinates for easier plotting later
                cxcywhr=xywhr_to_cxcywhr(annotation["bbox"], attributes["rotation"]), #convert bbox to compare with predictions later
                occluded=attributes["occluded"],
                track_id=attributes["track_id"],
                action=attributes.get("action")  # Optional
            )
            annotations.append(ann)

    return annotations