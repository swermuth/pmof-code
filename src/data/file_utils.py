# File-level path management utils
import os
import cv2
import re
from pathlib import Path
from src.utils import logger, DATA_BASE_DIR

def imgid_to_recid(image_id):
    """
    Returns record_id for given image_id
    
    Parameters
    -------
    image_id: str
        image_id, e.g. rec1_00001
    
    Returns
    -------
    Str
        record_id, e.g. rec1 (None if invalid image_id format)

    Raises
    -------
    ValueError
        If image_id has invalid format.
    """
    if '_' not in image_id:
        record_id = None
        raise ValueError(f'Invalid image_id: {image_id}. Expecting underscore.')
    else:
        record_id = image_id.split('_')[0]
        digits = image_id.split('_')[1] 
        if not re.fullmatch(r"\d{6}", digits):
            record_id = None
            raise ValueError(f'Invalid image_id: {image_id}. Expecting 6 digits.')            
    return record_id

def imgid_to_imgpath(image_id, data_base_dir=DATA_BASE_DIR):
    """
    Returns path of image for given image_id
    
    Parameters
    -------    
    image_id: str
        image_id, e.g. rec1_00001

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
        
    Returns
    -------
    Str
        Path to image with this id.

    Raises
    -------
    FileNotFound
        If data_base_dir or image_path are not found.
    """    
    data_base_dir = Path(data_base_dir)
    if not os.path.isdir(data_base_dir):
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")
    try:
        record_id = imgid_to_recid(image_id)
        image_dir = os.path.join(data_base_dir, "images" , record_id)
        
        # Possible image extensions to try (case-insensitive)
        possible_exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]
        for ext in possible_exts:
            image_path = os.path.join(image_dir, f"{image_id}{ext}")
            if Path(image_path).is_file():
                return image_path
                
        raise FileNotFoundError(f"Image file {image_id} not found in {image_dir}. Tested possible extensions: {possible_exts}.")
    
    except (ValueError) as e:
        logger.error(f"{e}")
        image_path = None

    return image_path

def imgid_to_annpath(image_id, data_base_dir=DATA_BASE_DIR):
    """
    Returns path of annotation-json-file for given image_id
    
    Parameters
    -------    
    image_id: str
        image_id, e.g. rec1_00001

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
    
    Returns
    -------
    Str
        Path to annotation with this id.

    Raises
    -------
    FileNotFound
        If data_base_dir or annotation-file are not found.
    """    
    data_base_dir = Path(data_base_dir)
    if not os.path.isdir(data_base_dir):
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")
    try:
        record_id = imgid_to_recid(image_id)
        annotation_path = os.path.join(data_base_dir, "annotations", f"{record_id}_annotations.json")
        annotation_path = Path(annotation_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Json file not found: {annotation_path}")
    
    except (ValueError) as e:
        logger.error(f"{e}")
        annotation_path = None

    return str(annotation_path)


def imgid_to_txtpath(image_id, data_base_dir=DATA_BASE_DIR):
    """
    Returns path of annotation-txt-file for given image_id
    
    Parameters
    -------    
    image_id: str
        image_id, e.g. rec1_00001

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
    
    Returns
    -------
    Str
        Path to annotation with this id.

    Raises
    -------
    FileNotFound
        If data_base_dir or annotation-file are not found.
    """    
    data_base_dir = Path(data_base_dir)
    if not os.path.isdir(data_base_dir):
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")
    try:
        record_id = imgid_to_recid(image_id)
        annotation_path = os.path.join(data_base_dir, "labels", record_id, f"{image_id}.txt")
        annotation_path = Path(annotation_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Txt file not found: {annotation_path}")
    
    except (ValueError) as e:
        logger.error(f"{e}")
        annotation_path = None

    return str(annotation_path)

def recid_to_annpath(record_id, data_base_dir=DATA_BASE_DIR):
    """
    Returns path of annotation-json-file for given record id
    
    Parameters
    -------  
    record_id: str
        record_id, e.g. rec1

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
    
    Returns
    -------
    Str
        Path to annotation with this id.

    Raises
    -------
    FileNotFound
        If data_base_dir or annotation-file are not found.
    """    
    data_base_dir = Path(data_base_dir)
    if not os.path.isdir(data_base_dir):
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")
    try:
        annotation_path = os.path.join(data_base_dir, "annotations", f"{record_id}_annotations.json")
        annotation_path = Path(annotation_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Json file not found: {annotation_path}")
    
    except (ValueError) as e:
        logger.error(f"{e}")
        annotation_path = None

    return str(annotation_path)

def get_image_size(image_id, data_base_dir=DATA_BASE_DIR):
    """
    Returns height, width of image
    
    Parameters
    -------  
    image_id: str
        image_id, e.g. rec1_000001

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
    
    Returns
    -------
    tuple(int, int)
        Height and width of image.

    Raises
    -------
    FileNotFound
        If image file is not found.
    """  
    image_path = imgid_to_imgpath(image_id, data_base_dir)
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(image_path))
    return img.shape[:2]