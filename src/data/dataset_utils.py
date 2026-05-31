# Dataset-level path management utils
import os
import cv2
import json
from pathlib import Path
from src.utils import logger, DATA_BASE_DIR

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DatasetParameters:
    """Container for dataset parameters."""
    nr_images: int
    img_width: int
    img_height: int
    first_file_name: str
    last_file_name: str

    def to_dict(self) -> Dict:
        """Convert dataclass to dictionary (useful for comparisons)."""
        return {
            "nr_images": self.nr_images,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "first_file_name": self.first_file_name,
            "last_file_name": self.last_file_name,
        }

def json_parameters(json_path):
    """
    Extract parameters of the dataset from the annotation file.

    Parameters
    ----------
    json_path : str or Path
        Path to the json file.

    Returns
    -------
    DatasetParameters
        Dataset parameters extracted from annotations.

    Raises
    ------
    FileNotFoundError
        If the json file does not exist.
    ValueError
        If the json file has no attribute "images".
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
            
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    nr_images = len(json_data['images'])
    if nr_images == 0:
        raise ValueError(f"No images found in annotations: {json_path}")

    sample_img = json_data['images'][0]
    img_width = sample_img['width']
    img_height = sample_img['height']

    file_names = sorted([img["file_name"] for img in json_data["images"]])
    
    return DatasetParameters(
        nr_images=nr_images,
        img_width=img_width,
        img_height=img_height,
        first_file_name=file_names[0],
        last_file_name=file_names[-1],
    )

def images_parameters(img_subdir_path):
    """
    Extract parameters of the dataset from the image subdirectory by paramters of the images themself.

    Parameters
    ----------
    img_subdir_path : str
        Path to the image subdirectory file.

    Returns
    -------
    DatasetParameters
        Dataset parameters extracted from images.

    Raises
    ------
    FileNotFoundError
        If the subdirecotry does not exist or it has no png-files.
    """
    img_subdir_path = Path(img_subdir_path)
    if not img_subdir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_subdir_path}")
        
    png_files = sorted([f for f in img_subdir_path.iterdir() if f.suffix.lower() == ".png"])
    if not png_files:
        raise FileNotFoundError(f"No PNG images found in: {img_subdir_path}")

    sample_img = cv2.imread(os.path.join(img_subdir_path, png_files[0]))
    if sample_img is None:
        raise ValueError(f"Failed to read image: {png_files[0]}")            
    img_height, img_width, _ = sample_img.shape
    
    return DatasetParameters(
        nr_images=len(png_files),
        img_width=img_width,
        img_height=img_height,
        first_file_name=png_files[0].name,
        last_file_name=png_files[-1].name,
    )

def json_image_match(data_base_dir=DATA_BASE_DIR):
    """
    Check if the key parameters of images and annotation match.

    Parameters
    ----------
    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.

    Returns
    -------
    Int
        total_image_nr: total number of images with matching anntoations.

    Raises
    ------
    FileNotFoundError
        If data_base_dir, img_dir or no subdirectories exist.
    """
    
    total_image_nr = 0
    
    data_base_dir = Path(data_base_dir)
    record_ids = list_record_ids(data_base_dir)


    for record_id in record_ids:
        img_subdir_path = data_base_dir / "images" / record_id
        json_path = data_base_dir / "annotations" / f"{record_id}_annotations.json"
        try:
            images_para = images_parameters(img_subdir_path)
            json_para = json_parameters(json_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"{record_id}: {e}")
            continue
        
        if images_para.to_dict() == json_para.to_dict():
            logger.info(f"{record_id}: images and annotations match.")
            total_image_nr += images_para.nr_images
        
        else:
            logger.warning(
                f"{record_id}: mismatch between images and annotations.\n"
                f"Image parameters: {images_para}\n"
                f"Annotation parameters: {json_para}")
    
    logger.info(f"Number of images with matching annotations: {total_image_nr}")
    return total_image_nr


def list_record_ids(data_base_dir=DATA_BASE_DIR) -> List[str]:
    """
    List all record ids in the dataset.

    Parameters
    ----------
    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.

    Returns
    -------
    List[str]
        List of record ids.

    Raises
    ------
    FileNotFoundError
        If data_base_dir or img_dir do not exist.
    """

    data_base_dir = Path(data_base_dir)
    if not data_base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")

    img_dir = data_base_dir/ "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    record_ids = [r.name for r in img_dir.iterdir() if r.is_dir() and r.name.startswith("rec")]
    if not record_ids:
        raise FileNotFoundError(f"No record ids found in dataset: {data_base_dir}")    

    return sorted(record_ids, key=lambda x: int(x.replace("rec", "")))

def recordid_to_imageids(record_id, data_base_dir=DATA_BASE_DIR) -> List[str]:
    """
    List all image ids in the given record id.

    Parameters
    ----------
    record_id: str
        record_id, e.g. rec1

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.

    Returns
    -------
    List[str]
        List of image ids.

    Raises
    ------
    FileNotFoundError
        If data_base_dir, img_dir or record_id do not exist.
    """
    data_base_dir = Path(data_base_dir)
    if not data_base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_base_dir}")

    img_dir = data_base_dir/ "images"/record_id
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
    image_ids = sorted(f.stem for f in img_dir.glob("*.png"))
    if not image_ids:
        raise FileNotFoundError(f"No PNG images found in: {img_dir}")
    
    return image_ids