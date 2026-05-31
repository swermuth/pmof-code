#augmentation_samples.py
# Plotting augmented image samples for visualization
from src.visualization.viz_config import VIZ_PARAMS
from src.visualization.color_utils import colorstr_to_bgr 
from src.utils import logger
from pathlib import Path
from typing import Dict, List, Optional

class ImagePathResolver:
    """Resolve image paths based on base directory and image id."""
    def __init__(self, data_base_dir: Path, aug_data_base_dir : Optional[Path] = None, num_aug: int = 3, extensions: Optional[List[str]] = None):
        self.data_base_dir = Path(data_base_dir)
        self.aug_data_base_dir = Path(aug_data_base_dir) if aug_data_base_dir else self.data_base_dir
        self.num_aug = num_aug
        self.extensions = extensions if extensions else ['.jpg', '.jpeg', '.png', '.PNG']

    def parse_image_id(self, image_id: str) -> Dict[str, str]:
        """Parse image id to directory."""
        parts = image_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"Invalid image_id format: {image_id}")
        base_name, aug_code, og_num, aug_num = parts
        return {
            "og_id": f"{base_name}_{og_num}",
            "og_dir": base_name,
            "og_num": og_num,
            "aug_dir": f"{base_name}_{aug_code}",
            "aug_num": aug_num,
        }
    
    def find_existing_image(self, base_dir: Path, subdir: str, image_id: str) -> Optional[Path]:
        """Find first existing image file in the specified image id."""
        img_folder = base_dir / "images" / subdir
        for ext in self.extensions:
            candidate = img_folder / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        return None
    
    def find_existing_label(self, base_dir: Path, subdir: str, image_id: str) -> Optional[Path]:
        """Find first existing label txt file in the specified image id."""
        label_folder = base_dir / "labels" / subdir
        candidate = label_folder / f"{image_id}.txt"
        if candidate.exists():
            return candidate
        return None
    
    def get_paths(self, image_id: str) -> Dict:
        """Get original and augmented image paths."""
        parsed = self.parse_image_id(image_id)

        #  ---- Original Image and Label Paths ----
        og_img_path = self.find_existing_image(self.data_base_dir, parsed["og_dir"], parsed["og_id"])
        og_label_path = self.find_existing_label(self.data_base_dir, parsed["og_dir"], parsed["og_id"])
        
        # ---- Augmented Image and Label Paths ----
        aug_ids = [f'{parsed["aug_dir"]}_{parsed["og_num"]}_{i:03d}' for i in range(self.num_aug)]
        aug_paths = []
        for aug_id in aug_ids:
            img_path = self.find_existing_image(self.aug_data_base_dir, parsed["aug_dir"], aug_id)
            label_path = self.find_existing_label(self.aug_data_base_dir, parsed["aug_dir"], aug_id)
            aug_paths.append({
                "aug_id": aug_id,
                "img_path": img_path,
                "label_path": label_path
            })

        return {
            "og_id": parsed["og_id"],
            "og_img_path": og_img_path,
            "og_label_path": og_label_path,
            "augmented": aug_paths}
    