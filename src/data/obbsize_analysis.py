""""Module for analyzing oriented bounding box (OBB) sizes in datasets."""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from math import atan2, degrees
from src import logger, DATA_BASE_DIR

class OBBBox:
    "Represents an oriented bounding box (OBB) with normalized coordinates."
    def __init__(self, corners, cls=None):
        self.corners = np.array(corners, dtype=np.float64).reshape(4, 2)
        self.centroid = np.mean(self.corners, axis=0)
        self.sorted_corners = self._sort_corners()

        self.width, self.height = self._compute_dimensions()
        self.rotation = self._compute_rotation()
        self.area = self._compute_area()
        self.cls = int(cls) if cls is not None else None

    def _sort_corners(self):
        """Sort corners by angle around the centroid to ensure the consistent order"""
        angles = np.arctan2(
            self.corners[:, 1] - self.centroid[1], 
            self.corners[:, 0] - self.centroid[0]
            )
        sorted_indices = np.argsort(angles)
        return self.corners[sorted_indices]
    
    def _compute_dimensions(self):
        """Compute normalized width and normalized height of the OBB"""
        edge1 = np.linalg.norm(self.sorted_corners[0] - self.sorted_corners[1])
        edge2 = np.linalg.norm(self.sorted_corners[1] - self.sorted_corners[2])
        return min(edge1, edge2), max(edge1, edge2)
    
    def _compute_rotation(self):
        """Compute rotation angle of the OBB in degrees relative to x-axis"""
        edge_vector = self.sorted_corners[1] - self.sorted_corners[0]
        angle_rad = atan2(edge_vector[1], edge_vector[0])
        return degrees(angle_rad) % 360
    
    def _compute_area(self):
       """Compute the polygon area using the shoelace formula."""
       x = self.sorted_corners[:, 0]
       y = self.sorted_corners[:, 1]
       return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) 
    
    def to_dict(self, image_id):
        """Return dictionary for DataFrame construction."""
        return {
            "image_id": image_id,
            "class": self.cls,
            "centroid_x": self.centroid[0],
            "centroid_y": self.centroid[1],
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "area": self.area
        }
    
class OBBSizeAnalyzer:
    """Handels OBB statistics for one dataset."""
    def __init__(self, dataset_name, data_base_dir=DATA_BASE_DIR):
        self.dataset_name = dataset_name
        self.data_base_dir = Path(data_base_dir)
        self.image_dir = self.data_base_dir / "images" / dataset_name
        self.label_dir = self.data_base_dir / "labels" / dataset_name
        self.results = []

    def _get_label_path(self, image_id):
        """Get the label file path for a given image ID."""
        return self.label_dir / f"{image_id}.txt"
    
    def _read_label_file(self, label_path):
        """Read the label file and return OBB corner coordinates."""
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    logger.warning(f"Invalid label format in {label_path}: {line.strip()}")
                    continue
                cls = parts[0]
                corners = list(map(float, parts[1:9]))
                yield cls,corners

    def process_single_imageid(self, image_id):
        """Process a single image ID to extract OBB statistics."""
        label_path = self._get_label_path(image_id)
        if not label_path.exists():
            logger.warning(f"Label file not found for image ID {image_id}")
            return
        
        for cls, corners in self._read_label_file(label_path):
            obb = OBBBox(corners = corners, cls=cls)
            self.results.append(obb.to_dict(image_id))

    def run(self):
        """Process all images in the dataset."""
        logger.info(f"Starting with dataset: {self.dataset_name}")
        image_ids = [p.stem for p in self.image_dir.iterdir() if p.suffix in {'.jpg', '.png', '.jpeg', '.PNG'}]
        for image_id in image_ids:
            self.process_single_imageid(image_id)
        logger.info(f"Completed processing dataset: {self.dataset_name}")

    def to_dataframe(self):
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_results(self, output_dir='.'):
        """Save the results DataFrame to a CSV file."""
        df = self.to_dataframe()
        output_path = Path(output_dir) / f"{self.dataset_name}_obb_stats.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return output_path
    
class MultiDatasetOBBSizeAnalyzer:
    """Convenience wrapper for running multiple datasets."""
    def __init__(self, datasets, data_base_dir=DATA_BASE_DIR, output_dir='.'):
        self.datasets = datasets
        self.data_base_dir = data_base_dir
        self.output_dir = Path(output_dir)

    def run_all(self):
        """Run OBB size analysis for all datasets."""
        for dataset in self.datasets:
            analyzer = OBBSizeAnalyzer(dataset, self.data_base_dir)
            analyzer.run()
            analyzer.save_results(self.output_dir)
