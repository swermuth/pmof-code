import cv2
import numpy as np
from matplotlib.patches import Patch

from src.visualization.viz_config import VIZ_PARAMS
from src.data import imgid_to_imgpath, imgid_to_annpath
from src.data import read_annotation
from src.visualization import colorstr_to_bgr
from src.utils import logger, DATA_BASE_DIR

def imgid_to_imgarray(image_id, annotated=False, data_base_dir=DATA_BASE_DIR, actions=True):
    """
    Returns numpy array of image with or without bounding boxes.

    Parameters
    -------        
    image_id: str
        image_id, e.g. rec1_00001

    data_base_dir: str or Path (optional)
        Path to the dataset. Defaults to global DATA_BASE_DIR.
    
    annotated: Boolean (optional)
        if True: display bounding boxes on image

    actions: Boolean (optional)
        if False: label class "person" instead of action

    Returns
    -------
    numpy array
        Image as numpy array.

    legend_handles : list[Patch] or None
        Matplotlib legend handles for the labels actually present in the image.

    Raises
    -------
    FileNotFound
        If the image is not found. Warning, if the annotation is not found but asked for.
    """   
    image_path = imgid_to_imgpath(data_base_dir=data_base_dir, image_id=image_id)
    BGR_img = cv2.imread(image_path)

    legend_handles_dict = {} 
    if annotated==True:
        annotation_path = imgid_to_annpath(data_base_dir=data_base_dir, image_id=image_id)
        annotations = read_annotation(annotation_path=annotation_path, image_id=image_id)
        for annotation in annotations:
            if annotation.occluded==False:
                box_coords = np.array(annotation.xyxyxyxy).reshape(4, 2).astype(int)
                label = annotation.category_name
                if actions: 
                    label = annotation.action or annotation.category_name
                color = VIZ_PARAMS['gt_bbox_colors'][label]
                cv2.polylines(BGR_img, [box_coords], isClosed=True, color=colorstr_to_bgr(color), thickness=VIZ_PARAMS['bbox_lines']['thickness'])
                # Build legend handle
                if label not in legend_handles_dict:
                    legend_handles_dict[label] = Patch(facecolor="none", edgecolor=color, label=label)
                
    legend_handles = list(legend_handles_dict.values())
    img_arr = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
   
    return img_arr, legend_handles