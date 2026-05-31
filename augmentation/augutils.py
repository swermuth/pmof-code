import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
from glob import glob
from itertools import chain
import math

# === File I/O Utilities ===
# (functions for loading/saving files)
def get_image_and_label_paths(data_base_dir, image_id):
    """ Returns dictionary with image file path and label file path for a given image_id.
     
    Parameters
    ----------
    data_base_dir: string
        path to dataset, e.g. "/home/user/fisheye_data/"
    image_id: string
        image identifier, e.g. "subset1_000001"
        
    Returns
    -------
    dict
        image_path and label_path

    Raises
    ------
    ValueError
        If image_id has invalid format.
    FileNotFoundError
        If image or label file does not exist. 
    """

    match = re.match(r'(.+)_\d{6}(?:_\d{3})?$', image_id)
    if not match:
        raise ValueError(f"Invalid image_id format: {image_id}")
    subset_dir = match.group(1)

    # Image path (check both jpg, PNG and png)
    for ext in ['.jpg', '.PNG', '.png', '.jpeg']:
        image_path = os.path.join(data_base_dir, 'images', subset_dir, f'{image_id}{ext}')
        if os.path.exists(image_path):
            break
    else:
        raise FileNotFoundError(f"Image file not found for {image_id} in {data_base_dir}/images/{subset_dir}")

    # Label path
    label_path = os.path.join(data_base_dir, 'labels', subset_dir, f'{image_id}.txt')
    if not os.path.exists(label_path):
        print(f"Label file not found for {image_id} in {label_path}")

    return {'image_path': image_path, 'label_path': label_path}

def get_image_size(image_path):
    """ Returns height and width of image.
    
    Parameters
    ----------
    image_path: string
        path to image, e.g. "/home/user/fisheye_data/images/dir1/dir1_0001.png"
    
    Returns
    -------
    int, int
        height, width

    Raises
    ------
    FileNotFoundError
        If image does not exist. 
    """
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
    else:
        raise FileNotFoundError(f"Image file not found in {image_path}")
    return image_height, image_width

def scaled_bbox_from_file(label_path, image_height, image_width):
    """ Load bounding boxes from label file and scale them to image dimensions.

    Parameters
    ----------
    label_path: string
        path to label-file, e.g. "/home/user/fisheye_data/labels/dir1/dir1_0001.txt"
    image_height: int
        height of image in pixels
    image_widht: int
        width of image in pixels
    
    Returns
    -------
    list of list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        
    """
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            coords = list(map(float, line.split()[1:]))  # Extract x1, y1, ..., x4, y4
            keypoints = [(int(coords[i] * image_width), int(coords[i + 1] * image_height)) for i in range(0, 8, 2)] #scale from normalized coordinates to image dimensions 
            bboxes.append(keypoints)            
    return bboxes

def pascal_voc_from_scaled_box(bboxes):
    """ Pascal VOC standard box from scaled box.

    Parameters
    ----------
    bboxes: list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Returns
    -------
    list of lists
        list of bounding boxes; each bounding box is a list of two (x, y) coordinates: [x_min, y_min, x_max, y_max]
        
    """
    pascal_voc = []
    for box in bboxes:
        x_min = min(x for x, _ in box)
        x_max = max(x for x, _ in box)
        y_min = min(y for _, y in box)
        y_max = max(y for _, y in box)
        coords = [x_min, y_min, x_max, y_max]
        pascal_voc.append(coords)            
    return pascal_voc

# === Visualization Utilities ===
def visualize_image_with_bbox(image_array, bboxes, show=True):
    """ Visualize a single image with overlaid OBB bounding boxes.

    Parameters
    ----------
    image_array: np.ndarray
        Image array (without bounding boxes drawn).
    bboxes: list of list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates
    show : bool, optional
        Whether to display the image plot. Default is True.
    
    Returns
    -------
    np.ndarray
        Image array with bounding boxes drawn.
    
    Raises
    ------
    ValueError
        If a bounding box has an unexpected shape.
    """
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        coords = np.array(bbox)
        if coords.shape!=(4,2):
            raise ValueError(f'Bounding Box has wrong shape: {coords.shape}')
        else:
            cv2.polylines(image, [coords], isClosed=True, color=(0, 255, 255), thickness=3)

    if show:
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    return image

def visualize_image_with_pascal_bbox(image_array, bboxes=None, pascal_bboxes=None, show=True):
    """ Visualize a single image with overlaid OBB bounding boxes.

    Parameters
    ----------
    image_array: np.ndarray
        Image array (without bounding boxes drawn).
    bboxes: list of list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates
    pascal_bboxes: list of list, optional
        list of bounding boxes; each bounding box is a list of x_min, y_min, x_max, y_max
    show : bool, optional
        Whether to display the image plot. Default is True.
    
    Returns
    -------
    np.ndarray
        Image array with bounding boxes drawn.
    
    Raises
    ------
    ValueError
        If a bounding box has an unexpected shape.
    """
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    if bboxes:
        for bbox in bboxes:
            coords = np.array(bbox)
            if coords.shape!=(4,2):
                raise ValueError(f'Bounding Box has wrong shape: {coords.shape}')
            else:
                cv2.polylines(image, [coords], isClosed=True, color=(0, 255, 255), thickness=3)

    if pascal_bboxes:
        for box in pascal_bboxes:
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 255), thickness=3) #top-left corner, #bottom-right corner
    
    if show:
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    return image



def visualize_image_ids(data_base_dir, image_ids, save_path=None):
    """ Visualize a collage of single or multiple image_ids.
    Only works, if images and labels are saved!

    Parameters
    ----------
    data_base_dir: string
        path to dataset, e.g. "/home/user/fisheye_data/"
    image_ids: list
        List of image_ids.
    save_path: string, optional
        path where to save the image, e.g. "image.png"

    Returns
    -------
    no returns
    
    """
    
    num_images = len(image_ids)
    columns = min(4, num_images)  # limit max columns to avoid tiny subplots
    rows = math.ceil(num_images / columns)
    
    fig, axs = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
    axs = axs.flatten() if num_images > 1 else [axs]
    
    for i, image_id in enumerate(image_ids):
        paths = get_image_and_label_paths(data_base_dir=data_base_dir, image_id=image_id)
        image_path = paths["image_path"]
        label_path = paths["label_path"]
    
        h, w = get_image_size(image_path)
        bboxes = scaled_bbox_from_file(label_path, h, w)
    
        image = cv2.imread(image_path)
        image = visualize_image_with_bbox(image, bboxes, show=False)
    
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(image_id, fontsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# === Augmentation & Transformation ===
def apply_transformation(image_array, bboxes, transform):
    """ Transforms image and bounding box.
    
    Parameters
    ----------
    image_array: np.ndarray
        Image array of original image.
    bboxes: list of list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates
    transform: albumentations.core.composition.Compose
        Compose of Albumentations transformations
        
    Returns
    -------
    np.ndarray
        Image array of transformed image.
    np.ndarray
        Array of transformed bounding boxes.

    """  
    # Load and flatten keypoints
    keypoints_flat = list(chain.from_iterable(bboxes))

    # Apply augmentations
    transformed = transform(image=image_array, keypoints=keypoints_flat)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]
    
    # Re-group keypoints into 4-point polygons
    regrouped_keypoints = [transformed_keypoints[i:i + 4] for i in range(0, len(transformed_keypoints), 4)]
    transformed_bboxes = [np.array(box, dtype=np.int32).reshape((4, 2)) for box in regrouped_keypoints]
    
    return transformed_image, transformed_bboxes

def apply_transformation_pascalvoc(image_array, bboxes, transform):
    """ Transforms image and bounding box.
    
    Parameters
    ----------
    image_array: np.ndarray
        Image array of original image.
    bboxes: list of list of tuple
        list of bounding boxes; each bounding box is a list of four (x, y) coordinates
    transform: albumentations.core.composition.Compose
        Compose of Albumentations transformations
        
    Returns
    -------
    np.ndarray
        Image array of transformed image.
    np.ndarray
        Array of transformed bounding boxes.

    """  
    class_labels=['person']*len(bboxes)
    # Apply augmentations
    transformed = transform(image=image_array, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]
    
    return transformed_image, transformed_bboxes
    
def convert_to_yolo_obb(bboxes, image_height,  image_width, class_id=0):
    """ Normalize bounding box coordinates for YOLO-OBB format.

    Parameters
    ----------
    bboxes: np.ndarray
        Array of bounding boxes coordinates.
    image_height: int
        height of image in pixels
    image_widht: int
        width of image in pixels
    class_id: int, optional
        Class label for all bounding boxes. Default is 0.
    
    Returns
    -------
    list:
        List with one string per line for a yolo-obb-format txt-label file.    
    """
    yolo_obb_lines = []
    
    for box in bboxes:
        normalized_points = []
        for x, y in box:
            x_norm = x / image_width
            y_norm = y / image_height
            normalized_points.extend([x_norm, y_norm])
            
        yolo_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
        yolo_obb_lines.append(yolo_line)

    return yolo_obb_lines

def convert_to_yolo_obb(bboxes, image_height,  image_width, class_id=0):
    """ Normalize bounding box coordinates for YOLO-OBB format.

    Parameters
    ----------
    bboxes: np.ndarray
        Array of bounding boxes coordinates.
    image_height: int
        height of image in pixels
    image_widht: int
        width of image in pixels
    class_id: int, optional
        Class label for all bounding boxes. Default is 0.
    
    Returns
    -------
    list:
        List with one string per line for a yolo-obb-format txt-label file.    
    """
    yolo_obb_lines = []
    
    for box in bboxes:
        normalized_points = []
        for x, y in box:
            x_norm = x / image_width
            y_norm = y / image_height
            normalized_points.extend([x_norm, y_norm])
            
        yolo_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
        yolo_obb_lines.append(yolo_line)

    return yolo_obb_lines
    
def save_augmented_image_and_label(data_base_dir, aug_suffix, image_id, transformation_index, transformed_image, yolo_obb_lines, aug_data_base_dir=None):
    """
    Saves an augmented image and label-file to a corresponding directory for augmented files and return information about these files.

    Parameters
    ----------
    data_base_dir: string
        path to dataset, e.g. "/home/user/fisheye_data/"
    aug_suffix: string
        suffix for direcotry and file names
    image_id: string
        image identifier, e.g. "subset1_000001"
    transformation_index: int
        index of transformation of this single image
    transformed_image: np.ndarray
        Image array of original image.
    yolo_obb_lines: list of strings
        List with one string per line for a yolo-obb-format txt-label file.
    aug_data_base_dir: string (optional)
        path where to save the augmented data, defaults to data_base_dir
    
    Returns
    -------
     dict
        new image_id, image_path and label_path
    """
    if not aug_data_base_dir:
        aug_data_base_dir = data_base_dir
        
    # Extract subset name from image_id (e.g. 'subset1' from 'subset1_000001')
    match = re.match(r'(.+?)_\d{6}$', image_id)
    if not match:
        raise ValueError(f"Invalid image_id format: {image_id}")
    subset_name = match.group(1)
    augmented_subset = f"{subset_name}_{aug_suffix}"

    # Create new image_id
    new_image_id = f"{augmented_subset}_{image_id[-6:]}_{transformation_index:03}"

    # Save image
    image_output_dir = os.path.join(aug_data_base_dir, "images", augmented_subset)
    os.makedirs(image_output_dir, exist_ok=True)
    image_path = os.path.join(image_output_dir, f"{new_image_id}.PNG")
    #Image.fromarray(transformed_image).save(image_path)
    cv2.imwrite(image_path, transformed_image)
    
    # Save label
    label_output_dir = os.path.join(aug_data_base_dir, "labels", augmented_subset)
    os.makedirs(label_output_dir, exist_ok=True)
    label_path = os.path.join(label_output_dir, f"{new_image_id}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_obb_lines))

    return {'image_id': new_image_id, 'image_path': image_path, 'label_path': label_path}

def augmentation_saving_pipeline(data_base_dir, aug_suffix, image_id, number_transformations, transform, aug_data_base_dir=None):
    """
    Full augmentation pipeline including saving for a single image_id.

    Parameters
    ----------
    data_base_dir: string
        path to dataset, e.g. "/home/user/fisheye_data/"
    aug_suffix: string
        suffix for direcotry and file names
    image_id: string
        image identifier, e.g. "subset1_000001"
    number_transformations: int
        number of transformation to execute
    transform: albumentations.core.composition.Compose
        Compose of Albumentations transformations
    aug_data_base_dir: string (optional)
        path where to save the augmented data, defaults to data_base_dir

    Returns
    -------
     list
        list of strings with new image_id's
    """
    if not aug_data_base_dir:
        aug_data_base_dir = data_base_dir
    
    paths = get_image_and_label_paths(data_base_dir=data_base_dir, image_id=image_id)
    image_path = paths["image_path"]
    label_path = paths["label_path"]
    
    image_array = cv2.imread(image_path)
    h, w = get_image_size(image_path)
    bboxes = scaled_bbox_from_file(label_path, h, w)
    
    transformed_ids = []
    for i in range(number_transformations):
        transformed_image, transformed_bboxes = apply_transformation(image_array, bboxes, transform)
        transformed_height, transformed_width, _ = transformed_image.shape
        if any(x > transformed_width or y > transformed_height for box in transformed_bboxes for x, y in box): #if bounding boxes reach outsite of image, continue
            continue
        yolo_obb_lines = convert_to_yolo_obb(bboxes=transformed_bboxes, image_height=transformed_height, image_width=transformed_width)
        transformed_id = save_augmented_image_and_label(data_base_dir=data_base_dir, aug_suffix=aug_suffix, image_id=image_id, transformation_index=i, transformed_image=transformed_image, yolo_obb_lines=yolo_obb_lines, aug_data_base_dir=aug_data_base_dir)
        transformed_ids.append(transformed_id['image_id'])
    return transformed_ids


def augmentation_saving_pipeline_pascalvoc(data_base_dir, aug_suffix, image_id, number_transformations, transform, aug_data_base_dir=None):
    """
    Full augmentation pipeline including saving for a single image_id.

    Parameters
    ----------
    data_base_dir: string
        path to dataset, e.g. "/home/user/fisheye_data/"
    aug_suffix: string
        suffix for direcotry and file names
    image_id: string
        image identifier, e.g. "subset1_000001"
    number_transformations: int
        number of transformation to execute
    transform: albumentations.core.composition.Compose
        Compose of Albumentations transformations
    aug_data_base_dir: string (optional)
        path where to save the augmented data, defaults to data_base_dir

    Returns
    -------
     list
        list of strings with new image_id's
    """
    if not aug_data_base_dir:
        aug_data_base_dir = data_base_dir
    
    paths = get_image_and_label_paths(data_base_dir=data_base_dir, image_id=image_id)
    image_path = paths["image_path"]
    label_path = paths["label_path"]
    
    image_array = cv2.imread(image_path)
    h, w = get_image_size(image_path)
    bboxes = scaled_bbox_from_file(label_path, h, w)
    pascal_voc_bboxes = pascal_voc_from_scaled_box(bboxes)
    
    transformed_ids = []
    for i in range(number_transformations):
        transformed_image, transformed_bboxes = apply_transformation_pascalvoc(image_array, bboxes, transform)
        transformed_height, transformed_width, _ = transformed_image.shape
        if any(x > transformed_width or y > transformed_height for box in transformed_bboxes for x, y in box): #if bounding boxes reach outsite of image, continue
            continue
        yolo_prep_bboxes = []
        for transformed_box in transformed_bboxes:
            x_min, y_min, x_max, y_max = transformed_box
            yolo_prep_boxes.append(x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min)
        yolo_obb_lines = convert_to_yolo_obb(bboxes=yolo_prep_boxes, image_height=transformed_height, image_width=transformed_width)
        transformed_id = save_augmented_image_and_label(data_base_dir=data_base_dir, aug_suffix=aug_suffix, image_id=image_id, transformation_index=i, transformed_image=transformed_image, yolo_obb_lines=yolo_obb_lines, aug_data_base_dir=aug_data_base_dir)
        transformed_ids.append(transformed_id['image_id'])
    return transformed_ids
    
# === Selection ===
def get_random_image_ids(dirs, percentage, seed=0):
    """ Randomly selects a given percentage of images_ids from each directory.
    
    Parameters
    ----------
    dirs: list
        list with strings to image-directories, e.g. "/home/user/fisheye_data/images/subset1"
    percentage: int
        percentage of all images in directories to select
    seed: int (optional)
        seed for random selection
    
    Returns
    -------
    list
        list of strings of randomly selected image_ids

    Raises
    ------
    ValueError
        If percentage is not between 0 and 100.
    FileNotFoundError
        If one of the directories is not found, or if it has no image-file of the type PNG or JPEG.
    """
    random.seed(seed)
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    selected_ids = []

    for directory in dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Collect files with all supported extensions
        image_files = []
        for ext in ['*.jpg', '*.PNG', '*.png']:
            image_files.extend(glob(os.path.join(directory, ext)))

        if not image_files:
            raise FileNotFoundError(f"No image files found in directory: {directory}")
            
        # Extract image IDs without extension
        image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        
        # Shuffle and select 50%
        k = max(1, int(len(image_ids) * (percentage / 100.0)))
        selected = random.sample(image_ids, k)
        selected_ids.extend(selected)

    return selected_ids