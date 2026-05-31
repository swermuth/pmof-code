# src/visualization/frame_visualizer.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from src.data import imgid_to_imgpath, imgid_to_annpath, read_annotation
from src.visualization.viz_config import VIZ_PARAMS
from src.visualization.color_utils import colorstr_to_bgr 
from src.utils import logger

class LegendBuilder:
    """ Collects legend handels and renders and RGBA matplotlib legend image for overlaying on frames. """
    def __init__(self, linewidth=6):
        self._handles_dict = {}
        self.linewidth = linewidth

    def add_handle(self, label_text, color):
        """ Add a legend handle if not already present. """
        if label_text not in self._handles_dict:
            self._handles_dict[label_text] = Patch(facecolor="none", edgecolor=color, label=label_text, linewidth=self.linewidth)

    def handles(self):
        """ Return list of legend handles. """
        return list(self._handles_dict.values())
    
    def render_legend_rgba(self, title=None, fontsize=30, title_fontsize=35, ncol=1):
        """ Render legend as RGBA image. """
        if len(self._handles_dict) == 0:
            return None
        
        fig = plt.figure(figsize=(5, len(self._handles_dict)+3))
        ax = fig.add_subplot(111)
        ax.axis('off')
        leg = ax.legend(handles=self.handles(), loc='center', frameon=True, fontsize=fontsize, title=title, title_fontsize=title_fontsize, ncol=ncol, labelcolor='black')
        
        fig.patch.set_alpha(0.0)  # fully transparent background

        canvas = FigureCanvas(fig)
        canvas.draw()
        legend_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        legend_img = legend_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        plt.close(fig)
        
        return legend_img

class BoxDrawer:
    """ Draws bounding boxes for predictions and ground truth on image arrays. """
    def __init__(self, gt_colors, pred_colors, bbox_thickness):
        self.gt_colors = gt_colors
        self.pred_colors = pred_colors
        self.bbox_thickness = bbox_thickness
        
    def draw_pred(self, img_bgr, obb, names_dict):
        """ Draw predicted bounding box and confidence score on image array. Return label and color used. """
        cls = obb.cls.item()
        label = names_dict.get(cls, "prediction")
        conf =float(obb.conf.item())

        bbox_coords = obb.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(int)

        color = self.pred_colors.get(label, (255, 255, 255))  # default white if label not found
        cv2.polylines(img_bgr, [bbox_coords], isClosed=True, color=colorstr_to_bgr(color), thickness=self.bbox_thickness)
        self._draw_conf(img_bgr, bbox_coords[0], conf)

        return label, color
    
    def draw_gt(self, img_bgr, annotation, actions=False):
        """ Draw ground truth bounding box on image array. Return label and color used. """
        if annotation.occluded:
            return None, None  # skip occluded boxes
        bbox_coords = np.array(annotation.xyxyxyxy).reshape(4, 2).astype(int)
        label = annotation.category_name
        if actions: 
            label = annotation.action or annotation.category_name
        color = self.gt_colors.get(label, (0, 255, 0))  # default green if label not found
        cv2.polylines(img_bgr, [bbox_coords], isClosed=True, color=colorstr_to_bgr(color), thickness=self.bbox_thickness)
        
        return label, color
    
    def _draw_conf(self, img_bgr, coord, conf):
        """ Draw confidence score at corner of bounding box. """
        x, y = coord
        conf_text = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        text_color = (0, 0, 0)  # Black text color
        text_bg_color = (255, 255, 255)  # White background for text
        cv2.rectangle(img_bgr, (x, y - text_h - 10), (x + text_w, y), text_bg_color, -1)
        cv2.putText(img_bgr, conf_text, (x, y - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

class FrameVisualizer:
    """ Main orchestrator: draws gt and prediction boxes, builds legend, overlays legend on frames. """
    def __init__(self, data_base_dir, viz_params=VIZ_PARAMS):
        self.data_base_dir = data_base_dir
        self.viz_params = viz_params
        self.box_drawer = BoxDrawer(
            gt_colors=viz_params['gt_bbox_colors'],
            pred_colors=viz_params['pred_bbox_colors'],
            bbox_thickness=viz_params['bbox_lines']['thickness']
        )
        self.legend_builder = LegendBuilder(linewidth=viz_params['legend']['linewidth'] if 'legend' in self.viz_params else 6)

    def visualize_results(self, results, actions=False, display_gt=True):
        """ Visualize prediction results by drawing prediction and gt boxes, building legend, overlaying legend on frames. Option to display actions as labels. Default is False (show category names instead of actions)"""
        frames = {}
        for result in tqdm(results, desc="Visualizing results"):
            image_id = Path(result.path).stem
            img_bgr = result.orig_img.copy()
            names_dict = result.names

            for obb in result.obb:
                label, color = self.box_drawer.draw_pred(img_bgr, obb, names_dict)
                if label and color:
                    self.legend_builder.add_handle(f"Pred. {label}", color)

            if display_gt:
                annotation_path = imgid_to_annpath(data_base_dir=self.data_base_dir, image_id=image_id)
                annotations = read_annotation(annotation_path=annotation_path, image_id=image_id)
                for ann in annotations:
                    label, color = self.box_drawer.draw_gt(img_bgr, ann, actions=actions)
                    if label and color:
                        self.legend_builder.add_handle(f"GT {label}", color)
            
            out_bgr = self._overlay_legend(img_bgr, title=image_id)
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            frames[image_id] = out_rgb
        return frames

    def visualize_groundtruth(self, image_ids, actions):
        """ 
        Visualize only ground truth drawing gt boxes, building legend, overlaying legend on frames. 

        Parameters
        -----
        image_ids: list[strings]
            list with image_ids, e.g. ["rec1_000001", "rec1_000002"]

        actions: Boolean
            if True, the action will be become the label, if false "person" will be the label

        Returns
        -----
        dict: {image_id:np.arr}
            image_id and frame as np.array
        """
        frames = {}
        for image_id in tqdm(image_ids, desc="Visualizing ground truth annotation"):
            image_path = imgid_to_imgpath(data_base_dir=self.data_base_dir, image_id=image_id)
            img_bgr = cv2.imread(image_path)
            
            annotation_path = imgid_to_annpath(data_base_dir=self.data_base_dir, image_id=image_id)
            annotations = read_annotation(annotation_path=annotation_path, image_id=image_id)
            for ann in annotations:
                label, color = self.box_drawer.draw_gt(img_bgr, ann, actions=actions)
                if label and color:
                    self.legend_builder.add_handle(f"GT {label}", color)
            
            out_bgr = self._overlay_legend(img_bgr, title=image_id)
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            frames[image_id] = out_rgb
        return frames

    def _overlay_legend(self, img_bgr, title=None):
        """ Overlay legend on image array. """
        legend_rgba = self.legend_builder.render_legend_rgba(title=title)
        if legend_rgba is None:
            return img_bgr  # no legend to overlay
        legend_rgb = legend_rgba[..., :3] 
        legend_bgr = legend_rgb[..., ::-1]
        alpha = legend_rgba[..., 3:] / 255.0

        h, w, _ = img_bgr.shape
        ly, lx = legend_rgba.shape[:2]
        offset_x, offset_y = w - lx - 20, h - ly - 20
        y1, y2 = offset_y, offset_y + ly
        x1, x2 = offset_x, offset_x + lx

        img_bgr[y1:y2, x1:x2] = (alpha * legend_bgr + (1 - alpha) * img_bgr[y1:y2, x1:x2]).astype(np.uint8)
        return img_bgr
    
def results_to_frames(results, data_base_dir, viz_params=VIZ_PARAMS, actions=None, display_gt=True):
    """Visualize results from yolo11-obb and return frames as dict of image arrays. """
    visualizer = FrameVisualizer(data_base_dir=data_base_dir, viz_params=viz_params)
    frames = visualizer.visualize_results(results, actions=actions, display_gt=display_gt)
    return frames

def imageids_to_gtframes(image_ids, data_base_dir, actions=None, viz_params=VIZ_PARAMS):
    """Visualize results from yolo11-obb and return frames as dict of image arrays. """
    visualizer = FrameVisualizer(data_base_dir=data_base_dir, viz_params=viz_params)
    frames = visualizer.visualize_groundtruth(image_ids, actions=actions)
    return frames