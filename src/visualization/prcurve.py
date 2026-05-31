import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from src.utils import logger

class PRRun:
    """Represents one YOLO model checkpoint for PR-curve evaluation. 
    Handles caching, inference and JSON result loading.
    """
    def __init__(self, model_path, plot_name, color, linestyle='-', val_yaml="./yaml/PMOF-person.yaml", conf=0.25, device="0"):
        self.model_path = Path(model_path)
        self.plot_name = plot_name
        self.color = color
        self.linestyle = linestyle
        self.val_yaml = Path(val_yaml)
        self.conf = conf
        self.device = device

        self.recall = None
        self.prec = None
        self.ap50 = None
        self.p = None
        self.r = None
        self.f1 = None
        self.ms_per_frame = None
        self.fps = None

        # Resolve the actual .pt model path
        self.model_path = self._resolve_model_path()

        self.run_dir = self._get_run_dir()
        self.run_id = self._get_run_id()
        self.training_imgsz = self._get_imgsz()
        self.val_set = self.val_yaml.stem
        self.cache_path = self.run_dir / f"pr_{self.run_id}_{self.val_set}_{int(self.conf*100)}.json"

    def _resolve_model_path(self):
        """ Check if the model path exists and if it's a valid YOLO model file. If not, identify the saved epoch with highest mAP50. """
        model_path = Path(self.model_path)
        
        if not Path(model_path).exists():
            logger.error(f"Path {model_path} does not exist.")
        
        # Case 1: explicit checkpoint file provided
        if model_path.is_file() and model_path.suffix == '.pt':
            logger.info(f"Using provided model checkpoint: {model_path}")
            return model_path
        
        # Case 2: directory provided, find best available epoch
        elif model_path.is_dir():
            logger.info(f"Model path {model_path} is a directory. Searching for best epoch...")
            ckpt_path = self._find_best_ckpt(model_path)
            if ckpt_path is None:
                logger.error(f"Could not determine best epoch in {model_path}.")
            if not ckpt_path.exists():
                logger.error(f"Best checkpoint {ckpt_path} does not exist.")
            logger.info(f"Using model: {ckpt_path}.")
            return ckpt_path
        
        # Case 3: invalid input
        else:
            logger.error(f"Could not resolve model path: {model_path}. Must be a .pt file or a directory.")
            return None
        
    @staticmethod
    def _find_best_ckpt(run_dir):
        """ Find the highest mAP50 epoch for which a checkpoint file exists. If the best epoch's .pt file is missing, use the next best available. Returns path to the checkpoint file."""        
        results_csv = run_dir / 'results.csv'
        weights_dir = run_dir / "weights"

        if not results_csv.exists():
            logger.error(f"No results.csv found in {run_dir} to determine best epoch.")
            return None
        
        results = pd.read_csv(results_csv)
        if 'metrics/mAP50(B)' not in results.columns:
            logger.error(f"results.csv missing 'metrics/mAP50(B)' column: {results_csv}")
            return None
        
        # if best index for mAP50 = index for best mAP50-95, checkpoint is "best"
        if results['metrics/mAP50(B)'].idxmax() == results['metrics/mAP50-95(B)'].idxmax():
            epoch = int(results['epoch'][results['metrics/mAP50(B)'].idxmax()])
            ckpt = "best"
            ckpt_path = weights_dir / f'{ckpt}.pt'
            if ckpt_path.exists():
                logger.info(f"Best checkpoint found: epoch {epoch}, ckpt {ckpt} {ckpt_path} with mAP50: {results['metrics/mAP50(B)'].max():.3f}.")
                return ckpt_path
            else:
                logger.warning(f"Best checkpoint best.pt not found at {ckpt_path}. Searching for next best epoch.")

        results_sorted = results.sort_values(by='metrics/mAP50(B)', ascending=False)
        for _, row in results_sorted.iterrows():
            epoch = int(row['epoch'])
            ckpt = epoch - 1  # zero-based index for checkpoints
            ckpt_path = weights_dir / f'epoch{ckpt}.pt'
            if epoch == len(results): # last epoch saved as 'last.pt'
                ckpt = 'last'
                ckpt_path = weights_dir / f'{ckpt}.pt'
            if ckpt_path.exists():
                logger.info(f"Best epoch found: epoch {epoch}, ckpt {ckpt} with mAP50: {row['metrics/mAP50(B)']:.3f}.")
                return ckpt_path  # zero-based index
            else:
                logger.warning(f"Checkpoint for epoch {epoch}, ckpt {ckpt} not found at {ckpt_path}. Trying next best epoch.")
        logger.error("No valid checkpoints found in weights directory {weights_dir}.")
        return None


    def _get_run_dir(self):
        """ Extract the run directory form model path. 
        e.g. './runs/obb/myrun/weights/epoch7.pt' -> './runs/obb/myrun'
        """
        path_parts = Path(self.model_path).parts
        if "weights" in path_parts:
            weights_index = path_parts.index("weights")
            run_dir = Path(*path_parts[:weights_index])
        else:
            logger.error(f"Model path {self.model_path} does not contain 'weights' directory.")
        return run_dir
    
    def _get_run_id(self):
        """ Extract the run ID from model path.
        e.g. './runs/obb/myrun/weights/epoch7.pt' -> 'myrun_epoch7'
        """
        run_name = Path(self.run_dir).name  # e.g. 'myrun'
        weight_file = Path(self.model_path).stem  # e.g. 'epoch7'
        run_id = f"{run_name}_{weight_file}"
        return run_id
    
    def _get_imgsz(self):
        """ Extract training image size from YOLO model config. """
        imgsz = int(self.run_id.split('-')[2])  # assuming format includes imgsz as third part
        return imgsz
    
    def load_cache(self) -> bool:
        """ Load cached PR results if available. 
        Returns True if cache was loaded, False otherwise.
        """
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                self.recall = np.array(data['recall'])
                self.prec = np.array(data['prec'])
                self.ap50 = data['ap50']
                self.p = data['p']
                self.r = data['r']
                self.f1 = data['f1']
                self.ms_per_frame = data['ms_per_frame']
                self.fps = data['fps']
            logger.info(f"Loaded cached PR results from {self.cache_path}.")
            return True
        else:
            logger.info(f"No cache found at {self.cache_path}.")
            return False
        
    def run_inference(self, iou=0.5):
        """ Run YOLO inference on validation dataset and save results to cache. """
        model = YOLO(self.model_path)
        results = model.val(data=self.val_yaml, imgsz=self.training_imgsz, conf=self.conf, iou=iou, device=self.device, 
                            save_txt=False, save_conf=False, save_json=False, save=True, plots=True,
                            project=self.run_dir, name=f"{self.run_id}_{self.val_set}", exist_ok=True) #needed for saving  
        self.recall = np.linspace(0, 1, 1000)
        self.prec = results.box.prec_values[0]
        self.ap50 = results.box.map50
        self.p = results.box.p[0]
        self.r = results.box.r[0]
        self.f1 = results.box.f1[0]
        self.ms_per_frame = sum(results.speed.values())
        self.fps = 1000/self.ms_per_frame
        self.save_cache()

    def save_cache(self):
        """ Save PR results to cache file. """
        data = {
            'recall': self.recall.tolist(),
            'prec': self.prec.tolist(),
            'ap50': self.ap50,
            'p' : self.p,
            'r' : self.r,
            'f1' : self.f1,
            'ms_per_frame' : self.ms_per_frame,
            'fps' : self.fps
        }
        with open(self.cache_path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved PR results to cache at {self.cache_path}.")
        
    def ensure_data(self):
        """ Ensure PR data is available, either by loading cache or running inference. """
        if not self.load_cache():
            self.run_inference()

class PRPlotter:
    """Handles plotting of PR curves for multiple PRRun instances."""
    def __init__(self, runs, output_path=None):
        self.runs = runs
        self.output_path = output_path

    def plot(self):
        """Plot PR curves for all PRRun instances and save the figure."""
        plt.figure(figsize=(8, 6))
        for run in self.runs:
            print(fr"{run.plot_name} (AP$_{{50}}$: {run.ap50*100:.1f}) - (P: {run.p*100:.1f}) - (R: {run.r*100:.1f}) - (F1: {run.f1*100:.1f}) - (FPS: {run.fps*100:.1f}) ")
            print(fr"LaTex-ready: {run.plot_name} & {run.ap50*100:.1f} & {run.p*100:.1f} & {run.r*100:.1f} & {run.f1*100:.1f}")
            plt.plot(run.recall, run.prec, label=fr"{run.plot_name} (AP$_{{50}}$: {run.ap50:.3f})",
                     color=run.color, linestyle=run.linestyle)
        
        plt.xlabel('Recall', fontsize=14, color='black')
        plt.ylabel('Precision', fontsize=14, color='black')
        plt.title('Precision-Recall Curve', fontsize=16, color='black')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend(loc='lower left', frameon=True, title=r'Average Precision (AP$_{{50}}$)',  title_fontsize = 11, fontsize=11, framealpha=1)
        if self.output_path:
            plt.savefig(self.output_path)
            logger.info(f"Saved PR curve plot to {self.output_path}.")
        plt.show()