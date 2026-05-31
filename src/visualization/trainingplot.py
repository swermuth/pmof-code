# visualization/trainingplot.py
# This module provides functionality to plot training metrics such as loss and accuracy over epochs.
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src import logger

metric_mapping = {
    'epoch': 'Epoch',
    'train/box_loss': 'Train Box Loss',
    'train/cls_loss': 'Train Cls. Loss',
    'val/box_loss': 'Val Box Loss',
    'val/cls_loss': 'Val Cls. Loss',
    'metrics/precision(B)':'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': r'AP$_{50}$',
    'metrics/mAP50-95(B)': 'mAP50-95',
}


class TrainingRun:
    """Represents one training run, loads data from a CSV file and provides methods to plot metrics."""

    def __init__(self, model_path, name, color, linestyle='-'):
        self.model_path = Path(model_path)
        self.name = name
        self.color = color
        self.linestyle = linestyle

        self.run_dir = self._resolve_run_dir()
        self.results_csv = self.run_dir/'results.csv'

        if not self.results_csv.exists():
            logger.error(f"Results CSV not found at {self.results_csv}")
            raise FileNotFoundError(f"Results CSV not found at {self.results_csv}")
        
        self.data = pd.read_csv(self.results_csv)

        self.epoch = len(self.data)
        logger.info(f"Loaded training run '{self.name}' with {self.epoch} epochs.")

    def _resolve_run_dir(self):
        if self.model_path.is_file() and self.model_path.suffix == ".pt":
            parts = self.model_path.parts
            if "weights" in parts:
                idx = parts.index("weights")
                return Path(*parts[:idx])
        elif self.model_path.is_dir():
            return self.model_path
        else:
            logger.error(f"Invalid model_path: {self.model_path}")
            return None
        
    def plot_metric(self, metric_name, ax, val_metric=None):
        """ Generic method to plot a specified metric over epochs. """
        if metric_name not in self.data.columns:
            logger.error(f"Metric '{metric_name}' not found in results.")
            return
        
        ax.plot(self.data['epoch'], self.data[metric_name], label=self.name, color=self.color, linestyle=self.linestyle)
        if val_metric and val_metric in self.data.columns:
            ax.plot(self.data['epoch'], self.data[val_metric], label=f"{self.name} - val", color=self.color, linestyle='--')
        #ax.set_title(f"{metric_name} over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_mapping.get(metric_name, metric_name))
        ax.grid()

    # Optional shortcut methods for common metrics:
    def plot_box_loss(self, ax): self.plot_metric('train/box_loss', ax, 'val/box_loss')
    def plot_cls_loss(self, ax): self.plot_metric('train/cls_loss', ax, 'val/cls_loss')
    def plot_precision(self, ax): self.plot_metric('metrics/precision(B)', ax)
    def plot_recall(self, ax): self.plot_metric('metrics/recall(B)', ax)
    def plot_map50(self, ax): self.plot_metric('metrics/mAP50(B)', ax)
    def plot_map5095(self, ax): self.plot_metric('metrics/mAP50-95(B)', ax)

class TrainingPlotter:
    """Handles plotting of multiple TrainingRun instances with consistent layout."""
    def __init__(self, runs, title=None, output_path=None):
        self.runs = runs
        self.title = title or "Training Progress Comparison"
        self.output_path = output_path
        self.fontsize_lg = 8

    def plot(self):
        """ Generate and display/save all metric plots for multiple training runs."""

        fig = plt.figure(figsize=(6.78,3), dpi=200)
        gs = fig.add_gridspec(1, 3, wspace=0.5, hspace=0.5)

        #axBox = fig.add_subplot(gs[0, 0])
        #axCls = fig.add_subplot(gs[0, 1])
        axPrec = fig.add_subplot(gs[0, 0])
        axRec = fig.add_subplot(gs[0, 1])
        axMap50 = fig.add_subplot(gs[0, 2])


        for run in self.runs:
            #run.plot_box_loss(axBox)
            #run.plot_cls_loss(axCls)
            run.plot_precision(axPrec)
            run.plot_recall(axRec)
            run.plot_map50(axMap50)
     
        #fig.suptitle(self.title)
        fig.tight_layout()
        for ax in [axPrec, axRec, axMap50]: #axBox, axCls
            ax.legend(fontsize=self.fontsize_lg, loc='lower left')
            ax.grid()
            ax.set_xlim(1, 20)
            #ax.set_ylim(0.65, 1)

        if self.output_path:
                plt.savefig(self.output_path,  bbox_inches="tight", dpi=200, pad_inches=0)
                logger.info(f"Plot saved to {self.output_path}")
        plt.show()