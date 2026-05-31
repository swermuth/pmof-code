# PMOF

Official Repositiory for the PMOF Dataset.

## Getting started:

pip install -r requirements.txt

## Dataset Overview

Start with ```data_analysis.ipynb``` to get to know the dataset and look at some samples.

## Augmentation Pipeline

Start with ```augmentation/augmentation_pipeline.ipynb``` to work with our obb-augmentation pipeline.

## Benchmarks

Checkout ```benchmarks/README.md``` for instructions to follow the full workflow from data prep to training and evaluation for our benchmarks.

## Support
For support, write: stella.wermuth@hsbi.de

## Code Structure
```
./
│
├── src/          
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── annotation_utils.py
│   │   ├── dataset_utils.py
│   │   ├── file_utils.py
│   │   └── obbsize_analysis.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── augmentation_samples.py
│   │   ├── color_utils.py
│   │   ├── frame_visualizer.py
│   │   ├── image_utils.py
│   │   ├── prcurve.py
│   │   ├── trainingplot.py
│   │   ├── video_utils.py
│   │   └── viz_config.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── databasedir_config.py
│       └── logging_config.py
│
├── benchmarks/
│   ├── README.md            #read README, full workflow for benchmarking explained here
│   ├── dataset_txt/         #txt-files with all images for a dataset configurations, referenced in yaml-files
│   ├── runs/
│   ├── yaml/                #yaml-files references in training
│   ├── 1_dataprep_yoloformat.ipynb
│   ├── 2_generateYOLO-TXT.ipynb
│   ├── 3_train_yolo.py        #used for training
│   ├── 4_find_best_epoch.py   #find best epoch for trained models on one validation set
│   └── 5_evaluate_predictions.ipynb
│
├── augmentation/            #generate augemented frames
│   ├── augmentation_pipeline.ipynb
│   └── augutils.py
│
├── paper_plots/             # dir for paper-ready plots and Times-font
├── video_samples/           # dir to save videos

│
├── data_analysis.ipynb      # data analysis (except bounding box sizes)
├── paper_ready_PMOF_samples.ipynb  # samples (train & val& bg), in paper-ready format
│
├── requirements.txt
└── README.md
```