# PMOF

Official Repositiory for the PMOF Dataset.

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
├── yolo/
│   ├── README.md            #read README, full workflow for benchmarking explained here
│   ├── dataset_txt/         #txt-files with all images for a dataset configurations, referenced in yaml-files
│   ├── paper_plots/
│   ├── runs/
│   ├── video_samples/
│   ├── yaml/                #yaml-files references in training
│   ├── train_yolo.py        #used for training
│   ├── find_best_epoch.py   #find best epoch for trained models on one validation set
│   ├── yolo11n.pt
│   ├── yolo11n-obb.pt
│   ├── yolo11s.pt
│   ├── yolo11s-obb.pt
│   ├── dataprep_yoloformat.ipynb
│   ├── generateYOLO-TXT.ipynb
│   ├── train_yolo11.ipynb    #used for quick tests, not for "real" training
│   └── evaluate_predictions.ipynb
│
├── augmentation/            #generate augemented frames
│   ├── augmentation_pipeline.ipynb
│   ├── augutils.py
│   └── bg_augmentation.ipynb
│
├── bboxtype/                #plotting to compare aabb vs obb
│   ├── bbox_comparison.ipynb
│   ├── instances_default.json
│   └── rec7_001728.png
│
├── obbsizes/                # dir for csv-files to study bbox sizes
├── paper_plots/             # dir for paper-ready plots and Times-font
├── video_samples/           # dir for mp4-samples
│
├── dataprep.ipynb           # to rename frames & adapt annotations to new names (just for my postprocessing)
├── data_analysis.ipynb      # data analysis (except bounding box sizes)
├── obbsize_analysis.ipynb   # calculate csv-files (stored in obbsized) and analyse them
├── paper_ready_PMOF_samples.ipynb  # samples (train & val& bg), in paper-ready format
├── paper_ready_aug_samples.ipynb    # samples of aug CEPDOF, aug PMOF and HABBOF in paper-ready format
│
├── requirements.txt
└── README.md
```

## Getting started:

pip install -r requirements.txt

## Dataset Overview

Start with ```data_analysis.ipynb``` to get to know the dataset and look at some samples.

## Augmentation Pipeline

## Benchmarking with YOLO26

## Support
For support, write: stella.wermuth@hsbi.de