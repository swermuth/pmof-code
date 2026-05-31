# Workflow:

1. 1_dataprep_yoloformat.ipynb
Create YOLO-Format labels for all PMOF images and create a basic train/val-split of PMOF dataset

2. 2_generateYOLO-TXT.ipynb
Create the dataset-txt's and yaml file for training.

3. 3_train_yolo.py
Train models

4. 4_find_best_epoch.py
To find the best epoch for each validation set.
´´´python find_best_epoch.py 20-n-1024-Ca18-P Hval 0´´´ (Model-name, validation set (Pval or Hval), device (0 or 1))

5. 5_evaluate_prediction.ipynb
Plot PR-Curve and training progress of metrics.