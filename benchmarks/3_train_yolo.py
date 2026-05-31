from ultralytics import YOLO
import shutil
import os
import subprocess

print('Lets start!')

epochs = 20
model_size = 's'
img_size = 1024

data_base_dir = '/home/stella/computer_vision/pmof-code/benchmarks/yaml'
datasets = ["P-P", "PPa18-P", "C-P", "CCa18-P", "CP-P", "CCa18PPa18-P"] 

for dataset in datasets:    
    data_path = f"{data_base_dir}/{dataset}.yaml"

    name = f"{epochs}-{model_size}-{img_size}-{dataset}"
    output_dir = f"{os.getcwd()}/runs/obb/{name}/"

    model = YOLO(f"yolo26{model_size}-obb.yaml").load(f"yolo26{model_size}.pt")  # build from YAML and transfer weights
    
    results = model.train(data=data_path, epochs=epochs, name=name, imgsz=img_size, device=[0], save_period=1,
                          optimizer='SGD', lr0=0.001, momentum=0.9, weight_decay=0.0005,
                          hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, flipud=0, fliplr=0, 
                          translate=0, scale=0, mosaic=0, mixup=0, erasing=0, close_mosaic=epochs,
                          val=True, iou=0.5, augmentations=[])
    
    print('Training done!')
    shutil.copy(data_path, os.path.join(output_dir, f"{dataset}.yaml"))
    print('Copied Data-File')
    
    current_file = os.path.abspath(__file__)
    shutil.copy(current_file, os.path.join(output_dir, os.path.basename(current_file)))
    print('Copied Python-Script')

    print(f"Running Pval and Hval evaluations for {name}...")
    
    processes = []
    
    # Run PVAL on GPU 0
    processes.append(
        subprocess.Popen([
            "python", "find_best_epoch.py",
            name,
            "Pval",
            "0"
        ])
    )
    
    # Run HVAL on GPU 0
    processes.append(
        subprocess.Popen([
            "python", "find_best_epoch.py",
            name,
            "Hval",
            "0"
        ])
    )
    
    # Optional: wait for both to finish
    for p in processes:
        p.wait()
    
    print("Both validations finished!")
