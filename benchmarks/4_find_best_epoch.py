import sys
from pathlib import Path

# Add project root to sys.path (so imports like src.data work)
project_root = Path().cwd().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualization import PRRun, PRPlotter
from src import logger
import csv

def get_best_epoch(file_path):
    max_ap50 = -1.0
    best_epoch = None

    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            current_ap50 = float(row['AP50'])
            if current_ap50 > max_ap50:
                max_ap50 = current_ap50
                best_epoch = row['epoch']
                
    return best_epoch, max_ap50


def find_best_epoch(val_set, run_name, device="0", conf=0.3):
    val_yaml = f'./yaml/{val_set}.yaml'

    outfile_name = run_name.split('/')[-1] # remove potential "blur/" prefix for better file naming
    output_path = Path(f"./runs/obb/{run_name}/ap50_per_epoch_{outfile_name}_{val_set}_{int(conf*100)}.txt") #0_paper_runs
    if output_path.exists():
        logger.error(f"Best epoch already identified, checkout {str(output_path)}")
        best_epoch, max_ap50 = get_best_epoch(output_path)
        logger.info(f"Best epoch: {best_epoch}, Max AP50: {max_ap50:.1f}%")
        return
        #sys.exit(1)

    weights_dir = Path(f'./runs/obb/{run_name}/weights/') #0_paper_runs
    if not weights_dir.exists():
        logger.error(f"Error: weights directory not found: {weights_dir}")
        sys.exit(1)
        
    epoch_files = sorted(
        [p for p in weights_dir.glob('*.pt') if 'epoch' in p.stem or 'last' in p.stem or 'best' in p.stem],
        key=lambda p: int(p.stem.replace('epoch', '').replace('last', '9999').replace('best', '9998'))
    )
        
    runs = [
        PRRun(str(weight_path), run_name, "black", '--', val_yaml, conf, device)
        for weight_path in epoch_files
    ]
    
    results = []
    
    best_ap50 = 0
    best_epoch = None
    for i, run in enumerate(runs):
        run.ensure_data()
        ap50 = run.ap50*100
        results.append((i, ap50))
        
        if ap50 > best_ap50:
            best_ap50 = ap50
            best_epoch = i
    
    # Save AP50 per epoch
    with open(output_path, "w") as f:
        f.write("epoch,AP50\n")
        for epoch, ap50 in results:
            f.write(f"{epoch},{ap50:.4f}\n")
    
    print(f"Saved AP50 results to {output_path}")
    print(f"Best epoch: {best_epoch} with AP50 = {best_ap50:.2f}")

model = "m"
run_list = ["20-m-1024-C-P", "20-m-1024-CCa18-P", "20-m-1024-P-P", "20-m-1024-PPa18-P", "20-m-1024-CP-P", "20-m-1024-CCa18PPa18-P"]

val_sets = ["Pval", "Hval"]

for val_set in val_sets:
    print(f"Evaluating for validation set: {val_set}")
    for run_name in run_list:
        print(f"Processing run: {run_name}")
        find_best_epoch(val_set, run_name)