import pandas as pd
import glob
import re
import shutil

val_data = pd.read_csv("validation.csv")
files = glob.glob("./drivaerml_rans_processed_with_connectivity_fixed_with_pt_data_full/run_*.zarr")

for file in files:
    match = int(re.search(r'drivaerml_rans_processed_with_connectivity_fixed_with_pt_data_full/run_(\d+).zarr', file).group(1))
    if match in val_data["run_idx"].values:
        shutil.move(file, "./drivaerml_rans_processed_with_connectivity_fixed_with_pt_data_full/val/")
    else:
        shutil.move(file, "./drivaerml_rans_processed_with_connectivity_fixed_with_pt_data_full/train/")
