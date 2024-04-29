import os
import numpy as np
from tqdm import tqdm

split = "train"
n_shards = 10

def concatenate_files(input_paths, output_path):
    with open(output_path, 'wb') as output_file:
        for input_path in tqdm(input_paths, desc=f'Merging {os.path.basename(output_path)}'):
            with open(input_path, 'rb') as input_file:
                output_file.write(input_file.read())

# Assuming save_path is the directory where your binary shard files are stored
save_path = f"/home/mundus/abadawi696/slm_project/slm-60k/codes-train"
# Ensure the directory for merged files exists
merged_path = f"/home/mundus/abadawi696/slm_project/slm-60k/merged-codes"
os.makedirs(merged_path, exist_ok=True)

for ext, dtype in zip(["bin", "dur", "len"], [np.uint16, np.uint8, np.uint16]):
    shard_files = [f"{save_path}/{split}_{i}_{n_shards}.{ext}" for i in range(n_shards)]
    # Output file path for the concatenated binary file
    concatenated_path = f"{merged_path}/{split}.{ext}"
    # Concatenate the binary files
    concatenate_files(shard_files, concatenated_path)
    # Load concatenated file from the correct location
    concatenated = np.memmap(concatenated_path, dtype=dtype, mode='r')
    bins_len = [len(np.memmap(f"{save_path}/{split}_{i}_{n_shards}.{ext}", dtype=dtype, mode='r')) for i in range(n_shards)]
    bins_splits = [0] + np.cumsum(bins_len).tolist()
    
    print(f"Validating merged file: {os.path.basename(concatenated_path)}")
    for i in tqdm(range(n_shards), desc='Validating'):
        shard = np.memmap(f"{save_path}/{split}_{i}_{n_shards}.{ext}", dtype=dtype, mode='r')
        global_split = concatenated[bins_splits[i]:bins_splits[i + 1]]
        assert np.array_equal(shard, global_split)
    
    # Corrected to remove the correct shard files
    for i in range(n_shards):
        os.remove(os.path.join(save_path, f'{split}_{i}_{n_shards}.{ext}'))
