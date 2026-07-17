# merge_chunks.py
import os
import glob
import numpy as np

dataset = "MNIST"
n_qubits = 784
sigma = 7.1
total_samples = 1_000_000

ops_mmap = np.lib.format.open_memmap(f'ops_pool_{dataset}_q{n_qubits}_s{sigma}.npy', mode='w+', dtype=np.int32, shape=(total_samples, n_qubits))
expvals_mmap = np.lib.format.open_memmap(f'expvals_pool_{dataset}_q{n_qubits}_s{sigma}.npy', mode='w+', dtype=np.float32, shape=(total_samples,))

files = sorted(glob.glob(f'pool_{dataset}_q{n_qubits}_s{sigma}_chunk_*.npz'))

count = 0
for file in files:
    data = np.load(file)
    chunk_size = len(data['ops'])
    ops_mmap[count:count + chunk_size] = data['ops']
    expvals_mmap[count:count + chunk_size] = data['expvals']
    print(f"Processed chunk {count}")
    count += chunk_size

ops_mmap.flush()
expvals_mmap.flush()
print("All chunks merged successfully into master pool files.")