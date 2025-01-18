"""
Usage:
bash /usr/local/bin/pagecache-management.sh python3 profile_bandwidth.py
"""

import argparse
import numpy as np
import os
import time
import torch

from flexgen.utils import GB, MB, KB

import pandas as pd


def benchmark_func(func, number, repeat, warmup=3):
    for i in range(warmup):
        func()

    costs = [0]

    for i in range(repeat):
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(number):
            func()
        torch.cuda.synchronize()
        costs.append((time.time() - tic) / number)

    return costs

size_bw_gtoc = {"bw": [], "size": []}
size_bw_ctog = {"bw": [], "size": []}

def profile_bandwidth(path):
    s, h = 1, 1
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)

    links = [("cpu", "gpu"), ("gpu", "cpu")]

    for (dst, src) in links:
        min_unit = 1 
        mark = min_unit * 512 * 512 * 512
        for b in [ min_unit, min_unit * 2, min_unit * 4, min_unit * 8, min_unit * 16, min_unit * 32, min_unit * 64, min_unit * 128, min_unit * 256, min_unit * 512, min_unit * 512 * 2, min_unit * 512 * 4, min_unit * 512 * 8, min_unit * 512 * 16, min_unit * 512 * 32, min_unit * 512 * 64, 
                min_unit * 512 * 128, min_unit * 512 * 256, min_unit * 512 * 512, min_unit * 512 * 512 * 2, min_unit * 512 * 512 * 4, min_unit * 512 * 512 * 8, min_unit * 512 * 512 * 16, min_unit * 512 * 512 * 32, min_unit * 512 * 512 * 64, min_unit * 512 * 512 * 128, min_unit * 512 * 512 * 256,min_unit * 512 * 512 * 512, mark * 2, mark *4, mark*8, mark*16, mark*32, mark*64]:
            if dst == "cpu":
                dst_tensor = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
            elif dst == "gpu":
                dst_tensor = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
            elif dst == "disk":
                np.lib.format.open_memmap(path, mode="w+", shape=((b,s,h)), dtype=np.int8)
                dst_tensor = path

            if src == "cpu":
                src_tensor = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
            elif src == "gpu":
                src_tensor = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
            elif src == "disk":
                np.lib.format.open_memmap(path, mode="w+", shape=((b,s,h)), dtype=np.int8)
                src_tensor = path

            dst_indices = (slice(0, b), slice(0, s), slice(0, h))
            src_indices = (slice(0, b), slice(0, s), slice(0, h))

            def func():
                if isinstance(src_tensor, str):
                    src_tensor_ = torch.from_numpy(np.lib.format.open_memmap(src_tensor))
                else:
                    src_tensor_ = src_tensor
                if isinstance(dst_tensor, str):
                    dst_tensor_ = torch.from_numpy(np.lib.format.open_memmap(dst_tensor))
                else:
                    dst_tensor_ = dst_tensor
                dst_tensor_[dst_indices].copy_(src_tensor_[src_indices])

            size = np.prod([(x.stop - x.start) / (x.step or 1) for x in dst_indices])
            cost = np.mean(benchmark_func(func, number=20, repeat=5))
            bandwidth = size / cost / GB
            # if size / GB >= 1:
            #     print(f"size: {size/GB:6.2f} GB, {src}-to-{dst} bandwidth: {bandwidth:.6f} GB/s")
            # elif size / MB >= 1:
            #     print(f"size: {size/MB:6.2f} MB, {src}-to-{dst} bandwidth: {bandwidth:.6f} GB/s")
            # elif size / KB >= 1:
            #     print(f"size: {size/KB:6.2f} KB, {src}-to-{dst} bandwidth: {bandwidth:.6f} GB/s")
            # else:
            #     print(f"size: {size}B, {src}-to-{dst} bandwidth: {bandwidth:.6f} GB/s")
            if dst == "cpu":
                size_bw_gtoc["bw"].append(bandwidth)
                size_bw_gtoc["size"].append(size)
            else:
                size_bw_ctog["bw"].append(bandwidth)
                size_bw_ctog["size"].append(size)
        if dst == "cpu":
            print("gtoc: " + str(size_bw_gtoc["bw"]))
        else:
            print("ctog: " + str(size_bw_ctog["bw"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-path", type=str, default="~/flexgen_offload_dir/tmp.npy")
    parser.add_argument("--csv-name", type=str)
    args = parser.parse_args()

    profile_bandwidth(os.path.expanduser(args.offload_path))

    # df = pd.DataFrame.from_dict(size_bw)

    # Save the DataFrame to a CSV file
    # df.to_csv('dictionary_data.csv')
