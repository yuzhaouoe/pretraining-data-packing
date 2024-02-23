import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from scipy.stats import zipf
from scipy.optimize import minimize_scalar


def zipfian_fit(data):
    def negative_log_likelihood(s, data):
        return -np.sum(np.log(zipf.pmf(data, s)))

    result = minimize_scalar(negative_log_likelihood, bounds=(1.01, 10),
                             args=(data,), method='bounded')

    s_estimated = result.x
    return s_estimated


def get_batch_dir_and_path(data_dir, batch_idx):
    dir_idx = batch_idx // 1024
    batch_dir = os.path.join(data_dir, f"{dir_idx:05d}")
    batch_path = os.path.join(batch_dir, f"{batch_idx:08d}.pt")
    return batch_dir, batch_path


def get_chunks_iterator(dataset_dir):
    batch_idx = 0
    while True:
        batch_dir, batch_path = get_batch_dir_and_path(dataset_dir, batch_idx)
        if not os.path.exists(batch_path):
            break
        batch_data = torch.load(batch_path)

        for item in batch_data:
            yield item["input_ids"]

        batch_idx += 1


def debug_zipfian(dataset_dir, target_chunk_size=8192, mask_chunk=False):
    max_num_chunks = int(2 * 1024 ** 3 / 8192)  # sample 2B tokens to estimate the coefficient
    tqdm_bar = tqdm(total=max_num_chunks)
    estimates = []
    pass_n = 0
    for chunk in get_chunks_iterator(dataset_dir):
        if pass_n < 1 * 1024 ** 3 / 8192:
            pass_n += 1
            continue
        if mask_chunk is False:
            if target_chunk_size < len(chunk):
                for start_idx in range(0, len(chunk), target_chunk_size):
                    cnt = Counter(chunk[start_idx: start_idx + target_chunk_size])
                    cnt = sorted(cnt.values(), reverse=True)
                    estimates.append(zipfian_fit(cnt))
            else:
                cnt = Counter(chunk)
                cnt = sorted(cnt.values(), reverse=True)
                estimates.append(zipfian_fit(cnt))
        else:
            if target_chunk_size < len(chunk):
                for start_idx in range(0, len(chunk), target_chunk_size):
                    cur_split_chunk = chunk[start_idx: start_idx + target_chunk_size]
                    while len(cur_split_chunk) > 0:
                        try:
                            index = cur_split_chunk.index(2)
                            split = cur_split_chunk[:index + 1]
                            cur_split_chunk = cur_split_chunk[index + 1:]
                        except ValueError:
                            split = cur_split_chunk
                            cur_split_chunk = []
                        if len(split) > 0:
                            cnt = Counter(split)
                            cnt = sorted(cnt.values(), reverse=True)
                            estimates.append(zipfian_fit(cnt))
            else:
                while len(chunk) > 0:
                    try:
                        index = chunk.index(2)
                        split = chunk[:index + 1]
                        chunk = chunk[index + 1:]
                    except ValueError:
                        split = chunk
                        chunk = []
                    if len(split) > 0:
                        cnt = Counter(split)
                        cnt = sorted(cnt.values(), reverse=True)
                        estimates.append(zipfian_fit(cnt))
        tqdm_bar.update(1)
        tqdm_bar.set_description(f"avg = {np.mean(estimates)}")
        if tqdm_bar.n > max_num_chunks:
            break

    print(f"\n{dataset_dir}\n"
          f"avg: {np.mean(estimates)}\n"
          f"std: {np.std(estimates)}")
