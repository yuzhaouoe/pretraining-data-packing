import os
import pickle
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from packing_dataset import JsonlDataset, CorpusDataset, AsyncDataset
from utils import load_tokenizer
from retrieval_packing import DefragmentConfig
from tqdm import tqdm
import math
from retriv_bm25 import SparseRetriever
from retrieval_packing import bm25_defragment_retriv, bm25_defragment_retriv_simplified
from project_config import *


def save_data_collator(examples):
    examples = [{"input_ids": example["input_ids"]} for example in examples]
    return examples


def load_dataset_with_retriever(
        file_path, chunk_size: int, defragment_config: DefragmentConfig, index_path
):
    retriever = SparseRetriever.load(index_path)
    doc_num = retriever.doc_count
    iter_order = list(range(doc_num))
    rng = np.random.RandomState(666)
    rng.shuffle(iter_order)
    jsonl_dataset = JsonlDataset(
        name="slimpajama", jsonl_paths=[file_path], is_train_data=True,
        iter_in_order=False, has_tokenized=True, iter_order=iter_order
    )
    tokenizer = load_tokenizer()
    defragmentation_fn = bm25_defragment_retriv_simplified
    corpus_dataset = CorpusDataset(
        jsonl_dataset, tokenizer=tokenizer, chunk_size=chunk_size,
        is_eval_data=False, defragment_config=defragment_config, mask_chunk=False,
        defragmentation_fn=defragmentation_fn, retriever=retriever
    )

    return corpus_dataset


def create_one_file_dataset(
        subset_name, file_idx, index_path, data_name="debug", fragments_buffer_size=4096,
        multihop=True, shuffle_chains=False, over_fragmented_length=16, chunk_size=8192,
        save_batch_size=256, num_workers=16, target_tokens=1024 ** 3, suffix=None
):
    jsonl_dir = os.path.join("./data/SlimPajama-150B", subset_name)
    file_path = os.path.join(jsonl_dir, f"{subset_name}_chunk{file_idx}_processed.jsonl")

    defragment_config = DefragmentConfig(
        defragmentation_method="bm25", fragments_buffer_size=fragments_buffer_size,
        shuffle_chains=shuffle_chains, over_fragmented_length=over_fragmented_length, multihop=multihop
    )
    save_name = f"{data_name}_{chunk_size}_{fragments_buffer_size}_s{shuffle_chains}_mh{multihop}"
    if suffix is not None:
        save_name += f"_{suffix}"

    outer_data_dir = os.path.join("./data/saved_datasets", save_name)
    if os.path.exists(outer_data_dir):
        pass
    else:
        os.mkdir(outer_data_dir)

    subset_dir = os.path.join(outer_data_dir, subset_name)
    if not os.path.exists(subset_dir):
        os.mkdir(subset_dir)
    split_subset_dir = os.path.join(subset_dir, str(file_idx))
    if not os.path.exists(split_subset_dir):
        os.mkdir(split_subset_dir)

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    train_dataset = load_dataset_with_retriever(file_path, chunk_size, defragment_config, index_path)

    iter_batch_size = 1
    dataloader = DataLoader(
        train_dataset,
        batch_size=iter_batch_size,
        collate_fn=save_data_collator,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=8 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    target_batch_num = math.ceil(target_tokens / (save_batch_size * chunk_size))
    num_chunks = target_tokens / chunk_size
    tqdm_bar = tqdm(total=num_chunks)

    batch_cache = []
    batch_idx = 0
    start_time = time.perf_counter()
    for iter_batch_idx, iter_batch_data in enumerate(dataloader):
        batch_cache.extend(iter_batch_data)
        tqdm_bar.update(len(iter_batch_data))
        if len(batch_cache) >= save_batch_size:
            batch_dir, batch_path = get_batch_dir_and_path(split_subset_dir, batch_idx)
            if not os.path.exists(batch_dir):
                os.mkdir(batch_dir)
            torch.save(batch_cache[:save_batch_size], batch_path)
            batch_cache = batch_cache[save_batch_size:]
            batch_idx += 1
            print(f"yield {batch_idx}th batch, {time.perf_counter() - start_time:.5f}")
            start_time = time.perf_counter()
            if batch_idx == target_batch_num:
                break

    print(f"saved {target_tokens / 1024 ** 3:.3f}B tokens")


def estimate_total_batch_num(cur_all_split_dirs):
    cnt = 0
    for split_dir in cur_all_split_dirs:
        # get all paths of saved chunks, each chunk contains 1024 batches at most
        chunk_1024_ids = os.listdir(split_dir)
        chunk_1024_dirs = [os.path.join(split_dir, chunk_1024_id) for chunk_1024_id in chunk_1024_ids]
        for chunk_idx, chunk_dir in enumerate(chunk_1024_dirs):
            # get all batch-paths in a chunk
            load_batch_ids = os.listdir(chunk_dir)
            cnt += len(load_batch_ids)
    return cnt


def collect_data(load_from, dump_to):
    """

    original:
    ./saved_datasets/[bm25_name]/[subset_name]/[split_id]/[chunk_1024_id]/[batch_id.pt]

    combine and dump to:
    ./saved_datasets/[combined_name]/[subset_name]/1/[chunk_1024_id]/[batch_id.pt]
    with only one split, split_id = 1
    """

    # load_from = os.path.join(OFFLINE_DATA_DIR, bm25_data_name)
    # dump_to = os.path.join(OFFLINE_DATA_DIR, combined_name)
    if not os.path.exists(dump_to):
        os.mkdir(dump_to)

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    for subset in SUBSET_NAMES:

        # locate into one subset
        cur_load_from = os.path.join(load_from, subset)
        # get all split dirs
        cur_all_split_dirs = os.listdir(cur_load_from)
        cur_all_split_dirs = [os.path.join(cur_load_from, split_dir) for split_dir in cur_all_split_dirs]

        # combined subset save dir: XXXX/subset/1/
        subset_dump_to = os.path.join(dump_to, subset)
        if not os.path.exists(subset_dump_to):
            os.mkdir(subset_dump_to)
        subset_dump_to = os.path.join(subset_dump_to, "1")  # combined to "...... /{subset}/1/"
        if not os.path.exists(subset_dump_to):
            os.mkdir(subset_dump_to)

        # re-count batch-idx for combined subset
        cur_saved_batch_idx = 0

        bar = tqdm(total=estimate_total_batch_num(cur_all_split_dirs))
        for split_dir in cur_all_split_dirs:
            # get all chunk paths, each chunk contains 1024 batches at most
            chunk_1024_ids = os.listdir(split_dir)
            chunk_1024_dirs = [os.path.join(split_dir, chunk_1024_id) for chunk_1024_id in chunk_1024_ids]
            for chunk_idx, chunk_dir in enumerate(chunk_1024_dirs):
                # get all batch-paths in a chunk
                load_batch_ids = os.listdir(chunk_dir)
                load_batch_paths = [os.path.join(chunk_dir, batch_id) for batch_id in load_batch_ids]

                # load every batch, and save to the target dir
                for load_batch_path in load_batch_paths:
                    # loaded_batch = torch.load(load_batch_path)
                    save_batch_dir, save_batch_path = get_batch_dir_and_path(subset_dump_to, cur_saved_batch_idx)
                    if not os.path.exists(save_batch_dir):
                        os.mkdir(save_batch_dir)
                    try:
                        x = torch.load(load_batch_path)
                        if len(x) != 256:
                            print(f"incomplete batch size: {len(x)}")
                            continue
                    except Exception as err:
                        print(f"cannot load: {load_batch_path}")
                        continue

                    # torch.save(loaded_batch, save_batch_path)
                    shutil.copyfile(load_batch_path, save_batch_path)

                    cur_saved_batch_idx += 1

                    bar.update(1)
                    bar.set_description(f"{os.path.basename(split_dir)}-"
                                        f"{chunk_idx:05d}-"
                                        f"{os.path.basename(load_batch_path)}"
                                        f", saved {cur_saved_batch_idx} batches")
