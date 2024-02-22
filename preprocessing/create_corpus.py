import json
import pickle
import time
import logging
from tqdm import tqdm
from multiprocessing import Pool, RLock, freeze_support
import mmap
import os
import math
from transformers import LlamaTokenizer

from project_config import *
from utils import get_offset_path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def build_offset(jsonl_path):
    offsets = []
    tqdm_bar = tqdm()
    with open(jsonl_path, "rb") as fn:
        mfn = mmap.mmap(fn.fileno(), 0, access=mmap.ACCESS_READ)
        mfn.seek(0)
        cur = 0

        start_time = time.perf_counter()
        while True:
            line = mfn.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
            tqdm_bar.update(1)
    logger.info(f"process {jsonl_path}, {len(offsets)} lines, {time.perf_counter() - start_time:.3f} secs")
    offset_path = get_offset_path(jsonl_path)
    pickle.dump(offsets, open(offset_path, 'wb'))


def sample_and_pretokenize_slimpajama_train(total_tokens):
    tokenize_tokens = {subset_name: SUBSET_WEIGHTS[subset_name] * total_tokens for subset_name in SUBSET_NAMES}

    chunk_ids = list(range(1, 11))
    all_args = []
    pid = 0
    for subset_name, subset_tokens in tokenize_tokens.items():
        jsonl_dir = f"./data/SlimPajama-split/train/{subset_name}"
        for cid in chunk_ids:
            cur_kwargs = {
                "input_path": os.path.join(jsonl_dir, f"{subset_name}_chunk{cid}"),
                "target_tokens": subset_tokens // len(chunk_ids),
                "output_path": os.path.join(jsonl_dir, f"{subset_name}_chunk{cid}_tokenized_sampled"),
                "desc": f"{pid}-{subset_name}_chunk{cid}",
                "pid": pid
            }
            pid += 1

            all_args.append(cur_kwargs)
    pro_num = min(len(all_args), 128)

    freeze_support()
    pool = Pool(pro_num, initializer=tqdm.set_lock, initargs=(RLock(),))
    pool.map(sample_and_pretokenize_process, all_args)
    time.sleep(5)
    pool.close()
    pool.join()


def sample_and_pretokenize_process(kwargs):
    tokenizer = LlamaTokenizer.from_pretrained("yuzhaouoe/BM25Chunk")
    target_tokens = kwargs["target_tokens"]
    total_tokens = 0
    # [warn]: need to shuffle lines in "input_path"
    with open(kwargs["input_path"], "r") as fin, open(kwargs["output_path"], "w") as fout:
        tqdm_bar = tqdm(position=kwargs["pid"], ncols=100)
        line_id = 0
        while True:
            try:
                line = fin.readline()
                item = json.loads(line)
            except Exception as e:
                logger.info(e)
                if line == "":
                    logger.info(f"{kwargs['input_path']} ended")
                else:
                    logger.error(f"json parse error, {kwargs['input_path']}, line {line_id}")
                break

            token_ids = tokenizer.encode(item["text"], add_special_tokens=False, truncation=False)
            fout.write(json.dumps({"contents": item["text"], "token": token_ids}) + "\n")
            total_tokens += len(token_ids)

            desc = f"{kwargs['desc']}, tokenized {total_tokens / 1024 ** 3:.3f}B tokens"
            tqdm_bar.set_description(desc)

            if total_tokens >= target_tokens:
                break

            tqdm_bar.update(1)
            line_id += 1


def iter_documents_from_sampled_subset(subset_name):
    # [warn] need to shuffle
    files = [f"./data/SlimPajama-split/train/{subset_name}/{subset_name}_chunk{cid}_tokenized_sampled"
             for cid in range(1, 11)]
    for file in files:
        with open(file, "r") as fn:
            for line in fn.readlines():
                yield json.loads(line)


def group_documents(target_tokens):
    for subset_name in SUBSET_NAMES:
        subset_dump_dir = f"./data/SlimPajama-150B/{subset_name}"
        if not os.path.exists(subset_dump_dir):
            os.mkdir(subset_dump_dir)

        subset_target_token = target_tokens * SUBSET_WEIGHTS[subset_name]
        each_split_tokens = math.ceil(subset_target_token / SUBSET_SPLIT_NUMS[subset_name])
        each_split_tokens = [each_split_tokens * (idx + 1) for idx in range(SUBSET_SPLIT_NUMS[subset_name])]

        jsonl_paths = [f"./data/SlimPajama-split/train/{subset_name}/{subset_name}_chunk{cid}_tokenized_sampled"
                       for cid in range(1, 11)]

        cur_split = 0
        cur_internal_idx = 0
        split_dump_path = os.path.join(subset_dump_dir, f"{subset_name}_chunk{cur_split}_processed.jsonl")
        with open(split_dump_path, "w") as fn:
            pass

        cur_get_tokens = 0
        tqdm_bar = tqdm(total=subset_target_token, desc=f"{subset_name}")

        # for batch_lines in dataloader: # todo
        for line in iter_documents_from_sampled_subset(subset_name):
            batch_lines = [line]

            cur_batch_num_tokens = sum([len(item["token"]) for item in batch_lines])
            write_lines = []
            if cur_get_tokens + cur_batch_num_tokens > subset_target_token:
                for item in batch_lines:
                    item_num_tokens = len(item["token"])
                    cur_get_tokens += item_num_tokens
                    write_lines.append(item)
                    tqdm_bar.update(item_num_tokens)
                    if cur_get_tokens > subset_target_token:
                        break
            else:
                cur_get_tokens += cur_batch_num_tokens
                write_lines = batch_lines
                tqdm_bar.update(cur_batch_num_tokens)

            with open(split_dump_path, "a") as fout:
                for item in write_lines:
                    item["internal_idx"] = cur_internal_idx
                    cur_internal_idx += 1
                    fout.write(json.dumps(item) + "\n")

            if cur_get_tokens > each_split_tokens[cur_split]:
                print(f"{subset_name} split-{cur_split} completed")

                cur_split += 1
                cur_internal_idx = 0
                split_dump_path = os.path.join(subset_dump_dir, f"{subset_name}_chunk{cur_split}_processed.jsonl")
                with open(split_dump_path, "w") as fn:
                    pass

            tqdm_bar.set_description(
                f"{subset_name} {cur_get_tokens / 1024 ** 3:.3f}/{subset_target_token / 1024 ** 3:.3f}B tokens."
            )

            if cur_get_tokens > subset_target_token:
                break

        print(f"{subset_name}: {cur_get_tokens / 1024 ** 3:.3f}B tokens")


def main():
    total_tokens = 152 * 1024 ** 3
    sample_and_pretokenize_slimpajama_train(total_tokens)
    group_documents(total_tokens)


if __name__ == '__main__':
    main()
