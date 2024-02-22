import os
import numpy as np
import time
import logging
from typing import List, Optional
import json
import mmap
from retrieval_packing import DefragmentConfig
from utils import load_offset, load_iter_order
from collections import Counter
import torch
from torch.utils.data import IterableDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_fragment_lens(chunk):
    chunk_size = len(chunk)
    cur_fragment_lens = []
    prev = 0
    for token_idx, token in enumerate(chunk):
        if token == 2:  # eos token
            cur_fragment_lens.append(token_idx - prev + 1)
            prev = token_idx + 1
    if prev != chunk_size:
        cur_fragment_lens.append(chunk_size - prev)

    return cur_fragment_lens, len(cur_fragment_lens)


def data_collator(examples: dict, max_num_fragments_in_chunk=65):
    input_ids = torch.LongTensor([example["input_ids"] for example in examples])
    if "labels" not in examples[0]:
        labels = input_ids
    else:
        labels = torch.LongTensor([example["labels"] for example in examples])
    batch_inputs = {"input_ids": input_ids, "labels": labels}
    if "fragment_lens" in examples[0]:
        fragment_lens = [
            torch.tensor(item["fragment_lens"] + (max_num_fragments_in_chunk - len(item["fragment_lens"])) * [-1])
            for item in examples
        ]
        batch_inputs["fragment_lens"] = torch.stack(fragment_lens)
        fragment_nums = torch.tensor([item["fragment_nums"] for item in examples], dtype=torch.int32)
        batch_inputs["fragment_nums"] = fragment_nums
    return batch_inputs


def get_worker_id_and_iter_indices(indices, worker_info):
    logger.debug(f"worker_info: {worker_info}")
    if worker_info is not None and worker_info.num_workers > 1:
        num_workers, worker_id = worker_info.num_workers, worker_info.id
        split_indices = [indices[idx * num_workers + worker_id] for idx in range(len(indices) // num_workers)]
        if worker_id < len(indices) % num_workers:
            rest_index = (len(indices) // num_workers) * num_workers + worker_id
            split_indices.append(indices[rest_index])
        split_indices = np.array(split_indices)
    else:
        split_indices = indices
        worker_id = 0
    logger.debug(f"worker {worker_id}, total {len(indices)}, get {len(split_indices)}")
    return worker_id, split_indices


class AsyncDataset(IterableDataset):
    def __init__(
            self,
            data_dir,
            name=None,
            dir_nums=1,
            mask_chunk=False,
            is_eval_test=False,
            chunk_size=None,
            host_nums_if_use_host_dispatch=None,
    ):
        self.max_retry_times = 100
        self.data_dir = data_dir
        self.name = name
        self.dir_nums = dir_nums
        self.cur_dir_idx = 1
        self.mask_chunk = mask_chunk
        self.is_eval_test = is_eval_test
        self.chunk_size = chunk_size

        self.host_nums_if_use_host_dispatch = host_nums_if_use_host_dispatch

    @staticmethod
    def get_batch_dir_and_path(data_dir, batch_idx):
        dir_idx = batch_idx // 1024
        batch_dir = os.path.join(data_dir, f"{dir_idx:05d}")
        batch_path = os.path.join(batch_dir, f"{batch_idx:08d}.pt")
        return batch_dir, batch_path

    def _wait_batch(self, path):
        for retry_times in range(self.max_retry_times):
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 10:
                if not os.path.exists(path):
                    time.sleep(60)  # wait to create the file
                else:
                    return True
            logger.info(f"request {path}, retry {retry_times} times.")
        raise TimeoutError(f"timeout when waiting batch file: {path}")

    def _load_batch(self, path):
        for retry_times in range(self.max_retry_times):
            try:
                batch_examples = torch.load(path)
                return batch_examples
            except Exception as err:
                time.sleep(10)
                continue
        raise TimeoutError(f"timeout when waiting batch file: {path}")

    def iterator(self, max_num_fragments_in_chunk=65):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"iterator, worker_info: {worker_info}")
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError

        example_idx = -1
        load_batch_idx = 0
        loaded_chunk_size = None

        while True:
            batch_dir, batch_path = self.get_batch_dir_and_path(
                os.path.join(self.data_dir, str(self.cur_dir_idx)),
                load_batch_idx
            )
            try:
                batch_data = torch.load(batch_path)
            except Exception as err:
                logger.info(f"{err}, does not exist, break")
                break

            for in_batch_example_idx, example in enumerate(batch_data):

                example_idx += 1

                if len([1 for xx in example["input_ids"] if xx == 2]) > max_num_fragments_in_chunk:
                    logger.info(f"pass {example_idx}: batch {load_batch_idx}-{in_batch_example_idx}")
                    continue

                if loaded_chunk_size is None:
                    loaded_chunk_size = len(example["input_ids"])
                    logger.info(f"async dataset: loaded_chunk_size = {loaded_chunk_size}")
                    if loaded_chunk_size != self.chunk_size:
                        logger.info(f"loaded chunk size != target chunk size: {loaded_chunk_size} != {self.chunk_size}")
                        assert loaded_chunk_size % self.chunk_size == 0

                if example_idx % num_workers == worker_id:

                    input_ids = example["input_ids"]
                    while len(input_ids) > 0:
                        cur_item = {"input_ids": input_ids[:self.chunk_size]}
                        if self.mask_chunk:
                            cur_fragment_lens, cur_fragment_nums = get_fragment_lens(cur_item["input_ids"])
                            cur_item["fragment_lens"] = cur_fragment_lens
                            cur_item["fragment_nums"] = cur_fragment_nums
                        input_ids = input_ids[self.chunk_size:]
                        yield cur_item

            load_batch_idx += 1

    def __iter__(self):
        return self.iterator()


class JsonlDataset:
    def __init__(
            self,
            name,
            jsonl_paths: List,
            is_train_data: bool = True,
            is_test_data: bool = False,
            iter_in_order: bool = False,
            iter_order: Optional[List[int]] = None,
            has_tokenized: bool = False,
    ):
        super(JsonlDataset, self).__init__()

        if len(jsonl_paths) == 0:
            raise ValueError

        logger.info(f"loading {name}")

        self.name = name
        self.jsonl_paths = jsonl_paths
        self.has_tokenized = has_tokenized

        self.all_offsets = [load_offset(p) for p in self.jsonl_paths]
        self.all_num_lines = [len(offsets) for offsets in self.all_offsets]
        self.lines_offset = np.cumsum(self.all_num_lines)  # [3,5,4,6] --> [3,8,12,18]

        if not is_train_data:
            assert iter_in_order

        if iter_order is None:
            if iter_in_order or (not is_train_data):
                self.iter_order = np.arange(len(self))
            else:
                rng = np.random.RandomState(123)
                iter_order = list(range(sum(self.all_num_lines)))
                rng.shuffle(iter_order)
                self.iter_order = iter_order
        else:
            self.iter_order = iter_order

    def _get_item(self, jsonl_idx, line_idx):
        with open(self.jsonl_paths[jsonl_idx], "rb") as fn:
            mfn = mmap.mmap(fn.fileno(), 0, access=mmap.ACCESS_READ)
            mfn.seek(self.all_offsets[jsonl_idx][line_idx])
            item = mfn.readline().decode("utf-8")
            item = json.loads(item.replace('\ufeff', ''))
        return item

    def __len__(self):
        return sum(self.all_num_lines)

    def __getitem__(self, idx):
        prev_line_offset = 0
        for jsonl_idx, line_offset in enumerate(self.lines_offset):
            if idx < line_offset:
                return self._get_item(jsonl_idx, idx - prev_line_offset)
            prev_line_offset = line_offset
        raise ValueError(f"index {idx} in {self.jsonl_paths}")


class CorpusDataset(IterableDataset):
    def __init__(
            self,
            jsonl_dataset: JsonlDataset,
            tokenizer,
            is_eval_data=False,
            chunk_size=2048,
            defragment_config: Optional[DefragmentConfig] = None,
            mask_chunk=False,
            defragmentation_fn=None,
            retriever=None,
    ):
        self.jsonl_dataset = jsonl_dataset
        self.indices = self.jsonl_dataset.iter_order

        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.is_eval_data = is_eval_data

        self.mask_chunk = mask_chunk

        self.remained_fragments = None
        self.defragment_config = defragment_config

        self.defragmentation_fn = defragmentation_fn
        self.retriever = retriever

    def get_token_ids(self, item):
        if self.jsonl_dataset.has_tokenized:
            token_ids = item["token"] + [self.tokenizer.eos_token_id]
        else:
            text = item['contents']
            token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            token_ids.append(self.tokenizer.eos_token_id)
        return token_ids

    def random_chunk_iterator(self):
        worker_id, indices = get_worker_id_and_iter_indices(self.indices, torch.utils.data.get_worker_info())

        cur_chunk = []
        cur_chunk_remain = self.chunk_size

        for idx in indices:
            item = self.jsonl_dataset[idx]
            token_ids = self.get_token_ids(item)
            num_tokens = len(token_ids)
            item_offset = 0

            while num_tokens:
                num_to_take = min(num_tokens, cur_chunk_remain)
                cur_chunk.extend(token_ids[item_offset:item_offset + num_to_take])
                item_offset += num_to_take
                cur_chunk_remain -= num_to_take
                num_tokens -= num_to_take

                if cur_chunk_remain == 0:
                    yield {"input_ids": cur_chunk}
                    cur_chunk = []
                    cur_chunk_remain = self.chunk_size

    def pad_iterator(self, pad_token_id=0, ignore_id=-100):
        worker_id, indices = get_worker_id_and_iter_indices(self.indices, torch.utils.data.get_worker_info())
        for idx in indices:
            item = self.jsonl_dataset[idx]
            token_ids = self.get_token_ids(item)
            while len(token_ids) >= self.chunk_size:
                yield {
                    "input_ids": token_ids[:self.chunk_size],
                    "labels": token_ids[:self.chunk_size],
                }
                token_ids = token_ids[self.chunk_size:]

            if len(token_ids) > 1:
                input_ids = token_ids + [pad_token_id] * (self.chunk_size - len(token_ids))
                labels = token_ids + [ignore_id] * (self.chunk_size - len(token_ids))
                del token_ids
                yield {"input_ids": input_ids, "labels": labels, }

    def defragment_chunk_iterator(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id, indices = get_worker_id_and_iter_indices(self.indices, worker_info)

        defragmentation_fn = self.defragmentation_fn
        retriever = self.retriever

        fragments_buffer = []
        self.remained_fragments = []

        for idx in indices:
            item = self.jsonl_dataset[idx]
            token_ids = self.get_token_ids(item)
            while len(token_ids) >= self.chunk_size:
                yield {"input_ids": token_ids[:self.chunk_size]}
                token_ids = token_ids[self.chunk_size:]

            if len(item["token"]) > 1:
                fragments_buffer.append(item)

            while len(fragments_buffer) >= self.defragment_config.fragments_buffer_size:
                fragments_buffer, chunk = defragmentation_fn(
                    retriever,
                    fragments_buffer,
                    self.chunk_size,
                    self.defragment_config,
                    self.tokenizer
                )
                if chunk is None:
                    raise ValueError
                yield {"input_ids": chunk}

        while len(fragments_buffer) > 1:
            fragments_buffer, chunk = defragmentation_fn(
                retriever,
                fragments_buffer,
                self.chunk_size,
                self.defragment_config,
                self.tokenizer
            )
            if chunk is None:
                break
            yield {"input_ids": chunk}

    def __iter__(self, max_num_fragments_in_chunk=65):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.is_eval_data:
            logger.info(f"{self.jsonl_dataset.name} get pad iterator")
            iterator = self.pad_iterator()
        elif self.defragment_config is None:
            logger.info(f"{self.jsonl_dataset.name} get random iterator")
            iterator = self.random_chunk_iterator()
        else:
            logger.info(f"{self.jsonl_dataset.name} get bm25chunk iterator")
            iterator = self.defragment_chunk_iterator()

        try:
            while True:
                yield next(iterator)
        except Exception as e:
            logger.info(f"stop jsonls interation")


class BlendCorpusDataset(IterableDataset):
    def __init__(
            self,
            corpus_list: List[CorpusDataset],
            iter_dataset_indices: List[int],
            corpus_weights=None,
    ):
        self.corpus_list = corpus_list
        self.corpus_weights = corpus_weights

        if len(corpus_list) > 1:
            corpus_iter_cnt = Counter(iter_dataset_indices)
            for cid in range(len(corpus_list)):
                subset_name = corpus_list[cid].jsonl_dataset.name
                logger.info(
                    f"{subset_name}\n"
                    f"set: {corpus_weights[subset_name]:.3f}\n"
                    f"iter_indices:  {corpus_iter_cnt[cid] / len(iter_dataset_indices):.3f}\n"
                )

        self.iter_dataset_indices = iter_dataset_indices
        self.worker_offset = 0

    def __iter__(self):
        worker_id, indices = get_worker_id_and_iter_indices(
            self.iter_dataset_indices,
            torch.utils.data.get_worker_info()
        )
        corpus_iterators = [iter(corpus) for corpus in self.corpus_list]

        for idx in indices:
            cur_item = {"corpus_idx": idx}
            try:
                cur_item_inputs = next(corpus_iterators[idx])
            except StopIteration:
                return
            cur_item.update(cur_item_inputs)
            yield cur_item
        logger.info(f"worker {worker_id}: blendable dataset iteration ended.")
