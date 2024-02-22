import os
import pickle
import time
import logging
from tqdm import tqdm
import mmap
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from multiprocessing import Pool, RLock
from project_config import *

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pickle(path):
    if os.path.exists(path):
        retry = 0
        while retry < 3:
            try:
                data = pickle.load(open(path, "rb"))
            except Exception:
                retry += 1
                continue
            return data
        logger.error(f"cannot open {path}")
    else:
        raise FileNotFoundError(f"{path} is not found")


def rank0_dump_pickle(data, path):
    if get_local_rank() in [0, -1]:
        pickle.dump(data, open(path, "wb"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", -1))


def get_offset_path(jsonl_path):
    return os.path.join(os.path.dirname(jsonl_path), os.path.basename(jsonl_path) + ".offset.pkl")


def load_offset(jsonl_path):
    offset = load_pickle(get_offset_path(jsonl_path))
    return offset


def load_iter_order(jsonl_name, data_dir):
    iter_order_path = os.path.join(data_dir, f"{jsonl_name}_iter_order.pkl")
    iter_order = load_pickle(iter_order_path)
    return iter_order


def load_tokenizer(path="./llama"):
    return LlamaTokenizer.from_pretrained(path)
