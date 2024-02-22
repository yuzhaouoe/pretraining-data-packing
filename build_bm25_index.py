from retriv_bm25 import SparseRetriever
from multiprocessing import Pool
import os
import time
from project_config import *


def build_retriv_index(kwargs):
    os.environ["RETRIV_BASE_PATH"] = "./data/bm25index"
    if not os.path.exists("./data/bm25index"):
        os.mkdir("./data/bm25index")
    index_name = kwargs["index_name"]
    if os.path.exists(os.path.join("./data/bm25index", f"/collections/{index_name}")):
        print(f"{index_name} exists")
        return

    file_path = kwargs["file_path"]
    print(f"build index: {index_name}")
    sr = SparseRetriever(
        index_name=index_name,
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
    )
    print(f"load file from {file_path}\n"
          f"save index to {os.environ.get('RETRIV_BASE_PATH')}, {index_name}")
    sr.index([file_path])
    print(f"saved {index_name}")


def multiporcess_index():
    all_args = []
    for subset_name, split_nums in SUBSET_SPLIT_NUMS.items():
        jsonl_dir = os.path.join("./data/SlimPajama-150B", subset_name)
        for file_idx in range(split_nums):
            file_path = os.path.join(jsonl_dir, f"{subset_name}_chunk{file_idx}_processed.jsonl")
            all_args.append({
                "file_path": file_path,
                "index_name": f"{subset_name}_{file_idx}"
            })
    print(f"{len(all_args)} files, 5 processes")
    pool = Pool(5)
    pool.map(build_retriv_index, all_args)
    time.sleep(5)
    pool.close()
    pool.join()


if __name__ == '__main__':
    multiporcess_index()
