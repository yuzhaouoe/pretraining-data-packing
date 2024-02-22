import json
from tqdm import tqdm
import zstandard as zstd
import logging
import os
from project_config import *

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def read_zstd_file(file_path):
    with open(file_path, 'rb') as compressed:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed) as reader:
            file_content = reader.read()
            lines = file_content.decode('utf-8').splitlines()
            for line in lines:
                json_data = json.loads(line)
                yield json_data


def extract_subset_data(split_name):
    assert split_name in ["test", "validation"]
    logger.info(f"process {split_name} set.")

    zstd_files = []
    compressed_split_dir = os.path.join("./data/SlimPajama-compressed", split_name)
    for root, dirs, files in os.walk(compressed_split_dir):
        zstd_files.extend([os.path.join(root, file) for file in files])
    logger.info(f"detect {len(zstd_files)} compressed files from {compressed_split_dir}.")

    subset_dir = f"./data/SlimPajama-split/{split_name}"
    if not os.path.exists(subset_dir):
        os.mkdir(subset_dir)

    subset_file_path = dict()
    for subset_name in SUBSET_NAMES:
        subset_file_path[subset_name] = os.path.join(subset_dir, f"{subset_name}.jsonl")
        fn = open(subset_file_path[subset_name], "w")
        fn.close()

    num_items_cnt = {subset_name: 0 for subset_name in SUBSET_NAMES}
    for zstd_file in tqdm(zstd_files, desc=f"SlimPajama {split_name}, extract subsets"):
        for item in read_zstd_file(zstd_file):
            text = item["text"]
            subset_name = item["meta"]["redpajama_set_name"]
            with open(subset_file_path[subset_name], "a") as fout:
                fout.write(json.dumps({"text": text}) + "\n")
            num_items_cnt[subset_name] += 1

    logger.info(f"{split_name} num docs: {num_items_cnt}")


def extract_subset_data_train():
    dctx = zstd.ZstdDecompressor()

    for subset_name in SUBSET_NAMES:
        subset_dump_dir = f"./data/SlimPajama-split/train/{subset_name}"
        if not os.path.exists(subset_dump_dir):
            os.mkdir(subset_dump_dir)

    split_path = f"./data/SlimPajama-compressed/train"
    split_chunks = [os.path.join(split_path, vci) for vci in os.listdir(split_path)]
    for cdir in split_chunks:
        logger.info(f"processing {cdir}")
        files = os.listdir(cdir)
        for file in tqdm(files, ncols=50):
            file_path = os.path.join(cdir, file)
            with open(file_path, 'rb') as ifh, open("./tmpfile", 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
            with open("./tmpfile", "rb") as fn:
                for line in fn.readlines():
                    try:
                        item = json.loads(line)
                    except Exception as e:
                        logger.error(f"error {file_path}\n{e}")
                        exit(-1)
                    item["meta"] = item["meta"]["redpajama_set_name"]
                    assert item["meta"] in SUBSET_NAMES

                    add_to_subset = f"./data/SlimPajama-split/train/{item['meta']}/{item['meta']}_chunk{cdir[-1]}"
                    with open(add_to_subset, "a") as fout:
                        fout.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    if not os.path.exists("./data/SlimPajama-split"):
        os.mkdir("./data/SlimPajama-split")
    extract_subset_data("validation")
    extract_subset_data("test")
    extract_subset_data_train()
