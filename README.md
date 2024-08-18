# Analysing The Impact of Sequence Composition on Language Model Pre-Training

---
This repository hosts the data and code of the paper: [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/)

* [Data Processing](#data-processing)
    * [Download SlimPajama](#download-slimpajama)
    * [Split Data Based on Meta-Information](#split-data-based-on-meta-information)
    * [Pre-Tokenize and Build Memmap Offset](#pre-tokenize-and-build-memmap-offset)
* [Save Offline Dataset](#save-offline-dataset)
    * [MixChunk](#mixchunk)
    * [UniChunk](#unichunk)
    * [BM25Chunk](#bm25chunk)
* [Evaluation](#evaluation)
* [Analysis](#analysis)
    * [Burstiness](#burstiness)
    * [Distraction Proportion](#distraction-proportion)
* [Citing](#citing)

## Data Processing

### Download SlimPajama

```bash
bash ./scripts/download_slimpajama.sh
```

### Split Data Based on Meta-Information

Decompress and split SlimPajama to subsets according to the meta-information of documents

```bash
export PYTHONPATH="./"
python ./preprocessing/split_to_subsets.py
```

### Pre-Tokenize and Build Memmap Offset

```bash
export PYTHONPATH="./"
python ./preprocessing/create_corpus.py
```

We split each subset to several files, defined by ```SUBSET_SPLIT_NUMS``` in ```project_config.py```. Each file is saved
in ```./data/SlimPajama-150B/[subset_name]/[[subset_name]_chunk[file_idx]_processed.jsonl]```.

## Save Offline Dataset

### MixChunk

```bash
python ./save_offline_dataset.py --packing_strategy=mixchunk
```

The result data is saved in ```./data/offline_datasets/mixchunk```.

### UniChunk

```bash
python ./save_offline_dataset.py --packing_strategy=unichunk
```

### BM25Chunk

BM25 retrieval is based on [Retriv](https://github.com/AmenRa/retriv)

Build index:

```bash
python build_bm25_index.py
```

It builds BM25 index for each file independently. Each index is saved
in ```./data/bm25index/collections/[subset_name]_[file_idx]```

Retrieval strategy: ```retriv_bm25.py``` and ```retrieval_packing.py```

Construct BM25Chunk in one host:

```bash
python ./save_offline_dataset.py --packing_strategy=bm25chunk
```

Or construct BM25Chunk for each file by running:

```bash
python ./save_offline_dataset.py \
  --packing_strategy=bm25chunk \
  --bm25chunk_onefile \
  --subset_name=RedPajamaWikipedia \
  --file_idx=0
```

This is an example to construct BM25Chunk for one file, and we can distribute these construction tasks to different CPU
cores and hosts. ```subset_name``` and its total number of split files are defined in ```project_config.py```.
After constructing BM25Chunk for all files, combine the data together by running:

```text
python ./save_offline_dataset.py --packing_strategy=bm25chunk --combine_data
```

## Evaluation

Use models from huggingface [link](https://huggingface.co/yuzhaouoe)

Download datasets

```
python ./scripts/download_eval_data.py
```

Reading comprehension and retrieval-augmented generation:

```bash
cd ./evaluation
bash ./mrc.sh
```

Knowledge memorisation:

```bash
cd ./evaluation
bash ./cbqa.sh
```

In-context learning:

```bash
cd ./evaluation
bash ./icl.sh
```


## Analysis

### Burstiness

Calculate the Zipf's coefficient of token frequency: ```./analysis/burstiness.py```

### Distraction Proportion

Visualise the distraction proportion: ```./analysis/distraction.py```

[//]: # (## Sequence Compositions of Existing LLMs)

[//]: # ()

[//]: # (| Model | Dataset | Type |)

[//]: # (|-------|---------|------|)

[//]: # (|       |         |      |)

[//]: # (|       |         |      |)

[//]: # (|       |         |      |)

## Citing

```
@inproceedings{zhao-etal-2024-analysing,
    title = "Analysing The Impact of Sequence Composition on Language Model Pre-Training",
    author = "Zhao, Yu  and
      Qu, Yuanbin  and
      Staniszewski, Konrad  and
      Tworkowski, Szymon  and
      Liu, Wei  and
      Mi{\l}o{\'s}, Piotr  and
      Wu, Yuxiang  and
      Minervini, Pasquale",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.427",
    pages = "7897--7912",
}
```

