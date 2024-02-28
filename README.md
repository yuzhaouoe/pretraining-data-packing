# Analysing The Impact of Sequence Composition on Language Model Pre-Training

---
This repository hosts the data and code of the paper: [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991)

![](https://github.com/yuzhaouoe/pretraining-data-packing/pics/random.png)

![](https://github.com/yuzhaouoe/pretraining-data-packing/pics/random.png)

## Data Processing

Download SlimPajama
```bash
bash ./scripts/download_slimpajama.sh
```

Decompress and split SlimPajama to subsets according to the meta-information of documents
```bash
export PYTHONPATH="./"
python ./preprocessing/split_to_subsets.py
```

Sample documents, pre-tokenize and build offset
```bash
export PYTHONPATH="./"
python ./preprocessing/create_corpus.py
```

We split each subset to several files, defined by ```SUBSET_SPLIT_NUMS``` in ```project_config.py```. Each file is saved in ```./data/SlimPajama-150B/[subset_name]/[[subset_name]_chunk[file_idx]_processed.jsonl]```.

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
It builds BM25 index for each file independently. Each index is saved in ```./data/bm25index/collections/[subset_name]_[file_idx]``` 

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
This is an example to construct BM25Chunk for one file, and we can distribute these construction tasks to different CPU cores and hosts. ```subset_name``` and its total number of split files are defined in ```project_config.py```. 
After constructing BM25Chunk for all files, combine the data together by running:
```text
python ./save_offline_dataset.py --packing_strategy=bm25chunk --combine_data
```



## Evaluation

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

