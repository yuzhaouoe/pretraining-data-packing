# Analysing The Impact of Sequence Composition on Language Model Pre-Training


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
## Save Offline Dataset

### MixChunk
```bash
python ./save_offline_dataset.py --packing_strategy=mixchunk
```

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

Retrieval strategy: ```retriv_bm25.py``` and ```retrieval_packing.py```

Save BM25Chunk in one host:
```bash
python ./save_offline_dataset.py --packing_strategy=bm25chunk
```
or distribute files to different hosts and CPU cores.

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

