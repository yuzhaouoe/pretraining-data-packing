#!/bin/bash -l

set -e
set -u

model="BM25Chunk"

for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="nq" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="tq" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done
