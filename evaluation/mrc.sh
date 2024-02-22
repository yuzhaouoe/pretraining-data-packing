#!/bin/bash -l

set -e
set -u

model="BM25Chunk"

for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="hotpotqa" \
    --n_shot=2 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done


for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="squad" \
    --n_shot=4 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done


for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="nq_obqa" \
    --n_shot=16 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done


for seed in {42..46}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="tq_obqa" \
    --n_shot=16 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done
