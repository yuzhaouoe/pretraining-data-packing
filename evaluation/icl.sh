#!/bin/bash -l

set -e
set -u

model="BM25Chunk"

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="agnews" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="yelp" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="amazon" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="dbpedia" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="sst2" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="tweet_hate" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done

for seed in {42..57}; do
  python ./fewshot.py \
    --model_path="yuzhaouoe/${model}" \
    --task="tweet_offensive" \
    --n_shot=48 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=4
done
