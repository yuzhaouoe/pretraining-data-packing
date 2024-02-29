import argparse
import os

eval_args = argparse.ArgumentParser()
eval_args.add_argument("--model_path", type=str, required=True)
eval_args.add_argument("--task", type=str, required=True)
eval_args.add_argument("--n_shot", type=int, required=True)
eval_args.add_argument("--seed", type=int, required=True)
eval_args.add_argument("--device", type=int, required=False, default=0)
eval_args.add_argument("--batch_size", type=int, required=False, default=4)
eval_args.add_argument("--flash_attn_2", type=bool, default=True)
eval_args.add_argument("--max_length", type=int, default=8192)
eval_args.add_argument("--downsample", action="store_true")
eval_args.add_argument("--print", action="store_true")
eval_args = eval_args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{eval_args.device}"

from eval_utils import (
    eval_generation_em,
    eval_generation_em_answers,
    exact_match_score,
    exact_match_score_with_multiple_candidates
)

import torch
import json
import numpy as np

from tqdm import tqdm
from transformers.models.llama import LlamaTokenizer
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TASK_DATA_PATH = {
    "nq": {
        "train": "./eval_data/NQ-open.train-train.jsonl",
        "test": "./eval_data/NQ-open.test.jsonl",
    },
    "tq": {
        "train": "./eval_data/triviaqa.train-train.jsonl",
        "test": "./eval_data/triviaqa.test.jsonl",
    },
    "agnews": {
        "train": "./eval_data/ag_news_train.jsonl",
        "test": "./eval_data/ag_news_test.jsonl",
    },
    "nq_obqa": {
        "validation": "./eval_data/nq-dev-dense-results.json",
        "test": "./eval_data/nq_test.json",
    },
    "tq_obqa": {
        "train": "./eval_data/tq_obqa_train.json",
        "test": "./eval_data/tq_obqa_test.json",
    },
    "hotpotqa": {
        "train": "./eval_data/hotpot_train_v1.1.json",
        "validation": "./eval_data/hotpot_dev_fullwiki_v1.json",
    },
    "amazon": {
        "train": "./eval_data/amazon_train.jsonl",
        "test": "./eval_data/amazon_test.jsonl"
    },
    "dbpedia": {
        "train": "./eval_data/dbpedia_train.jsonl",
        "test": "./eval_data/dbpedia_test.jsonl"
    },
    "yelp": {
        "train": "./eval_data/yelp_train.jsonl",
        "test": "./eval_data/yelp_test.jsonl"
    },
    "sst2": {
        "train": "./eval_data/sst2_train.jsonl",
        "test": "./eval_data/sst2_test.jsonl",
    },
    "tweet_hate": {
        "train": "./eval_data/tweet_hate_train.jsonl",
        "test": "./eval_data/tweet_hate_test.jsonl"
    },
    "tweet_offensive": {
        "train": "./eval_data/tweet_offensive_train.jsonl",
        "test": "./eval_data/tweet_offensive_test.jsonl"
    },
    "squad": {
        "train": "./eval_data/squad_train.jsonl",
        "validation": "./eval_data/squad_validation.jsonl"
    },
}
for task in TASK_DATA_PATH:
    for split in TASK_DATA_PATH[task]:
        if not os.path.exists(TASK_DATA_PATH[task][split]):
            raise FileNotFoundError

PRINT = eval_args.print
DOWNSAMPLE = eval_args.downsample

if not os.path.exists("./outputs/logs"):
    with open("./outputs/logs", "w") as fn:
        pass


def load_json(path):
    return json.load(open(path, "r"))


def load_jsonl(path, max_line=None):
    with open(path, "r", encoding="utf-8") as fn:
        data = [json.loads(line) for line in fn.readlines()]
        if max_line is not None:
            rng = np.random.RandomState(666)
            data = rng.choice(data, min(max_line, len(data)), replace=False)
    return data


def normalise_pred(pred):
    return pred.strip().split("\n")[0].strip()


class PromptDataset(Dataset):
    def __init__(self, prompt_list, tokenizer):
        self.data = prompt_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dataloader(self, batch_size, max_length):
        def collate_fn(items):
            batch = [item["prompt"] for item in items]
            return self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length)

        return DataLoader(self, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)


@torch.no_grad()
def generate(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt_list, generation_kwargs,
             max_length, batch_size, task, n_shot, seed):
    predictions = []
    bar = tqdm(total=len(prompt_list), desc=f"{task}-{n_shot}-{seed}")
    prompt_dataset = PromptDataset(prompt_list, tokenizer)
    dataloader = prompt_dataset.get_dataloader(batch_size, max_length)
    for batch in dataloader:
        model_inputs = batch.to("cuda")
        generate_ids = model.generate(**model_inputs, **generation_kwargs)
        pred_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
        pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        predictions.extend(pred)
        bar.update(len(batch["input_ids"]))

        if PRINT:
            for cur_pred, cur_input in zip(pred, batch):
                print(cur_input, cur_pred)

    assert len(predictions) == len(prompt_list)
    return predictions


def get_cbqa_prompt(input_example, demonstrations):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Question: {item['question']} Answer: {item['answer'][0]}\n"
    prompt = prompt + f"Question: {input_example['question']} Answer:"
    return prompt


def cbqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"], max_line=4096)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    demonstrations = train_data[:n_shot]
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": get_cbqa_prompt(item, demonstrations)})
    all_pred_ans = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot,
                            seed)
    all_pred_ans = [pred.split("\n")[0] for pred in all_pred_ans]
    em_score = eval_generation_em(test_data, all_pred_ans) * 100
    return {"score": em_score}


def get_sampled_demonstrations(train_data, n_shot, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    return train_data[:n_shot]


def get_agnews_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Article: {item['text'].strip()} Category: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Article: {input_example['text'].strip()} Category:"
    return prompt


def agnews_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 4
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "world", 1: "sports", 2: "business", 3: "science"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_agnews_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_amazon_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()} Sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()} Sentiment:"
    return prompt


def amazon_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["amazon"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["amazon"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_amazon_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_dbpedia_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()}\nCategory: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()}\nCategory:"
    return prompt


def dbpedia_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    names = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
             "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork", ]
    generation_kwargs["max_new_tokens"] = 8
    train_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 10000, replace=False)
    label2str = {idx: name for idx, name in enumerate(names)}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_dbpedia_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_yelp_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def yelp_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["yelp"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["yelp"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_yelp_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_sst2_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def sst2_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "positive", 1: "negative"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_sst2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_tweet_hate_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Text: {item['text'].strip()}\nLabel: {label2str[item['label']]}\n"
    prompt = prompt + f"Text: {input_example['text'].strip()} Label:"
    return prompt


def tweet_hate_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 6
    train_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["test"])
    label2str = {0: "Non-hate", 1: "Hate"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_tweet_hate_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def tweet_offensive_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 6
    train_data = load_jsonl(TASK_DATA_PATH["tweet_offensive"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["tweet_offensive"]["test"])
    label2str = {0: "Non-hate", 1: "Hate"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_tweet_hate_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_obqa_demonstration(train_data, n_shot, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    demonstrations = train_data[:n_shot]
    return demonstrations


def process_ctx(ctx):
    # ctx = ctx.strip()
    # if ctx[-1] not in [".", "!", "?"]:
    #     ctx = ctx + "."
    # return ctx
    return ctx


def get_nq_obqa_prompt(input_example, demonstrations):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for ctx in demon["ctxs"][:2]:
            ctx_text = process_ctx(ctx['text'])
            context += f"{ctx['title']}. {ctx_text}\n"
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {demon['answers'][0]}\n\n"

    context = ""
    for ctx in input_example["ctxs"][:2]:
        ctx_text = process_ctx(ctx['text'])
        context += f"{ctx['title']}. {ctx_text}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def nq_obqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_json(TASK_DATA_PATH["nq_obqa"]["validation"])
    data = load_json(TASK_DATA_PATH["nq_obqa"]["test"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_nq_obqa_prompt(item, demonstrations)})
    all_pred_ans = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot,
                            seed)
    all_pred_ans = [pred.split("\n")[0] for pred in all_pred_ans]
    em_score = eval_generation_em_answers(data, all_pred_ans) * 100
    return {"score": em_score}


def get_hotpotqa_prompt(input_example, demonstrations):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for title, text in demon["context"]:
            context += f"{title}. {''.join(text)}\n"
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {demon['answer']}\n\n"

    context = ""
    for title, text in input_example["context"]:
        context += f"{title}. {''.join(text)}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def hotpotqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 20
    train_data = load_json(TASK_DATA_PATH["hotpotqa"]["train"])
    data = load_json(TASK_DATA_PATH["hotpotqa"]["validation"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_hotpotqa_prompt(item, demonstrations)})
    all_pred = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    all_pred = [pred.split("\n")[0] for pred in all_pred]
    correct_cnt = 0
    for pred, item in zip(all_pred, data):
        if exact_match_score(pred, item["answer"]):
            correct_cnt += 1
    em_score = correct_cnt / len(data) * 100
    return {"score": em_score}


def get_squad_prompt(input_example, demonstrations):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Passage: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: {item['answers']['text'][0].strip()}\n\n"
    prompt = prompt + f"Passage: {input_example['context'].strip()}\nQuestion: {input_example['question'].strip()}\nAnswer:"
    return prompt


def squad_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_jsonl(TASK_DATA_PATH["squad"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["squad"]["validation"])
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    correct_cnt = 0
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": get_squad_prompt(item, demonstrations)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    for pred, item in zip(predictions, test_data):
        pred = normalise_pred(pred)
        targets = item['answers']['text']
        if exact_match_score_with_multiple_candidates(pred, targets):
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return {"score": acc}


def get_tq_obqa_prompt(input_example, demonstrations):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for ctx in demon["ctxs"][:2]:
            context += f"{ctx['title']}. {ctx['text']}\n"
        if "target" in demon.keys():
            cur_answer = demon["target"]
        else:
            cur_answer = demon["answers"][0]
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {cur_answer}\n\n"

    context = ""
    for ctx in input_example["ctxs"][:2]:
        context += f"{ctx['title']}. {ctx['text']}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def tq_obqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_json(TASK_DATA_PATH["tq_obqa"]["train"])
    data = load_json(TASK_DATA_PATH["tq_obqa"]["test"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_tq_obqa_prompt(item, demonstrations)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    predictions = [pred.split("\n")[0] for pred in predictions]
    em_score = eval_generation_em_answers(data, predictions) * 100
    return {"score": em_score}


def metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=None, normalise_target=None):
    correct_cnt = 0
    for pred, item in zip(predictions, test_data):
        if normalise_pred is not None:
            pred = normalise_pred(pred)
        target = label2str[get_label_call(item)]
        if normalise_target is not None:
            target = normalise_target(target)
        if target == pred:
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return acc


eval_callables = {
    "nq": cbqa_evaluation,
    "tq": cbqa_evaluation,
    "wq": cbqa_evaluation,
    "sst2": sst2_evaluation,
    "agnews": agnews_evaluation,
    "nq_obqa": nq_obqa_evaluation,
    "hotpotqa": hotpotqa_evaluation,
    "amazon": amazon_evaluation,
    "dbpedia": dbpedia_evaluation,
    "yelp": yelp_evaluation,
    "tweet_hate": tweet_hate_evaluation,
    "tweet_offensive": tweet_offensive_evaluation,
    "squad": squad_evaluation,
    "tq_obqa": tq_obqa_evaluation,
}


def main():
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "min_length": 1,
        "eos_token_id": 13,
        "use_cache": True,
    }
    results = {
        "task": eval_args.task,
        "n_shot": eval_args.n_shot,
        "seed": eval_args.seed,
        "model": eval_args.model_path
    }
    tokenizer = LlamaTokenizer.from_pretrained(eval_args.model_path, padding_side='left', truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if eval_args.flash_attn_2:
        try:
            model = LlamaForCausalLM.from_pretrained(
                eval_args.model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2",
            )
        except Exception as err:
            logger.error(err)
            logger.info("cannot use FlashAttention2")
            eval_args.batch_size = 1
            model = LlamaForCausalLM.from_pretrained(eval_args.model_path, torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained(eval_args.model_path, torch_dtype=torch.float16)

    model.eval()
    model = model.cuda()

    evaluation = eval_callables[eval_args.task]

    score = evaluation(model, tokenizer, generation_kwargs, eval_args.task, eval_args.n_shot,
                       eval_args.seed, eval_args.max_length - 5, eval_args.batch_size)
    results.update(score)
    results = json.dumps(results)
    logger.info(results)
    with open("./outputs/logs", "a") as fn:
        fn.write(results + "\n")


if __name__ == '__main__':
    main()
