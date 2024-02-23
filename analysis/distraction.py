import os
from transformers.models.llama import LlamaForCausalLM
from utils import load_tokenizer
from torch.utils.data import DataLoader
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def tokenize_one_item(tokenizer, item):
    text = item['text']
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return token_ids


def data_collator(examples: list):
    input_ids = torch.LongTensor([example["input_ids"] for example in examples])
    if "labels" in examples[0]:
        labels = torch.LongTensor([example["labels"] for example in examples])
    else:
        labels = input_ids
    batch_inputs = {"input_ids": input_ids, "labels": labels}
    return batch_inputs


class AttnAnalysisDataset(IterableDataset):
    def __init__(
            self,
            jsonl_path,
            tokenizer,
            doc1_len=None,
            doc2_len=None,
            end_token=None,
            doc1_jsonl_path=None,
    ):
        self.jsonl_path = jsonl_path
        self.subset_name = os.path.basename(jsonl_path).removesuffix(".jsonl")
        self.tokenizer = tokenizer

        self.doc1_len = doc1_len
        self.doc2_len = doc2_len
        if end_token == "eos":
            self.end_token = 2
        else:
            assert end_token == "nextline"
            self.end_token = 13
        print(f"load {self.subset_name} ")

        self.doc1_jsonl_path = doc1_jsonl_path
        if self.doc1_jsonl_path is not None:
            print(f"doc1 is from {self.doc1_jsonl_path} ")

    def get_doc1(self):
        with open(self.doc1_jsonl_path, "r") as fn:
            while True:
                doc1_token_ids = []
                while len(doc1_token_ids) < self.doc1_len:
                    cur_doc = json.loads(fn.readline())
                    doc1_token_ids.extend(
                        tokenize_one_item(self.tokenizer, cur_doc) + [self.end_token]
                    )
                yield doc1_token_ids

    def attn_score_check_iterator(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info.num_workers == 1

        if self.doc1_jsonl_path is not None:
            doc1_iter = self.get_doc1()
        else:
            doc1_iter = None

        with open(self.jsonl_path, "r") as fn:
            while True:
                if doc1_iter is not None:
                    doc1_token_ids = next(doc1_iter)
                else:
                    doc1_token_ids = []
                    while len(doc1_token_ids) < self.doc1_len:
                        cur_doc = json.loads(fn.readline())
                        doc1_token_ids.extend(
                            tokenize_one_item(self.tokenizer, cur_doc) + [self.end_token]
                        )
                while True:
                    doc2 = json.loads(fn.readline())
                    doc2_token_ids = tokenize_one_item(self.tokenizer, doc2) + [self.end_token]
                    if len(doc2_token_ids) >= self.doc2_len + 1:
                        break
                doc1_token_ids = doc1_token_ids[-self.doc1_len:]
                doc2_token_ids = doc2_token_ids[:self.doc2_len]

                seq = doc1_token_ids + doc2_token_ids
                yield {"input_ids": seq, "labels": seq}

    def __iter__(self):
        return self.attn_score_check_iterator()


def process_batch_attn(attn):
    batch_item_attention_matrix = []
    for item_idx in range(len(attn[0])):
        item_attention_matrix = []
        for layer_idx in range(len(attn)):
            item_attention_matrix.append(attn[layer_idx][item_idx].mean(dim=0))
        item_attention_matrix = torch.stack(item_attention_matrix)
        batch_item_attention_matrix.append(item_attention_matrix.float().cpu())
    return batch_item_attention_matrix


def d_item(nx, vx, ux, uo):
    return nx * vx + - 2 * (uo - ux) * nx * ux + nx * (uo ** 2 - ux ** 2)


def update_mean_and_std(u1, u2, d1, d2, n1, n2):
    u = (n1 / (n1 + n2)) * u1 + (n2 / (n1 + n2)) * u2
    v = (1 / (n1 + n2)) * (d_item(n1, d1 ** 2, u1, u) + d_item(n2, d2 ** 2, u2, u))
    return u, np.sqrt(v)


@torch.no_grad()
def main(model_name, end_token, corpus_type, doc1_len, doc1_jsonl_path=None):
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.eval()
    model.cuda()
    tokenizer = load_tokenizer()

    doc2_len = 256
    if corpus_type == "text":
        jsonl_path = "./data/SlimPajama-split/validation/RedPajamaWikipedia.jsonl"
    else:
        jsonl_path = "./data/SlimPajama-split/validation/RedPajamaGithub.jsonl"

    dev_dataset = AttnAnalysisDataset(
        jsonl_path=jsonl_path, tokenizer=tokenizer, doc1_len=doc1_len,
        doc2_len=doc2_len, end_token=end_token, doc1_jsonl_path=doc1_jsonl_path,
    )

    iter_batch_size = 6
    dataloader = DataLoader(
        dev_dataset,
        batch_size=iter_batch_size,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=False,
    )

    total = 4096
    cnt = 0
    batch_layer_pos_distractions_avg = []
    batch_layer_pos_distractions_std = []
    ppl_list = []
    acc_list = []
    for batch in tqdm(dataloader, total=total // iter_batch_size):
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=True
        )
        attentions = outputs.attentions

        examples_attention_scores = torch.stack(process_batch_attn(attentions))
        layers_examples_attention_scores = examples_attention_scores.transpose(0, 1)
        # layer, example, seq, seq
        distract_attn_prop = layers_examples_attention_scores[:, :, doc1_len:doc1_len + doc2_len, :doc1_len].sum(-1)
        distract_attn_avg = distract_attn_prop.mean(dim=1)  # layer, doc2_len
        distract_attn_std = distract_attn_prop.std(dim=1)
        # layer, doc2_len
        batch_layer_pos_distractions_avg.append(distract_attn_avg)
        batch_layer_pos_distractions_std.append(distract_attn_std)

        if cnt >= total:
            break

    sum_distract = torch.zeros_like(batch_layer_pos_distractions_avg[0])
    for bid in tqdm(range(len(batch_layer_pos_distractions_avg))):
        sum_distract += batch_layer_pos_distractions_avg[bid]
    avg_distract = sum_distract / len(batch_layer_pos_distractions_avg)

    latest_avg = batch_layer_pos_distractions_avg[0].tolist()
    latest_std = batch_layer_pos_distractions_std[0].tolist()
    latest_example_num = iter_batch_size
    for bid in tqdm(range(1, len(batch_layer_pos_distractions_avg))):
        for lid in range(24):
            for pid in range(len(batch_layer_pos_distractions_avg[bid][lid])):
                old_avg = latest_avg[lid][pid]
                old_std = latest_std[lid][pid]
                cur_avg = batch_layer_pos_distractions_avg[bid][lid][pid].item()
                cur_std = batch_layer_pos_distractions_std[bid][lid][pid].item()
                new_avg, new_std = update_mean_and_std(
                    old_avg, cur_avg, old_std, cur_std, latest_example_num, iter_batch_size
                )
                latest_avg[lid][pid] = new_avg
                latest_std[lid][pid] = new_std
        latest_example_num += iter_batch_size

    std_distract = torch.tensor(latest_std)

    distract_info = {
        "layer_pos_distractions_avg": avg_distract,
        "layer_pos_distractions_std": std_distract,
        "second_avg_ppl": np.mean(ppl_list),
    }

    model_name = model_name.split("/")[-1]
    print(model_name, end_token, corpus_type, doc1_len, doc1_jsonl_path)
    print(f"model: {model_name}, end token: {end_token}, {corpus_type}, "
          f"doc1_len={doc1_len}, doc1_jsonl_path={doc1_jsonl_path}")
    print(f"doc1+doc2 ppl: {np.mean(ppl_list)}")
    print(f"doc1+doc2 acc: {np.mean(acc_list)}")

    save_path = f"./analysis/outputs/{model_name}_{corpus_type}_doc1len{doc1_len}_{end_token}.pt"
    torch.save(distract_info, save_path)


def draw_distraction(end_token, corpus_type, doc1_len, weighted=True):
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'mathtext.default': 'regular'
    })

    def to_percent(y, position):
        return f"${100 * y:.0f}" + "\%$"

    models = [
        "MixChunk",
        "UniChunk",
        "BM25Chunk",
        "IntraDoc",
    ]
    legends = [
        r"\textsc{Mix}Chunk",
        r"\textsc{Uni}Chunk",
        r"\textsc{Bm25}Chunk",
        r"\textsc{Intra}Mask",
    ]

    model_distract_info = {
        model_name: torch.load(
            f"./analysis/outputs/{model_name}_{corpus_type}_doc1len{doc1_len}_{end_token}.pt"
        ) for model_name in models
    }

    plt.figure(figsize=(5, 4), dpi=180)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    for mid, model in enumerate(models):
        if weighted:
            t = torch.tensor([1 / rl for rl in range(24, 0, -1)])
            y = model_distract_info[model]["layer_pos_distractions_avg"] * t.unsqueeze(1)
            y = y.sum(dim=0) / t.sum()
        else:
            y = model_distract_info[model]["layer_pos_distractions_avg"].mean(dim=0)
        print(f"model: {model}, end token: {end_token}, {corpus_type}, doc1_len={doc1_len}")
        print(model_distract_info[model]["layer_pos_distractions_avg"].view(-1).mean())
        x = list(range(1, len(y) + 1))
        x = np.asarray(x)
        y = np.asarray(y)

        plt.plot(x, y, '', linewidth=1.0, label=f"{legends[mid]}")

    plt.plot(list(range(1, 256 + 1)), [doc1_len / (doc1_len + xi) for xi in range(1, 256 + 1)],
             linewidth=1.0, label=r"$y=|C_d|/(|C_d|+x)$", linestyle="--", color="grey")

    plt.xlabel('Position of The Second Document', fontsize=16)
    plt.ylabel('Average Distraction Proportion', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(visible=True)

    x_major_locator = MultipleLocator(32)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0.35, 1.0)
    plt.show(bbox_inches='tight')
