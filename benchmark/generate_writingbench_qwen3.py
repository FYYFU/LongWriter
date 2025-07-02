import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np
import random
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
import torch.distributed as dist
import torch.multiprocessing as mp


window_size = 0
group_size = 0
enable_thinking = True

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    total_count = chinese_char_count + english_word_count
    return total_count


def get_pred(rank, world_size, data, path, tokenizer, fout):

    config = AutoConfig.from_pretrained(path)
    config.rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768
    }
    device = torch.device(f'cuda:{rank}')
    model = AutoModelForCausalLM.from_pretrained(path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()

    for dt in data:
        prompt = dt['query']

        if enable_thinking:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            output = model.generate(
                **input,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95
            )
            output_ids = output[0][len(input.input_ids[0]):].tolist() 
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            dt['think_length'] = count_words(thinking_content)
            dt['think_response'] = thinking_content
        else:

            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            output = model.generate(
                **input,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.8
            )
            output_ids = output[0][len(input.input_ids[0]):].tolist() 
            dt['think_length'] = 0
            dt['think_response'] = None
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        
        dt["response_length"] = count_words(content)
        dt["response"] = content
        fout.write(json.dumps(dt, ensure_ascii=False)+'\n')
        fout.flush()
        print(content)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    seed_everything(42)

    model = 'Qwen3-32B'
    path = 'Qwen/Qwen3-32B'

    data_path = '/home/greenland-user/LongWriter/benchmark/WritingBench/benchmark_query/benchmark_all.jsonl'

    os.makedirs(f"WritingBench_outputs/models/{model}", exist_ok=True)
    fout = open(f"WritingBench_outputs/models/{model}/pred_extend_w{window_size}_g{group_size}_t{enable_thinking}.jsonl", 'w', encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    world_size = torch.cuda.device_count()

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    data_subsets = [data[i::world_size] for i in range(world_size)]
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], path, tokenizer, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
