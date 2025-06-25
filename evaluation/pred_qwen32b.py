import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def get_pred(path, max_new_tokens, temperature, tokenizer, fout):

    model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto"
        ) 
    model = model.eval()


    for dt in data:
        prompt = dt['prompt']
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            tokenizer=tokenizer
        )[0]
        import ipdb
        response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        dt["response_length"] = count_words(response)
        dt["response"] = response
        fout.write(json.dumps(dt, ensure_ascii=False)+'\n')
        fout.flush()
        print(response)


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
    # model = 'LongWriter-Zero-32B' # LongWriter-llama3.1-8b
    # path =  "THU-KEG/LongWriter-Zero-32B" # THUDM/LongWriter-llama3.1-8b

    model = "Qwen2.5-32B"
    path = 'Qwen/Qwen2.5-32B'

    os.makedirs(f"models/{model}", exist_ok=True)
    fout = open(f"models/{model}/pred.jsonl", 'w', encoding='utf-8')

    max_new_tokens = 15500
    temperature = 0.6
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    world_size = torch.cuda.device_count()

    with open('longwrite_ruler_factual.jsonl', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    data_subsets = [data[i::world_size] for i in range(world_size)]
    get_pred(path, max_new_tokens, temperature, tokenizer, fout)
