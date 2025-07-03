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
import LongLM.selfextend

window_size = 512
group_size = 2
enable_thinking = False
use_flash=False

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
    # device = torch.device(f'cuda:{rank}')
    model = AutoModelForCausalLM.from_pretrained(path, 
        config=config, 
        attn_implementation='eager', 
        trust_remote_code=True, 
        device_map='auto',
        torch_dtype=torch.bfloat16)
    model = model.eval()

    if window_size != 0 and group_size != 0:
        LongLM.selfextend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"


    for dt in data:
        prompt = dt['query']

        if enable_thinking:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            input = tokenizer(prompt, truncation=False, return_tensors="pt")
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
            input = tokenizer(prompt, truncation=False, return_tensors="pt")
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
        # print(content)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



def main_worker(rank, world_size, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(rank*world_size, (rank+1)*world_size))

    out_path = os.path.join(args.out_dir, f"pred_rank{rank}_w{window_size}_g{group_size}_t{enable_thinking}.jsonl")
    fout = open(out_path, 'w', encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 4. 让每个进程只处理 data 的一部分，避免重复
    with open(args.data_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    # 简单地按 rank 做切片：第 rank 条进程取 all_data[i] where i % world_size == rank
    subset = [dt for i, dt in enumerate(all_data) if i % world_size == rank]

    get_pred(rank, world_size, subset, args.model_path, tokenizer, fout)
    fout.close()


if __name__ == '__main__':
    seed_everything(42)

    model_name = 'Qwen3-32B'
    model_path = 'Qwen/Qwen3-32B'
    data_path = '/home/greenland-user/LongWriter/benchmark/WritingBench/benchmark_query/benchmark_all.jsonl'
    out_dir = f"WritingBench_outputs/models/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    world_size = 2

    args = argparse.Namespace(
        model_path=model_path,
        data_path=data_path,
        out_dir=out_dir
    )

    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

    merged_path = os.path.join(out_dir, f"pred_merged_w{window_size}_g{group_size}_t{enable_thinking}.jsonl")
    with open(merged_path, 'w', encoding='utf-8') as fout_merged:
        for rank in range(world_size):
            part = os.path.join(out_dir, f"pred_rank{rank}_w{window_size}_g{group_size}_t{enable_thinking}.jsonl")
            with open(part, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout_merged.write(line)
    print(f"Merged output into {merged_path}")



# if __name__ == '__main__':

#     seed_everything(42)

#     model = 'Qwen3-32B'
#     path = 'Qwen/Qwen3-32B'

#     data_path = '/home/greenland-user/LongWriter/benchmark/WritingBench/benchmark_query/benchmark_all.jsonl'

#     os.makedirs(f"WritingBench_outputs/models/{model}", exist_ok=True)
#     fout = open(f"WritingBench_outputs/models/{model}/pred_extend_w{window_size}_g{group_size}_t{enable_thinking}.jsonl", 'w', encoding='utf-8')

#     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#     world_size = torch.cuda.device_count()

#     with open(data_path, 'r') as f:
#         data = [json.loads(line) for line in f]


#     get_pred(0, world_size, data, path, tokenizer, fout)

