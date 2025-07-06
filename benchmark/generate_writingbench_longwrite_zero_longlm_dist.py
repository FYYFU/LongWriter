#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import re
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyper‑parameters – tweak as you like
# ---------------------------------------------------------------------------
window_size = 256          # sliding window length for LongLM.selfextend
group_size = 8             # number of heads to patch together
use_flash = False          # use flash‑attention v2 when available

# ---------------------------------------------------------------------------
# Small helpers that do not rely on torch / transformers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(chinese_characters) + len(english_words)


def seed_everything(seed: int):
    """Seed RNGs (local import of torch so we don’t pull it in globally)."""
    import torch  # local
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Worker logic – runs in every spawned process
# ---------------------------------------------------------------------------

def main_worker(rank: int, world_size: int, size: int, shift: int, args):
    """Each process owns `size` GPU(s) starting from `rank*size+shift`."""

    # 1️⃣ Restrict visible GPUs _before_ importing torch / transformers
    devices = ",".join(str(i + shift) for i in range(rank * size, (rank + 1) * size))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    # 2️⃣ Heavy imports after env var is set
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import LongLM.selfextend as selfextend

    # Pin to the single (index‑0) visible card
    torch.cuda.set_device(0)

    # Log – optional
    print(f"[Rank {rank}] Using CUDA_VISIBLE_DEVICES={devices}")

    # Create per‑rank output file
    out_path = os.path.join(
        args.out_dir,
        f"pred_rank{rank}_w{window_size}_g{group_size}.jsonl",
    )
    with open(out_path, "w", encoding="utf‑8") as fout:

        # ---- Load model & tokenizer ----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        config = AutoConfig.from_pretrained(args.model_path)
        config.rope_scaling = {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",          # single‑GPU per process
            torch_dtype=torch.bfloat16,
        ).eval()

        # Patch LongLM extension if requested
        if window_size and group_size:
            selfextend.apply(
                model,
                group_size,
                window_size,
                enable_flash_attention=use_flash,
                flash_attention_impl="flash_attn",
            )

        # ---- Slice the dataset for this rank -----------------------------------------
        with open(args.data_path, "r", encoding="utf‑8") as f_data:
            all_data = [json.loads(line) for line in f_data]
        subset = [dt for i, dt in enumerate(all_data) if i % world_size == rank]

        # ---- Inference loop -----------------------------------------------------------
        for dt in tqdm(subset, desc=f"Rank {rank}"):
            user_query = dt["query"]

            # Build prompt (thinking or not)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_query}],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=15500,
                temperature=0.6,
                do_sample=True,
                stop_strings=["<|user|>", "<|endoftext|>", "</answer>"],
                tokenizer=tokenizer
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            def get_response(response):
                # 使用正则提取 <answer>...</answer> 中的内容
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    return None
            def get_think(response):
                # 使用正则提取 <answer>...</answer> 中的内容
                match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    return None

            try:
                think = get_think(content)
                dt["think_length"] = count_words(think)
                dt["think_response"] = think
            except:
                dt["think_length"] = 0
                dt["think_response"] = None
            
            try:
                response = get_response(content)
                dt["response_length"] = count_words(response)
                dt["response"] = response
            except:
                dt["response_length"] = count_words(content)
                dt["response"] = content

            fout.write(json.dumps(dt, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[Rank {rank}] Finished → {out_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed_everything(42)

    model_name = "LongWriter-Zero-32B"
    model_path = "THU-KEG/LongWriter-Zero-32B"
    data_path = "/home/greenland-user/LongWriter/benchmark/WritingBench/benchmark_query/benchmark_all.jsonl"
    out_dir = f"WritingBench_outputs/models/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Multi‑processing settings
    world_size = 8   # number of processes
    size = 1         # number of GPUs per process (set 1 for one‑GPU‑per‑proc)
    shift = 0        # start GPU index

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=model_path)
    parser.add_argument("--data_path", default=data_path)
    parser.add_argument("--out_dir", default=out_dir)
    cli_args = parser.parse_args([])  # empty list -> use defaults

    # Spawn workers
    mp.spawn(
        main_worker,
        args=(world_size, size, shift, cli_args),
        nprocs=world_size,
        join=True,
    )

    # Merge outputs from all ranks
    merged_path = os.path.join(out_dir, f"pred_merged_w{window_size}_g{group_size}.jsonl")
    with open(merged_path, "w", encoding="utf‑8") as fout_merged:
        for rank in range(world_size):
            part = os.path.join(out_dir, f"pred_rank{rank}_w{window_size}_g{group_size}.jsonl")
            with open(part, "r", encoding="utf‑8") as fin:
                for line in fin:
                    fout_merged.write(line)
    print(f"Merged output → {merged_path}")
