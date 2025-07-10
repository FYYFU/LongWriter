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
from LongProc.longproc.longproc_data import load_longproc_data

# ---------------------------------------------------------------------------
# Hyper‑parameters – tweak as you like
# ---------------------------------------------------------------------------

enable_thinking = True     # whether to use Qwen "thinking" channel
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
        f"pred_rank{rank}_t{args.task_name}_w{args.window_size}_g{args.group_size}_s{args.sub_size}.jsonl",
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
        if args.window_size and args.group_size:
            selfextend.apply(
                model,
                args.group_size,
                args.window_size,
                enable_flash_attention=use_flash,
                flash_attention_impl="flash_attn",
            )

        # ---- Slice the dataset for this rank -----------------------------------------
        # with open(args.data_path, "r", encoding="utf‑8") as f_data:
        #     all_data = [json.loads(line) for line in f_data]
        # subset = [dt for i, dt in enumerate(all_data) if i % world_size == rank]
        all_data, eval_func = load_longproc_data(f"{args.task_name}_{args.sub_size}", './LongProc/data')
        subset = [dt for i, dt in enumerate(all_data) if i % world_size == rank]

        # ---- Inference loop -----------------------------------------------------------
        for dt in tqdm(subset, desc=f"Rank {rank}"):
            user_query = dt["input_prompt"]

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_query}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6 if enable_thinking else 0.7,
                top_k=20,
                top_p=0.95 if enable_thinking else 0.8,
            )[0][len(inputs.input_ids[0]):].tolist()

            if enable_thinking:
                try:
                    sep_idx = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    sep_idx = 0
                thinking_content = tokenizer.decode(output_ids[:sep_idx], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[sep_idx:], skip_special_tokens=True).strip("\n")
                dt["think_length"] = count_words(thinking_content)
                dt["think_response"] = thinking_content
            else:
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                dt["think_length"] = 0
                dt["think_response"] = None

            dt["response_length"] = count_words(content)
            dt["response"] = content

            metrics = eval_func(content, dt)
            dt['metrics'] = metrics

            fout.write(json.dumps(dt, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[Rank {rank}] Finished → {out_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed_everything(42)

    model_name = "Qwen3-8B"
    model_path = "Qwen/Qwen3-8B"
    task_name = 'countdown'
    out_dir = f"LongProc_outputs/models/{model_name}"
    
    os.makedirs(out_dir, exist_ok=True)


    for sub_size in ['0.5k','2k', '8k']:
        # Multi‑processing settings
        world_size = 8   # number of processes
        size = 1         # number of GPUs per process (set 1 for one‑GPU‑per‑proc)
        shift = 0        # start GPU index

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", default=model_path)
        parser.add_argument("--task_name", default=task_name)
        parser.add_argument("--out_dir", default=out_dir)
        parser.add_argument('--sub_size', default=sub_size)
        parser.add_argument('--window_size', default=512)
        parser.add_argument('--group_size', default=2)
        cli_args = parser.parse_args([])  # empty list -> use defaults

        # Spawn workers
        mp.spawn(
            main_worker,
            args=(world_size, size, shift, cli_args),
            nprocs=world_size,
            join=True,
        )
        # Merge outputs from all ranks
        merged_path = os.path.join(out_dir, f"pred_merged_t{task_name}_w{cli_args.window_size}_g{cli_args.group_size}_s{sub_size}_new.jsonl")
        with open(merged_path, "w", encoding="utf‑8") as fout_merged:
            for rank in range(world_size):
                part = os.path.join(out_dir, f"pred_rank{rank}_t{task_name}_w{cli_args.window_size}_g{cli_args.group_size}_s{sub_size}_new.jsonl")
                with open(part, "r", encoding="utf‑8") as fin:
                    for line in fin:
                        fout_merged.write(line)
        print(f"Merged output → {merged_path}")
