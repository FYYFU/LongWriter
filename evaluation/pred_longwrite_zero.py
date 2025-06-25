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
        def filter_thinking(response):
            # 使用正则提取 <answer>...</answer> 中的内容
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return None

        def format_prompt_with_template(prompt):

            base_format_zn = r"用户与助手之间的对话。用户提供一个写作/通用任务，助手完成它。助手首先在脑海中深入思考写作/回答过程，然后向用户提供最终的书面作品。助手应进行全面而深入的规划，确保写作/通用任务的每个方面都详细且结构合理。如果写作要求存在任何不确定性或歧义，助手应反思，向自己提出澄清性问题，并探索多种写作方式，以确保最终作品达到最高质量标准。由于写作是一个既富有创造性又需要结构性的任务，助手应从多个角度进行分析，考虑连贯性、清晰度、风格、语气、受众和目的，等等因素。此外，助手还应对作品进行审查和优化，以增强其表达效果。写作思考过程和最终的书面作品分别用 <think> </think> 和 <answer> </answer> 标签包裹，如下所示：<think>详细的写作规划和结构设计，可能包括头脑风暴、大纲制定、风格选择、受众适配、反思以及质量检查等等。</think> <answer>经过充分优化和润色的最终书面作品。</answer> <|用户|>: {question} <|助手|>:"
            base_format_en = r"A conversation between the user and the assistant. The user provides a writing/general task, and the assistant completes it. The assistant first deeply thinks through the writing/answering process in their mind before providing the final written work to the user. The assistant should engage in comprehensive and in-depth planning to ensure that every aspect of the writing/general task is detailed and well-structured. If there is any uncertainty or ambiguity in the writing request, the assistant should reflect, ask themselves clarifying questions, and explore multiple writing approaches to ensure the final output meets the highest quality standards. Since writing is both a creative and structured task, the assistant should analyze it from multiple perspectives, considering coherence, clarity, style, tone, audience, purpose, etc.. Additionally, the assistant should review and refine the work to enhance its expressiveness. The writing thought process and the final written work should be enclosed within <think> </think> and <answer> </answer> tags, respectively, as shown below: <think>A comprehensive strategy for writing that encompasses detailed planning and structural design—including brainstorming, outlining, style selection, audience adaptation, self-reflection, quality assurance, etc..</think> <answer>The final written work after thorough optimization and refinement.</answer>  <|user|>: {question} <|assistant|>:"
            base_format = base_format_zn if re.search(r'[\u4e00-\u9fff]', prompt) else base_format_en
            formatted_prompt = base_format.format(question=prompt)
            return formatted_prompt
        messages = [
            {"role": "user", "content": format_prompt_with_template(prompt)}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            stop_strings=["<|user|>", "<|endoftext|>", "</answer>"],
            tokenizer=tokenizer
        )[0]

        response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        dt['think_response_length'] = count_words(response)
        dt['think_response'] = response

        response = filter_thinking(response)
        dt["response_length"] = count_words(response) if response is not None else 0
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
    model = 'LongWriter-Zero-32B' # LongWriter-llama3.1-8b
    path =  "THU-KEG/LongWriter-Zero-32B" # THUDM/LongWriter-llama3.1-8b


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
