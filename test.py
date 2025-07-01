from vllm import LLM, SamplingParams

import os
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"   # 条件 2

llm = LLM(
    model="AQuarterMile/WritingBench-Critic-Model-Qwen-7B",
    max_model_len=32768,                            # 条件 1
)

prompt = "你好 " * 40000                            # 原始≈40 k token
params = SamplingParams(
    max_tokens=256,                                 # 预留生成长度
    truncate_prompt_tokens=32512,                   # 条件 3: 32768-256
)

out = llm.generate([prompt], params)
print(out[0].outputs[0].text)
