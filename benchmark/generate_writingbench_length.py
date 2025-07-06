import re
import json
import ipdb
import jsonlines
from collections import defaultdict


def count_words(text: str) -> int:
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(chinese_characters) + len(english_words)


data_path = '/home/greenland-user/LongWriter/benchmark/WritingBench_outputs/models/LongWriter-Zero-32B/pred_merged_w0_g0.jsonl'

with open(data_path, 'r') as f:
    data = [json.loads(l) for l in f]

final_length = []
final_think_length = []
final_input_length = []

for item in data:
    import ipdb
    input_length = count_words(item['query'])
    final_input_length.append(input_length)

    try:
        think_length = item['think_length']
        final_think_length.append(think_length)
    except:
        continue
    
    try:
        response_length = item['response_length']
        final_length.append(response_length)
    except:
        continue


avg_think_length = round(sum(final_think_length) / len(final_think_length), 2) if len(final_think_length) != 0 else None
avg_response_length = round(sum(final_length) / len(final_length), 2) if len(final_length) != 0 else None
avg_input_length = round(sum(final_input_length) / len(final_input_length), 2) if len(final_input_length) != 0 else None

print(f'input: {avg_input_length}')
print(f'think: {avg_think_length}')
print(f'response: {avg_response_length}')

