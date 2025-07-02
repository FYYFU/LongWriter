
import json
import ipdb
import jsonlines
from collections import defaultdict

def load_query_categories(jsonl_file_path):
    """
    Loads criteria from a JSONL file into a dictionary.
    """
    data_list = {}
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            data_list[obj['index']] = {}
            data_list[obj['index']]['query'] = obj['query']
            data_list[obj['index']]['domain'] = [obj['domain1'], obj['domain2']]
    return data_list


data_path = '/home/greenland-user/LongWriter/benchmark/WritingBench_outputs/models/Qwen3-8B/pred_extend_w0_g0_tTrue_eval.jsonl'
query_path = '/home/greenland-user/LongWriter/benchmark/WritingBench/benchmark_query/benchmark_all.jsonl'


with open(data_path, 'r') as f:
    data = [json.loads(l) for l in f]

final_scores = defaultdict(list)
categories = load_query_categories(query_path)

for item in data:
    index = item['index']
    cur_scores_list = item['scores']
    try:
        category = categories[index]['domain'][0]
        cur_scores = [i[0]['score'] for i in cur_scores_list.values()]
        avg_score = sum(cur_scores) / len(cur_scores)
        final_scores[category].append(avg_score)
    except Exception:
        print(f'{index} cannot be found')


# print(final_scores)

for key in final_scores.keys():
    avg_key_score = round(sum(final_scores[key]) / len(final_scores[key]), 1)
    print(f'{key}: {avg_key_score}')

