
import json
import ipdb
import collections

total_score = collections.defaultdict(list)
data_path = '/home/greenland-user/LongWriter/benchmark/LongProc_outputs/models/Qwen3-8B/pred_merged_thtml_to_tsv_w0_g0_s0.5k.jsonl'


eval_metrics = []
with open(data_path, 'r') as f:
    for line in f.readlines():
        item = json.loads(line)
        metrics = item['metrics']
        eval_metrics.append(metrics[0])

for k, v in metrics[0].items():
    print(f"{k}: {sum([m[k] for m in eval_metrics])/len(eval_metrics)}")
