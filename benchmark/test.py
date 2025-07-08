# def test_loading_all():
#     def test_loading(dataset):
#         data, eval_func = load_longproc_data(dataset, "./LongProc/data")
#         print(f"Dataset: {dataset}")
#         print(f"N samples: {len(data)}")
#         print(f"Eval func: {eval_func}")
#         print(f"Max input chars: {max([len(d['input_prompt']) for d in data])}")
#         print(f"Max output chars: {max([len(d['reference_output']) for d in data])}")

#     [test_loading(d) for d in ["path_traversal_0.5k", "path_traversal_2k", "path_traversal_8k"]]

#     [test_loading(d) for d in ["html_to_tsv_0.5k", "html_to_tsv_2k", "html_to_tsv_8k"]]

#     [test_loading(d) for d in ["pseudo_to_code_0.5k", "pseudo_to_code_2k",]]

#     [test_loading(d) for d in ["travel_planning_2k", "travel_planning_8k"]]

#     [test_loading(d) for d in ["tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k"]]

#     [test_loading(d) for d in ["countdown_0.5k", "countdown_2k", "countdown_8k"]]



from LongProc.longproc.longproc_data import load_longproc_data
import ipdb

data, eval_func = load_longproc_data('path_traversal_0.5k', './LongProc/data')
ipdb.set_trace()
