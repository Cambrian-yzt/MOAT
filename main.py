from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import json
from datasets import load_dataset

from eval import eval_vqa_item
from configs.constants import *


if __name__ == '__main__':
    for run_id in range(1, N_RUNS + 1):
        print(f'VQA Model: {VQA_MODEL}')
        print(f'Temperature: {VQA_TEMPERATURE}')
        print(f'Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}')
        print(f'Eval Model: {EVAL_MODEL}')

        launch_time = datetime.now()
        dataset = load_dataset("waltsun/MOAT", split='test')
        
        result_dict = {}
        n_correct = 0
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = [executor.submit(eval_vqa_item, question_dict) for question_dict in dataset]
            
            pbar = tqdm(total=len(dataset), dynamic_ncols=True)
            for future in as_completed(futures):
                index, answer, reason, ground_truth, verdict, time_delta = future.result()
                n_correct += verdict
                result_dict[index] = {
                    'index': index,
                    'answer': answer,
                    'reason': reason,
                    'ground_truth': ground_truth,
                    'verdict': verdict
                }
                pbar.update(1)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        with open(os.path.join(LOG_DIR, f'{VQA_MODEL}_{run_id}.json'), 'w') as f:
            result_list = [result_dict[key] for key in sorted(result_dict.keys())]
            json.dump({
                'summary': {
                    'Correct Count': n_correct,
                    'Accuracy': n_correct / len(dataset),
                    'Launch Time': str(launch_time),
                    'VQA Model': VQA_MODEL,
                    'Run ID': run_id,
                    'Temperature': VQA_TEMPERATURE,
                    'Eval Model': EVAL_MODEL,
                },
                'logs': result_list,
            }, f, indent=4)
    