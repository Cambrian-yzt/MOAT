from openai import OpenAI
from time import perf_counter
from PIL.Image import Image

from configs.constants import *
from configs.prompts import *
from inference.eval_API import get_response_API


def eval_answer(input: str, ground_truth: str, max_tokens=4096, max_tries=5) -> int:
    # build a query context
    eval_query = f'The answer to evaluate is {input}\nThe ground truth answer is {ground_truth}'
    messages = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': JUDGE_SYSTEM_PROMPT,
                }
            ]
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': eval_query,
                }
            ]
        }
    ]
    client = OpenAI(base_url=JUDGE_ENDPOINT, api_key=JUDGE_API_KEY)
    for _ in range(max_tries):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=max_tokens,
                messages=messages,
                response_format={'type': 'json_object'},
                timeout=API_TIMEOUT
            )
            break
        except Exception as e:
            print(e)
    response_json = json.loads(completion.model_dump()['choices'][0]['message']['content'])
    return int(response_json['verdict'])


def eval_vqa_item(question_dict: dict, max_tries=5) -> tuple[str, str, str, str, int, float]:
    """
        Evaluate one VQA item
    """
    start_time = perf_counter()
    try:
        question_index: int = question_dict['index']
        question_text: str = question_dict['question']
        choices: list[str] = question_dict['choices']
        images: list[Image] = question_dict['task_images']
        outside_knowledge_text: str = question_dict['external_knowledge']
        outside_knowledge_images: list[Image] = question_dict['external_knowledge_images']
        ground_truth: str = question_dict['ground_truth']
    except Exception as e:
        print(f"Failed to extract question data for question {question_dict}: {e}")
        raise
    
    try:
        # use the API
        answer, reason, time_delta, prompt_tokens, completion_tokens = get_response_API(question_text, images, outside_knowledge_text, outside_knowledge_images, choices, max_tries=max_tries, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f"Failed to get a valid answer for question {question_index}: {e}")
        raise
    
    try:
        verdict = eval_answer(answer, ground_truth)
    except Exception as e:
        print(f"Failed to get a valid verdict for the answer to question {question_index}: {e}")
        verdict = 0
        # raise
    
    end_time = perf_counter()
    return question_index, answer, reason, ground_truth, verdict, time_delta, prompt_tokens, completion_tokens


if __name__ == '__main__':
    from datasets import load_dataset, Dataset
    dataset = load_dataset("waltsun/MOAT", split='test')
    index, answer, reason, ground_truth, verdict, time_delta = eval_vqa_item(dataset[0])
    print(f"Model: {VQA_MODEL}")
    print(f"Question Index: {index}. Time taken: {time_delta:.2f} seconds.")
    print(f"Answer: {answer}")
    print(f"Reason: {reason}")
    print(f"Ground truth: {ground_truth}")
    print(f"Verdict: {verdict}")