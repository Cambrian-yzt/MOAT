from openai import OpenAI
from time import perf_counter

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
                    'text': EVAL_SYSTEM_PROMPT,
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
    client = OpenAI(base_url=EVAL_ENDPOINT, api_key=EVAL_API_KEY)
    for _ in range(max_tries):
        try:
            completion = client.chat.completions.create(
                model=EVAL_MODEL,
                temperature=EVAL_TEMPERATURE,
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


def eval_vqa_item(question_id: str, max_tries=5) -> tuple[str, str, str, str, int, float]:
    """
        Takes in a quesion ID (see dataset/questions.py). 
    """
    start_time = perf_counter()
    try:
        question = QUESTIONS[question_id]
        query_image_names = question['image_names']
        question_text = question['question']
        outside_knowledge_text = question['outside_knowledge_text']
        outside_knowledge_image_names = question['outside_knowledge_image_names']
        choices = question['choices']
        ground_truth = question['answer']
    except Exception as e:
        print(f"Failed to extract question data for question {question_id}: {e}")
        raise
    
    try:
        # use the API
        answer, reason = get_response_API(question_text, query_image_names, outside_knowledge_text, outside_knowledge_image_names, choices, max_tries=max_tries, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f"Failed to get a valid answer for question {question_id}: {e}")
        raise
    
    try:
        verdict = eval_answer(answer, ground_truth)
    except Exception as e:
        print(f"Failed to get a valid verdict for the answer to question {question_id}: {e}")
        verdict = 0
        # raise
    
    end_time = perf_counter()
    return question_id, answer, reason, ground_truth, verdict, end_time - start_time


if __name__ == '__main__':
    question_id = '10_11'
    question_id, answer, reason, ground_truth, verdict, time_delta = eval_vqa_item(question_id)
    print(f"Model: {VQA_MODEL}")
    print(f"Question ID: {question_id}. Time taken: {time_delta:.2f} seconds.")
    print(f"Answer: {answer}")
    print(f"Reason: {reason}")
    print(f"Ground truth: {ground_truth}")
    print(f"Verdict: {verdict}")