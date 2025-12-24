import json

# with cot
VQA_SYSTEM_PROMPT = json.dumps({
    'task': 'Answer the question presented to you truthfully.',
    'requirements': [
        'Analyze the image(s) first, then answer the question. If you are given a list of possible answers, you must choose from it.',
        'You must answer in the following json format: {"analysis": "(write your analysis here)", "answer": "(your answer)"}'
    ]
})

JUDGE_SYSTEM_PROMPT = json.dumps({
    'task': 'Evaluate whether the answer to a question is correct.',
    'requirements': [
        'Compare an answer to a question with the ground truth answer. Determine whether it is correct.',
        'You must ignore any analysis of the problem if present. You must focus only on the final answer.',
        'You must answer in the following json format: {"verdict": "(1 for correct, 0 for incorrect)"}'
    ]
})