import os
import json

# General API settings
API_TIMEOUT = 180  # API timeout, in seconds. 3 minutes should be enough for o1 to think ;)

# VQA API settings
VQA_ENDPOINT = ''           # put your API endpoint here (e.g. https://api.openai.com/v1)
VQA_API_KEY = ''            # put your API key here
VQA_MODEL = 'gpt-4o-mini'   # use any model you like :)
VQA_API_VISION_DETAIL_LEVEL = 'high'
VQA_TEMPERATURE = 0.0

# Evaluation settings
EVAL_ENDPOINT = ''          # put your API endpoint here (e.g. https://api.openai.com/v1)
EVAL_API_KEY = ''           # put your API key here
EVAL_MODEL = 'gpt-4o-mini'  # we used gpt-4o-mini to grade the answers in our paper. given that all questions have a single correct answer, we did not observe any grading errors by gpt-4o-mini.
EVAL_TEMPERATURE = 0.0

# Task settings
N_RUNS = 3  # account for randomness in LLM output
LOG_DIR = os.path.join('.', 'logs')
with open(os.path.join('.', 'dataset', 'questions.json'), 'r', encoding='utf-8') as f:
    QUESTIONS: dict = json.load(f)
ANALYTICS_DIR = os.path.join('.', 'analytics')

# Tweak this in accordance with your rate limit, which is defined by your API provider
MAX_CONCURRENT_REQUESTS = 32

# A lot models are very dumb and do not know how to write a very simple JSON object :(
# The evaluation script would try to decode the output of the LMM if this is set to True.
JSON_OUTPUT = True

# Some models are very dumb and have a limited output length :(
MAX_TOKENS = 2048

