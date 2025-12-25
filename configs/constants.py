import os
import json

# General API settings
API_TIMEOUT = 180  # API timeout, in seconds. 3 minutes should be enough for o1 to think ;)

# VQA API settings
VQA_ENDPOINT = ''       # put your API endpoint here (e.g. https://api.openai.com/v1)
VQA_API_KEY = ''        # put your API key here
VQA_MODEL = 'gpt-4.1'   # use any model you like :)
VQA_API_VISION_DETAIL_LEVEL = 'high'
VQA_TEMPERATURE = 0.0

# Evaluation settings
JUDGE_ENDPOINT = ''      # put your API endpoint here (e.g. https://api.openai.com/v1)
JUDGE_API_KEY = ''       # put your API key here
JUDGE_MODEL = 'gpt-4.1'  # we used gpt-4o-mini to grade the answers in our paper. given that all questions have a single correct answer, we did not observe any grading errors by gpt-4o-mini.
JUDGE_TEMPERATURE = 0.0

# Task settings
N_RUNS = 3  # account for randomness in LLM output
LOG_DIR = os.path.join('.', 'logs')
ANALYTICS_DIR = os.path.join('.', 'analytics')

# Tweak this in accordance with your rate limit, which is defined by your API provider
MAX_CONCURRENT_REQUESTS = 32

# Older models may have limited output length :(
MAX_TOKENS = 16384

