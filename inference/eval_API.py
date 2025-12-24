import time
from time import perf_counter
import random
from openai import OpenAI
from PIL.Image import Image

import base64
from io import BytesIO
import os
from random import shuffle

from configs.constants import *
from configs.prompts import *


def get_response_API(question: str, images: list[Image], outside_knowledge_text: str, outside_knowledge_images: list[Image], choices: list[str], max_tokens=4096, max_tries=5) -> tuple[str, str]:
    """
        Exactly what it says in the names of the function and parameters
    """
    # read the image files and convert them to base64
    base64_images: list[str] = []
    base64_outside_knowledge_images: list[str] = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
    for img in outside_knowledge_images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_outside_knowledge_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

    # build a query context
    messages = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': VQA_SYSTEM_PROMPT,
                }
            ]
        },
    ]
    if len(choices) > 0:
        shuffle(choices)
        question += f'\nThe choices are: {choices}'
    user_message_contents = [
        {
            'type': 'text',
            'text': question,   # may with choices
        }
    ]
    for idx, img in enumerate(base64_images):
        user_message_contents.append({
            'type': 'text',
            'text': f'Image {idx + 1}',
        })    
        user_message_contents.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpg;base64,{img}',
                'detail': VQA_API_VISION_DETAIL_LEVEL,
            }
        })
    if len(outside_knowledge_text) > 0:
        user_message_contents.append({
            'type': 'text',
            'text': 'Hint:\n' + outside_knowledge_text,
        })
    for idx, img in enumerate(base64_outside_knowledge_images):
        user_message_contents.append({
            'type': 'text',
            'text': f'Hint image {idx}',
        })    
        user_message_contents.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpg;base64,{img}',
                'detail': VQA_API_VISION_DETAIL_LEVEL,
            }
        })
    messages.append({
        'role': 'user',
        'content': user_message_contents,
    })

    # client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY, api_version=AZURE_API_VERSION)
    client = OpenAI(api_key=VQA_API_KEY, base_url=VQA_ENDPOINT)
    
    for i in range(max_tries):
        try:
            start_time = perf_counter()
            completion = client.chat.completions.create(
                model=VQA_MODEL,
                temperature=VQA_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
                response_format={'type': 'json_object'},
                timeout=API_TIMEOUT
            )
            end_time = perf_counter()
            completion = completion.model_dump()
            raw_response = completion['choices'][0]['message']['content']
            prompt_tokens = completion['usage']['prompt_tokens']
            completion_tokens = completion['usage']['completion_tokens']
            try:
                if '```json' in raw_response:
                    raw_response = raw_response[7: -3]
                response_json: dict = json.loads(raw_response)
                answer = response_json['answer']
                reason = response_json.get('analysis', 'NO REASON')
            except Exception as e:
                answer= raw_response
                reason = 'N/A'
            break
        except Exception as e:
            print(images, e)
            answer = 'RESPONSE FAILED'
            reason = 'NO REASON'
            prompt_tokens = 0
            completion_tokens = 0
    try:
        time_delta = end_time - start_time
    except Exception as e:
        time_delta = -1

    return answer, reason, time_delta, prompt_tokens, completion_tokens
