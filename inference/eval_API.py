import time
import random
from openai import OpenAI

import base64
import os
from random import shuffle

from configs.constants import *
from configs.prompts import *


def get_response_API(query_text: str, query_image_paths: list[str], outside_knowledge_text: str, outside_knowledge_image_paths: list[str], choices: list[str], max_tokens=4096, max_tries=5) -> tuple[str, str]:
    """
        Exactly what it says in the names of the function and parameters
    """
    # read the image files and convert them to base64
    base64_query_images: list[tuple[str, str]] = []  # a list of tuples with this format: ('BASE64STRING', 'png') or ('BASE64STRING', 'jpg')
    base64_outside_knowledge_images: list[tuple[str, str]] = []
    for query_image_path in query_image_paths:
        with open(os.path.join('dataset', query_image_path), 'rb') as img:
            base64_query_images.append((base64.b64encode(img.read()).decode('utf-8'), query_image_path.split('.')[-1]))
    for outside_knowledge_image_path in outside_knowledge_image_paths:
        with open(os.path.join('dataset', outside_knowledge_image_path), 'rb') as img:
            base64_outside_knowledge_images.append((base64.b64encode(img.read()).decode('utf-8'), outside_knowledge_image_path.split('.')[-1]))

    # build a query context
    messages = [
        {
            'role': 'system' if 'o1' not in VQA_MODEL else 'developer',
            'content': [
                {
                    'type': 'text',
                    'text': VQA_SYSTEM_PROMPT,
                }
            ] if 'doubao' not in VQA_MODEL and 'moonshot' not in VQA_MODEL else VQA_SYSTEM_PROMPT  # Doubao only accepts a simple string as system prompt, which is strange :(
        },
    ]
    if len(choices) > 0:
        shuffle(choices)
        query_text += f'\nThe choices are: {choices}'
    user_message_contents = [
        {
            'type': 'text',
            'text': query_text,
        }
    ]
    for idx, (img, img_format) in enumerate(base64_query_images):
        user_message_contents.append({
            'type': 'text',
            'text': f'Image {idx + 1}',
        })    
        user_message_contents.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/{img_format};base64,{img}',
                'detail': VQA_API_VISION_DETAIL_LEVEL,
            }
        })
    if len(outside_knowledge_text) > 0:
        user_message_contents.append({
            'type': 'text',
            'text': 'Hint:\n' + outside_knowledge_text,
        })
    for idx, (img, img_format) in enumerate(base64_outside_knowledge_images):
        user_message_contents.append({
            'type': 'text',
            'text': f'Hint image {idx}',
        })    
        user_message_contents.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/{img_format};base64,{img}',
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
            if 'o1' in VQA_MODEL:
                completion = client.chat.completions.create(
                    model=VQA_MODEL,
                    temperature=1,
                    max_completion_tokens=16384,
                    messages=messages,
                    timeout=API_TIMEOUT
                )
            else:
                completion = client.chat.completions.create(
                    model=VQA_MODEL,
                    temperature=VQA_TEMPERATURE,
                    max_tokens=max_tokens,
                    messages=messages,
                    response_format={'type': 'json_object'},
                    timeout=API_TIMEOUT
                )
            if JSON_OUTPUT:
                raw_response = completion.model_dump()['choices'][0]['message']['content']
                try: 
                    if '```json' in raw_response:
                        raw_response = raw_response[7: -3]
                    response_json: dict = json.loads(raw_response)
                    # weird gemini stuff
                    if VQA_MODEL == 'gemini-2.0-pro-exp-02-05' or VQA_MODEL == 'gemini-2.0-flash-lite-preview-02-05':
                        try:
                            response_json = response_json[0]
                        except Exception as e:
                            pass
                    answer = response_json['answer']
                    reason = response_json.get('analysis', 'NO REASON')
                except Exception as e:
                    # json decoding failed, treat as non-json output
                    answer = raw_response
                    reason = 'NO REASON'
            else:
                answer = completion.model_dump()['choices'][0]['message']['content']
                reason = 'NO REASON'
            time.sleep(0.5)
            break
        except Exception as e:
            print(query_image_paths, e)
            answer = 'RESPONSE FAILED'
            reason = 'NO REASON'
            time.sleep(5 * (i + random.random()))
    return answer, reason
