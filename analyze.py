"""
    For post analysis of dataset composition and model performance
"""
import os
import json
from collections import defaultdict
import pandas
from matplotlib import pyplot as plt

from configs.constants import *


CAPABILITIES = ['CNT', 'OCR', 'RLA', '3DTF', '3DQNT', 'GNDT', 'GNDV', 'UCV', 'RET', 'MTIMG', 'MTLIN']

def print_leaderboard():
    leaderboard_raw = defaultdict(list)
    for log_name in os.listdir(LOG_DIR):
        if log_name[-4: ] != 'json':
            continue
        model_name = '_'.join(log_name.split('_')[: -1])
        with open(os.path.join(LOG_DIR, log_name), mode='r',encoding='utf-8') as f:
            result_dict: dict = json.load(f)
        leaderboard_raw[model_name].append(result_dict['summary']['Accuracy'])
    leaderboard = []
    for model_name, results in leaderboard_raw.items():
        leaderboard.append((sum(results) / len(results), model_name))
    leaderboard.sort(key=lambda x: x[0], reverse=True)
    for item in leaderboard:
        print(f'Accuracy for {item[1]}:' + ' ' * (40 - len(item[1])) + f'{item[0]:.4f}')

def get_accuracy_by_capability_type():
    # produce a list of model names sorted by overall accuracy
    leaderboard_raw = defaultdict(list)
    for log_name in os.listdir(LOG_DIR):
        if log_name[-4: ] != 'json':
            continue
        model_name = '_'.join(log_name.split('_')[: -1])
        with open(os.path.join(LOG_DIR, log_name), mode='r',encoding='utf-8') as f:
            result_dict: dict = json.load(f)
        leaderboard_raw[model_name].append(result_dict['summary']['Accuracy'])
    leaderboard = []
    for model_name, results in leaderboard_raw.items():
        leaderboard.append((sum(results) / len(results), model_name))
    leaderboard.sort(key=lambda x: x[0], reverse=True)
    models = [item[1] for item in leaderboard]
    
    # init scoreboard
    df_raw = {
        'model': []
    }
    for capability in CAPABILITIES:
        df_raw[capability] = []
    df_raw['overall'] = []

    # iterate over all logs 
    for model in models:
        print(model)
        scoreboard_raw = defaultdict(list)
        for log_name in os.listdir(LOG_DIR):
            model_name = '_'.join(log_name.split('_')[: -1])
            if model_name != model or log_name.split('.')[-1] != 'json':
                continue
            with open(os.path.join(LOG_DIR, log_name), mode='r', encoding='utf-8') as f:
                result_dict: dict = json.load(f)
            for question_id, result in result_dict['logs'].items():
                for capability in QUESTIONS[question_id]['capability']:
                    scoreboard_raw[capability].append(result['verdict'])
            scoreboard_raw['overall'].append(result_dict['summary']['Accuracy'])
        df_raw['model'].append(model)
        for capability in df_raw.keys() - {'model'}:
            df_raw[capability].append(sum(scoreboard_raw[capability]) / len(scoreboard_raw[capability]))
            # print(f'{question_type} accuracy:' + ' ' * (25 - len(question_type)) + f'{sum(scoreboard_raw[question_type]) / len(scoreboard_raw[question_type]):.3f}')
        df = pandas.DataFrame(df_raw)
        df.to_csv(os.path.join(ANALYTICS_DIR, 'acc_by_capability.csv'))

def get_accuracy_by_capability_integration():
    # produce a list of model names sorted by overall accuracy
    leaderboard_raw = defaultdict(list)
    for log_name in os.listdir(LOG_DIR):
        if log_name[-4: ] != 'json':
            continue
        model_name = '_'.join(log_name.split('_')[: -1])
        with open(os.path.join(LOG_DIR, log_name), mode='r',encoding='utf-8') as f:
            result_dict: dict = json.load(f)
        leaderboard_raw[model_name].append(result_dict['summary']['Accuracy'])
    leaderboard = []
    for model_name, results in leaderboard_raw.items():
        leaderboard.append((sum(results) / len(results), model_name))
    leaderboard.sort(key=lambda x: x[0], reverse=True)
    models = [item[1] for item in leaderboard]
    
    # init scoreboard
    df_raw = {
        'model': []
    }
    for _, question in QUESTIONS.items():
        df_raw['+'.join(question['capability'])] = []
    df_raw['overall'] = []

    # iterate over all logs 
    for model in models:
        print(model)
        scoreboard_raw = defaultdict(list)
        for log_name in os.listdir(LOG_DIR):
            model_name = '_'.join(log_name.split('_')[: -1])
            if model_name != model or log_name.split('.')[-1] != 'json':
                continue
            with open(os.path.join(LOG_DIR, log_name), mode='r', encoding='utf-8') as f:
                result_dict: dict = json.load(f)
            for question_id, result in result_dict['logs'].items():
                scoreboard_raw['+'.join(QUESTIONS[question_id]['capability'])].append(result['verdict'])
            scoreboard_raw['overall'].append(result_dict['summary']['Accuracy'])
        df_raw['model'].append(model)
        for capability_list in df_raw.keys() - {'model'}:
            df_raw[capability_list].append(sum(scoreboard_raw[capability_list]) / len(scoreboard_raw[capability_list]))
            # print(f'{question_type} accuracy:' + ' ' * (25 - len(question_type)) + f'{sum(scoreboard_raw[question_type]) / len(scoreboard_raw[question_type]):.3f}')
        df = pandas.DataFrame(df_raw)
        df.to_csv(os.path.join(ANALYTICS_DIR, 'acc_by_capability_integration.csv'))


if __name__ == '__main__':
    if not os.path.exists(ANALYTICS_DIR):
        os.makedirs(ANALYTICS_DIR)
    print_leaderboard()
    get_accuracy_by_capability_type()