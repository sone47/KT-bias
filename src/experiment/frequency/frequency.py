# coding: utf-8
# 2021/5/11 @ sone

import numpy as np
from collections import Counter
from src.experiment.utils import Experiment, divide_groups, calculate_all_bias, calculate_all_group_performance, draw_stat_graph


def calculate_question_freq(questions):
    freq = Counter(questions)
    sum_question = sum(freq.values())
    for k in freq.keys():
        freq[k] /= sum_question
    return freq


def output_processor(questions, _, truth, pred):
    freq = calculate_question_freq(questions)
    result_data = [{
        'freq': freq[q],
        'truth': truth[i],
        'pred': pred[i],
    } for i, q in enumerate(questions)]

    # divide two groups
    groups = divide_groups(result_data, lambda x: x['freq'], 0.1)
    # calculate bias
    bias = calculate_all_bias((groups[0], groups[-1]))
    # calculate all groups performance
    group_accuracy, group_auc, group_mse = calculate_all_group_performance(groups)
    # calculate the correlation of question frequency with prediction performance
    values = np.array([sum([item['freq'] for item in group]) for group in groups])
    accuracy_corr = np.corrcoef(values, np.array(group_accuracy))[0, 1]
    auc_corr = np.corrcoef(values, np.array(group_auc))[0, 1]
    mse_corr = np.corrcoef(values, np.array(group_mse))[0, 1]

    print("The bias value (acc, auc, mse) of question frequency is %s." % str(bias))
    print('accuracy correlation value: ', accuracy_corr)
    print('auc correlation value: ', auc_corr)
    print('mse correlation value: ', mse_corr)
