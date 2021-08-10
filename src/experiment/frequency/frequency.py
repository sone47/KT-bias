# coding: utf-8
# 2021/5/11 @ sone

from collections import Counter
from src.experiment.utils import Experiment, divide_groups, calculate_all_bias, calculate_all_group_performance


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
    accuracy, auc, mse = calculate_all_group_performance(groups)

    print("The bias value (acc, auc, mse) of question frequency is %s." % str(bias))
    print('accuracy', accuracy)
    print('auc', auc)
    print('mse', mse)
