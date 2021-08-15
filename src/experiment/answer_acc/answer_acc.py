# coding: utf-8
# 2021/6/18 @ sone

import numpy as np
from collections import Counter
from src.experiment.utils import divide_groups, calculate_all_bias, calculate_all_group_performance, calculate_accuracy


def calculate_all_answer_accuracy(questions, answers):
    c = Counter(questions)
    answer_accuracy = {}
    for i, q in enumerate(questions):
        if answers[i] == 1:
            answer_accuracy.setdefault(q, answer_accuracy.get(q, 0) + 1)

    for k, v in answer_accuracy.items():
        answer_accuracy[k] /= c[k]

    return answer_accuracy


def output_processor(questions, _, answers, pred):
    answer_accuracy = calculate_all_answer_accuracy(questions, answers)
    result_data = [{
        'answer_accuracy': answer_accuracy.get(q, 0),
        'truth': answers[i],
        'pred': pred[i],
    } for i, q in enumerate(questions)]
    # divide two groups
    groups = divide_groups(result_data, lambda x: x['answer_accuracy'], 0.1, ascending=True)
    # calculate bias
    bias = calculate_all_bias((groups[0], groups[-1]))
    # calculate all groups performance
    group_accuracy, group_auc, group_mse = calculate_all_group_performance(groups)
    # calculate the correlation of answer accuracy with prediction performance
    values = np.array([sum([item['answer_accuracy'] for item in group]) for group in groups])
    accuracy_corr = np.corrcoef(values, np.array(group_accuracy))[0, 1]
    auc_corr = np.corrcoef(values, np.array(group_auc))[0, 1]
    mse_corr = np.corrcoef(values, np.array(group_mse))[0, 1]

    print("The bias value (acc, auc, mse) of answer accuracy is %s." % str(bias))
    print('accuracy correlation value: ', accuracy_corr)
    print('auc correlation value: ', auc_corr)
    print('mse correlation value: ', mse_corr)
