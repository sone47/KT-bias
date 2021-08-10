# coding: utf-8
# 2021/6/18 @ sone

from collections import Counter
from src.experiment.utils import divide_groups, calculate_all_bias, calculate_all_group_performance


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
    groups = divide_groups(result_data, lambda x: x['answer_accuracy'], 0.1)
    # calculate bias
    bias = calculate_all_bias((groups[0], groups[-1]))
    # calculate all groups performance
    accuracy, auc, mse = calculate_all_group_performance(groups)

    print("The bias value (acc, auc, mse) of answer accuracy is %s." % str(bias))
    print('accuracy', accuracy)
    print('auc', auc)
    print('mse', mse)
