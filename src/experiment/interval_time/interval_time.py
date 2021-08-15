# coding: utf-8
# 2021/8/8 @ sone

import numpy as np
from src.experiment.utils import divide_groups, calculate_all_bias, calculate_all_group_performance, draw_stat_graph, \
    calculate_accuracy, draw_stat_graph


def output_processor(sequence, features, truth, pred):
    result_data = [{
        'interval': features[0][i],
        'truth': truth[i],
        'pred': pred[i],
    } for i, q in enumerate(sequence)]
    # divide two groups
    groups = divide_groups(result_data, lambda x: x['interval'], 0.1)
    # calculate bias
    bias = calculate_all_bias((groups[0], groups[-1]))
    # calculate all groups performance
    group_accuracy, group_auc, group_mse = calculate_all_group_performance(groups)
    # calculate the correlation of interval time with prediction performance
    values = np.array([sum([item['interval'] for item in group]) for group in groups])
    accuracy_corr = np.corrcoef(values, np.array(group_accuracy))[0, 1]
    auc_corr = np.corrcoef(values, np.array(group_auc))[0, 1]
    mse_corr = np.corrcoef(values, np.array(group_mse))[0, 1]

    print("The bias value (acc, auc, mse) of interval time is %s." % str(bias))
    print('accuracy correlation value: ', accuracy_corr)
    print('auc correlation value: ', auc_corr)
    print('mse correlation value: ', mse_corr)
