# coding: utf-8
# 2021/8/8 @ sone

from src.experiment.utils import divide_groups, calculate_all_bias, calculate_all_group_performance


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
    accuracy, auc, mse = calculate_all_group_performance(groups)

    print("The bias value (acc, auc, mse) of interval_time is %s." % str(bias))
    print('accuracy', accuracy)
    print('auc', auc)
    print('mse', mse)
