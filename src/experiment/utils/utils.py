# coding: utf-8
# 2021/5/11 @ sone

from os import path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def calculate_all_group_performance(groups):
    accuracy = []
    auc = []
    mse = []
    for group in groups:
        truth = np.array([item['truth'] for item in group])
        pred = np.array([item['pred'] for item in group])
        accuracy.append(calculate_accuracy(truth, pred))
        auc.append(calculate_auc(truth, pred))
        mse.append(calculate_mse(truth, pred))
    return accuracy, auc, mse


def divide_groups(data, sort_func, ratio, ascending=True):
    """
    split data into groups by ratio
    :param data: data split
    :param sort_func: sorted function
    :param ratio: group ratio, recommend the number that could be divisible by 1
    :param ascending: if True, groups from smaller to larger
    :return: (large group, small group)
    """
    span = int(len(data) * ratio)
    n = int(1 / ratio)
    # sorted from large to small
    data.sort(key=sort_func, reverse=(not ascending))
    # divide data into n groups
    groups = []
    for i in range(n):
        if i == n - 1:
            groups.append(data[span * i:])
        else:
            groups.append(data[span * i: span * (i + 1)])
    return groups


def calculate_accuracy(truth, pred):
    pred[pred > 0.5] = 1.0
    pred[pred <= 0.5] = 0.0
    return accuracy_score(truth, pred)


def calculate_auc(truth, pred):
    return roc_auc_score(truth, pred)


def calculate_mse(truth, pred):
    return mean_squared_error(truth, pred)


def calculate_bias(groups, metrics_func):
    g1, g2 = groups
    t1 = np.array([item['truth'] for item in g1])
    t2 = np.array([item['truth'] for item in g2])
    p1 = np.array([item['pred'] for item in g1])
    p2 = np.array([item['pred'] for item in g2])
    return metrics_func(t1, p1) - metrics_func(t2, p2)


def calculate_all_bias(groups):
    # calculate bias
    bias_accuracy = calculate_bias(groups, calculate_accuracy)
    bias_auc = calculate_bias(groups, calculate_auc)
    bias_mse = calculate_bias(groups, calculate_mse)

    return bias_accuracy, bias_auc, bias_mse


def draw_stat_graph(x, y, graph_save_path='', title='', x_label='', y_label=''):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    if graph_save_path:
        pass
    plt.show()


class Experiment:
    def __init__(self, model, model_save_path='.'):
        self.model = model
        self.model_save_path = model_save_path

    def train(self, train_data, valid_data, epoch, lr, train_log_file='', test_log_file=''):
        self.model.train(
            train_data, valid_data,
            epoch=epoch, lr=lr,
            train_log_file=train_log_file, test_log_file=test_log_file,
            save_filepath=self.model_save_path,
        )

    def test(self, test_data):
        self.model.load(self.model_save_path)
        data, (auc, acc, mse) = self.model.eval(test_data)
        print("auc: %.6f, accuracy: %.6f, MSE: %.6f" % (auc, acc, mse))
        return data

    def run(self, train_data, valid_data, test_data, epoch, lr, train_log_file, test_log_file, output_processor):
        if not path.exists(self.model_save_path):
            self.train(train_data, valid_data, epoch=epoch, lr=lr, train_log_file=train_log_file, test_log_file=test_log_file)

        output = self.test(test_data)

        output_processor(*output)
