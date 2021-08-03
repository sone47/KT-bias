# coding: utf-8
# 2021/5/11 @ sone

from os import path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

from src import get_data_loader


def prepare_data(data_dir, dataset_dirname, train_filename, valid_filename, test_filename, device, num_question,
                 seq_len=50, batch_size=64):
    train_data_path = path.join(data_dir, dataset_dirname, train_filename)
    valid_data_path = path.join(data_dir, dataset_dirname, valid_filename)
    test_data_path = path.join(data_dir, dataset_dirname, test_filename)
    return get_data_loader(train_data_path, valid_data_path, test_data_path,
                           seq_len, batch_size, num_question, device=device)


def stat_question_ratio(question_sequences, num_questions) -> dict:
    total_questions = len(question_sequences)
    question_ratio = {i: 0 for i in range(num_questions)}
    for q in question_sequences:
        question_ratio[q] += 1
    for i in range(num_questions):
        question_ratio[i] /= total_questions
    return question_ratio


def __calc_perf(question_sequences, truth, pred, num_questions, perf_func) -> dict:
    questions_perf = {}
    for i in range(num_questions):
        index = question_sequences == i
        q = question_sequences[index]
        t = truth[index]
        p = pred[index]
        if len(q) > 0:
            try:
                questions_perf[i] = perf_func(t, p)
            except ValueError:
                questions_perf[i] = 0
    return questions_perf


def calc_questions_acc(question_sequences, truth, pred, num_questions) -> dict:
    return __calc_perf(question_sequences, truth, pred >= 0.5, num_questions, accuracy_score)


def calc_question_mse(question_sequences, truth, pred, num_questions) -> dict:
    return __calc_perf(question_sequences, truth, pred, num_questions, mean_squared_error)


def calc_question_auc(question_sequences, truth, pred, num_questions) -> dict:
    return __calc_perf(question_sequences, truth, pred, num_questions, roc_auc_score)


def draw_scatter_figure(x, y, x_label='', y_label='', title='', save_path='', fig_size=[10, 10]) -> ...:
    plt.figure(figsize=fig_size)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    plt.grid()
    if save_path != '':
        plt.savefig(fname=save_path)
    plt.show()


def corr(x, y) -> float:
    return np.corrcoef(x, y)[0, 1]


def calc_groups(data, ratio):
    """
    split data into two group by ratio
    :param data: data split
    :param ratio: group size
    :return: large group, small group
    """
    l = len(data)
    # sorted from large to small
    sorted_answer_acc = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_answer_acc[:int(l * ratio)]), dict(sorted_answer_acc[-int(l * ratio):])


def calc_bias(groups, metrics):
    g1, g2 = groups
    l1, l2 = len(g1), len(g2)
    sum1, sum2 = 0, 0
    for k in metrics.keys():
        if k in g1:
            sum1 += metrics[k]
        elif k in g2:
            sum2 += metrics[k]
    return sum1 / l1 - sum2 / l2


class Experiment:
    def __init__(self, model_class, num_question, hidden_size, num_layer, seq_len, batch_size, device, dataset,
                 data_dir, dataset_dirname,
                 model_save_path='.'):
        self.model = model_class(num_question, num_question // 10, hidden_size, num_layer, device=device)
        self.model_save_path = model_save_path
        self.num_question = num_question
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.dataset_dirname = dataset_dirname
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size

    def train(self, train_data, test_data, epoch, lr, train_log_file='', test_log_file=''):
        sequences = self.model.train(
            train_data, test_data,
            epoch, lr=lr,
            train_log_file=train_log_file, test_log_file=test_log_file,
        )
        self.model.save(self.model_save_path)
        return sequences

    def test(self, test_data):
        self.model.load(self.model_save_path)
        (sequences, y_truth, y_pred), (auc, acc, mse) = self.model.eval(test_data)
        print("auc: %.6f, accuracy: %.6f, MSE: %.6f" % (auc, acc, mse))
        return sequences, y_truth, y_pred

    def calculate_data(self, stat_func, prop_name, filename, test_sequences, y_truth, y_pred, group_ratio=0.1):
        data_path = path.join(self.data_dir, self.dataset_dirname, filename)
        # post-process
        question_acc = calc_questions_acc(test_sequences, y_truth, y_pred, self.num_question)
        question_auc = calc_question_auc(test_sequences, y_truth, y_pred, self.num_question)
        question_mse = calc_question_mse(test_sequences, y_truth, y_pred, self.num_question)
        related_data = stat_func(data_path)

        union_keys = question_acc.keys() | related_data.keys()
        # delete invalid item
        for k in union_keys:
            if related_data.get(k) is None:
                del question_acc[k]
                del question_auc[k]
                del question_mse[k]
            if question_acc.get(k) is None and related_data.get(k):
                del related_data[k]

        groups = calc_groups(related_data, group_ratio)
        bias_acc = calc_bias(groups, question_acc)
        bias_auc = calc_bias(groups, question_auc)
        bias_mse = calc_bias(groups, question_mse)

        question_acc = list(question_acc.values())
        question_auc = list(question_auc.values())
        question_mse = list(question_mse.values())
        related_data = list(related_data.values())
        # draw_scatter_figure(
        #     question_auc, related_data,
        #     x_label='acc', y_label=prop_name,
        #     save_path=self.dataset_name + '.png',
        # )
        corr_acc = corr(question_acc, related_data)
        corr_auc = corr(question_auc, related_data)
        corr_mse = corr(question_mse, related_data)

        return (corr_acc, corr_auc, corr_mse), (bias_acc, bias_auc, bias_mse)

    def run(self, epoch, lr, train_log_file, test_log_file, stat_func, prop_name, train_filename, valid_filename,
            test_filename, group_ratio=0.1):
        if path.exists(self.model_save_path):
            train_loader, valid_loader, test_loader = prepare_data(self.data_dir, self.dataset_dirname,
                                                                   '', '', test_filename,
                                                                   self.device, self.num_question, self.seq_len,
                                                                   self.batch_size)
        else:
            train_loader, valid_loader, test_loader = prepare_data(self.data_dir, self.dataset_dirname,
                                                                   train_filename, valid_filename, test_filename,
                                                                   self.device, self.num_question, self.seq_len,
                                                                   self.batch_size)
            self.train(train_loader, valid_loader,
                       epoch=epoch, lr=lr, train_log_file=train_log_file, test_log_file=test_log_file)

        test_sequences, truth, pred = self.test(test_loader)
        corr_value, bias = self.calculate_data(stat_func, prop_name, test_filename, test_sequences, truth, pred,
                                               group_ratio)
        corr_value = tuple(map(str, corr_value))
        bias = tuple(map(str, bias))

        print("The coefficient of correlation(acc, auc, mse) of %s and prediction accuracy is %s." % (
            prop_name, corr_value))
        print("The bias value (acc, auc, mse) of %s and prediction accuracy is %s." % (prop_name, bias))
