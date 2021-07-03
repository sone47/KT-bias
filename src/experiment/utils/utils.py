# coding: utf-8
# 2021/5/11 @ sone

from os import path

import numpy as np
from sklearn.metrics import accuracy_score
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


def get_questions_perf(question_sequences, truth, pred, num_questions) -> dict:
    questions_perf = {}
    for i in range(num_questions):
        index = question_sequences == i
        q = question_sequences[index]
        t = truth[index]
        p = pred[index]
        if len(q) > 0:
            acc = accuracy_score(t, p >= 0.5)
            questions_perf[i] = acc
    return questions_perf


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


class Experiment:
    def __init__(self, num_question, hidden_size, num_layer, seq_len, batch_size, device, dataset, data_dir, dataset_dirname,
                 model_save_path='.'):
        from src import DKT

        self.model = DKT(num_question, num_question // 10, hidden_size, num_layer, device=device)
        self.model_save_path = model_save_path
        self.num_question = num_question
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.dataset_dirname = dataset_dirname
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size

    def train(self, train_data, test_data, epoch=5, train_log_file='', test_log_file=''):
        sequences = self.model.train(
            train_data, test_data,
            epoch,
            train_log_file=train_log_file, test_log_file=test_log_file,
        )
        self.model.save(self.model_save_path)
        return sequences

    def test(self, test_data):
        self.model.load(self.model_save_path)
        (sequences, y_truth, y_pred), (auc, acc, rmse) = self.model.eval(test_data)
        print("auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (auc, acc, rmse))
        return sequences, y_truth, y_pred

    def calculate_corr(self, stat_func, prop_name, filename, test_sequences, y_truth, y_pred):
        data_path = path.join(self.data_dir, self.dataset_dirname, filename)
        # post-process
        question_perf = get_questions_perf(test_sequences, y_truth, y_pred, self.num_question)
        related_prop = stat_func(data_path)

        union_keys = question_perf.keys() | related_prop.keys()

        # delete invalid item
        for k in union_keys:
            if related_prop.get(k) is None:
                del question_perf[k]
            if question_perf.get(k) is None and related_prop.get(k):
                del related_prop[k]

        question_perf = list(question_perf.values())
        related_prop = list(related_prop.values())
        draw_scatter_figure(
            question_perf, related_prop,
            x_label='acc', y_label=prop_name,
            save_path=self.dataset_name + '.png',
        )

        return corr(question_perf, related_prop)

    def run(self, epoch, train_log_file, test_log_file, stat_func, prop_name, train_filename, valid_filename,
            test_filename):
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
                       epoch=epoch, train_log_file=train_log_file, test_log_file=test_log_file)

        test_sequences, truth, pred = self.test(test_loader)
        corr_value = self.calculate_corr(stat_func, prop_name, test_filename, test_sequences, truth, pred)

        print("The coefficient of correlation of %s and accuracy is %.6f." % (prop_name, corr_value))
