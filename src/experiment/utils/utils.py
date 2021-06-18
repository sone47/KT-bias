# coding: utf-8
# 2021/5/11 @ sone

from os import path

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src import get_data_loader


def prepare_data(data_dir, dataset_dirname, train_filename, valid_filename, test_filename, device, num_question, seq_len=50, batch_size=64):
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
