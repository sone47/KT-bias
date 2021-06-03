# coding: utf-8
# 2021/5/11 @ sone

import os.path as path

import torch
from src import DKT, get_data_loader
from src import config as conf
from utils import stat_question_ratio, get_questions_perf, draw_scatter_figure, corr

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
SEQ_LEN = conf.seq_len
device = torch.device(conf.device)
model_path = "dkt.params"

# get trained model
dkt = DKT(NUM_QUESTIONS, NUM_QUESTIONS // 10, HIDDEN_SIZE, NUM_LAYERS, device=device)
# prepare log file
log_train_file = conf.log + '-train.log'
log_valid_file = conf.log + '-valid.log'


def prepare_data(data_dir, dataset_dirname, train_filename, valid_filename, test_filename):
    train_data_path = path.join(data_dir, dataset_dirname, train_filename)
    valid_data_path = path.join(data_dir, dataset_dirname, valid_filename)
    test_data_path = path.join(data_dir, dataset_dirname, test_filename)
    return get_data_loader(train_data_path, valid_data_path, test_data_path,
                           SEQ_LEN, BATCH_SIZE, NUM_QUESTIONS, device=device)


def train(model_save_path, train_data, test_data, epoch=5, train_log_file='', test_log_file=''):
    sequences = dkt.train(
        train_data, test_data,
        epoch,
        train_log_file=train_log_file, test_log_file=test_log_file,
    )
    dkt.save(model_save_path)
    return sequences


def test(model_save_path, test_data):
    dkt.load(model_save_path)
    (sequences, y_truth, y_pred), (auc, acc, rmse) = dkt.eval(test_data)
    print("auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (auc, acc, rmse))
    return sequences, y_truth, y_pred


def stat_corr(train_sequences, test_sequences, y_truth, y_pred):
    # post-process
    question_perf = get_questions_perf(test_sequences, y_truth, y_pred, NUM_QUESTIONS)
    train_question_ratio = stat_question_ratio(train_sequences, NUM_QUESTIONS)

    # delete invalid question: never showed when training and been predicted when testing
    keys_deleted = []
    for k in train_question_ratio.keys():
        if train_question_ratio[k] == 0 or question_perf[k] == -1:
            keys_deleted.append(k)

    for k in keys_deleted:
        del train_question_ratio[k]
        del question_perf[k]

    question_perf = list(question_perf.values())
    train_question_ratio = list(train_question_ratio.values())
    draw_scatter_figure(
        question_perf, train_question_ratio,
        x_label='acc', y_label='freq',
        save_path=dataset + '-scatter.png',
    )

    return corr(question_perf, train_question_ratio)


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = prepare_data(conf.data_dir, conf.dataset_dirname[dataset],
                                                           conf.train_filename, conf.valid_filename, conf.test_filename)
    train_sequences = train(model_path, train_loader, valid_loader,
                            epoch=conf.epoch, train_log_file=log_train_file, test_log_file=log_valid_file)
    test_sequences, truth, pred = test(model_path, test_loader)

    corr_value = stat_corr(train_sequences, test_sequences, truth, pred)

    print("The coefficient of correlation of frequency and accuracy is %.6f." % corr_value)
