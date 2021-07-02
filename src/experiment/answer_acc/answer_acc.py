# coding: utf-8
# 2021/6/18 @ sone

from os import path
import torch
from src import DKT, get_data_loader
from src import config as conf
from src.experiment.utils import prepare_data, stat_question_ratio, get_questions_perf, draw_scatter_figure, corr
from src.experiment.answer_acc.stat import stat_answer_acc

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
SEQ_LEN = conf.seq_len
device = torch.device(conf.device)
model_path = 'dkt-' + dataset + '.params'

# get trained model
dkt = DKT(NUM_QUESTIONS, NUM_QUESTIONS // 10, HIDDEN_SIZE, NUM_LAYERS, device=device)
# prepare log file
log_train_file = conf.log + '-train.log'
log_valid_file = conf.log + '-valid.log'


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


def stat_corr(test_sequences, y_truth, y_pred):
    data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.test_filename)
    # post-process
    question_perf = get_questions_perf(test_sequences, y_truth, y_pred, NUM_QUESTIONS)
    answer_acc = stat_answer_acc(data_path)

    union_keys = question_perf.keys() | answer_acc.keys()

    # delete invalid question: never showed when training and been predicted when testing
    for k in union_keys:
        if answer_acc.get(k) is None:
            del question_perf[k]
        if question_perf.get(k) is None and answer_acc.get(k):
            del answer_acc[k]

    question_perf = list(question_perf.values())
    answer_acc = list(answer_acc.values())
    draw_scatter_figure(
        question_perf, answer_acc,
        x_label='predict_acc', y_label='answer_acc',
        save_path=dataset + '-scatter.png',
    )

    return corr(question_perf, answer_acc)


if __name__ == '__main__':
    if path.exists(model_path):
        train_loader, valid_loader, test_loader = prepare_data(conf.data_dir, conf.dataset_dirname[dataset],
                                                               '', '', conf.test_filename,
                                                               device, NUM_QUESTIONS, SEQ_LEN, BATCH_SIZE)
    else:
        train_loader, valid_loader, test_loader = prepare_data(conf.data_dir, conf.dataset_dirname[dataset],
                                                               conf.train_filename, conf.valid_filename,
                                                               conf.test_filename,
                                                               device, NUM_QUESTIONS, SEQ_LEN, BATCH_SIZE)
        train(model_path, train_loader, valid_loader,
              epoch=conf.epoch, train_log_file=log_train_file, test_log_file=log_valid_file)
    test_sequences, truth, pred = test(model_path, test_loader)

    corr_value = stat_corr(test_sequences, truth, pred)

    print("The coefficient of correlation of accuracy of answer and accuracy of prediction is %.6f." % corr_value)
