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

# get trained model
dkt = DKT(NUM_QUESTIONS, NUM_QUESTIONS // 10, HIDDEN_SIZE, NUM_LAYERS, device=device)
# get train/validation/test data loader
train_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.train_filename)
valid_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.valid_filename)
test_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.test_filename)
train_loader, valid_loader, test_loader = get_data_loader(train_data_path, valid_data_path, test_data_path, SEQ_LEN,
                                                          BATCH_SIZE, NUM_QUESTIONS, device=device)
# prepare log file
log_train_file = conf.log + '-train.log'
log_valid_file = conf.log + '-valid.log'

# train model
train_sequences = dkt.train(
    train_loader, valid_loader,
    epoch=5,
    train_log_file=log_train_file, test_log_file=log_valid_file,
)
dkt.save("dkt.params")

# test model
dkt.load("dkt.params")
(sequences, y_truth, y_pred), (auc, acc, rmse) = dkt.eval(test_loader)
print("auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (auc, acc, rmse))

# post-process
question_perf = get_questions_perf(sequences, y_truth, y_pred, NUM_QUESTIONS)
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

corr_value = corr(question_perf, train_question_ratio)
print("The coefficient of correlation of frequency and accuracy is %.6f." % corr_value)
