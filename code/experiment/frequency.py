# coding: utf-8
# 2021/5/11 @ sone

import os.path as path

import numpy as np

from code import DKT, get_data_loader
from code import config as conf
from utils import stat_question_ratio, get_questions_perf, draw_scatter_figure, corr

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
MAX_STEP = conf.max_step

dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)

train_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.train_filename)
valid_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.valid_filename)
test_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], conf.test_filename)
train_loader, valid_loader, test_loader = get_data_loader(train_data_path, valid_data_path, test_data_path, MAX_STEP,
                                                          BATCH_SIZE, NUM_QUESTIONS)

# train model
train_sequences = dkt.train(train_loader, valid_loader, epoch=5)
dkt.save("dkt.params")

train_question_ratio = stat_question_ratio(train_sequences, NUM_QUESTIONS)

# test model
dkt.load("dkt.params")
(sequences, y_truth, y_pred), (auc, acc, rmse) = dkt.eval(test_loader)
print("auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (auc, acc, rmse))

question_perf = get_questions_perf(sequences, y_truth, y_pred, NUM_QUESTIONS)

keys_deleted = []
for k in train_question_ratio.keys():
    if train_question_ratio[k] == 0 or question_perf[k] == -1:
        keys_deleted.append(k)

for k in keys_deleted:
    del train_question_ratio[k]
    del question_perf[k]

draw_scatter_figure(list(train_question_ratio.values()), list(question_perf.values()))

corr_value = corr(list(train_question_ratio.values()), list(question_perf.values()))
print("The coefficient of correlation of frequency and accuracy is %.6f ." % corr_value)
