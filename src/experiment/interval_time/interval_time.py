# coding: utf-8
# 2021/8/8 @ sone

import torch
import numpy as np
from src import DKT
from src import config as conf
from src.experiment.utils import Experiment, divide_groups, calculate_all_bias, calculate_accuracy, calculate_auc, calculate_mse

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
SEQ_LEN = conf.seq_len
device = torch.device(conf.device)
model_path = 'dkt-' + dataset + '.params'

# prepare log file
train_log_file = conf.log + '-train.log'
valid_log_file = conf.log + '-valid.log'


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


def output_processor(sequence, features, truth, pred):
    result_data = [{
        'skill_id': q,
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


exp = Experiment(DKT, NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, device,
                 conf.dataset, conf.data_dir, conf.dataset_dirname[dataset], model_path)
exp.run(conf.epoch, conf.learning_rate, train_log_file, valid_log_file,
        conf.train_filename, conf.valid_filename, conf.test_filename,
        output_processor)
