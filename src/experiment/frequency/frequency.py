# coding: utf-8
# 2021/5/11 @ sone

import torch
from collections import Counter
from src import DKT as Model
from src import config as conf
from src.experiment.utils import Experiment, divide_groups, calculate_all_bias, calculate_all_group_performance

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
SEQ_LEN = conf.seq_len
device = torch.device(conf.device)
model_path = 'dkt-' + dataset + '.params'

# prepare log file
log_train_file = conf.log + '-train.log'
log_valid_file = conf.log + '-valid.log'


def calculate_question_freq(questions):
    freq = Counter(questions)
    sum_question = sum(freq.values())
    for k in freq.keys():
        freq[k] /= sum_question
    return freq


def output_processor(questions, _, truth, pred):
    freq = calculate_question_freq(questions)
    result_data = [{
        'freq': freq[q],
        'truth': truth[i],
        'pred': pred[i],
    } for i, q in enumerate(questions)]

    # divide two groups
    groups = divide_groups(result_data, lambda x: x['freq'], 0.1)
    # calculate bias
    bias = calculate_all_bias((groups[0], groups[-1]))
    # calculate all groups performance
    accuracy, auc, mse = calculate_all_group_performance(groups)

    print("The bias value (acc, auc, mse) of question frequency is %s." % str(bias))
    print('accuracy', accuracy)
    print('auc', auc)
    print('mse', mse)


exp = Experiment(Model, NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, device,
                 conf.dataset, conf.data_dir, conf.dataset_dirname[dataset], model_path)
exp.run(conf.epoch, conf.learning_rate, log_train_file, log_valid_file,
        conf.train_filename, conf.valid_filename, conf.test_filename, output_processor)
