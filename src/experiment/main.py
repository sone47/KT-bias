# coding: utf-8
# 2021/6/18 @ sone

import torch
from src import DKT
from src import config as conf
from src.experiment.utils import Experiment

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

if __name__ == '__main__':
    # parse argument from terminal
    import argparse
    parser = argparse.ArgumentParser(description='Script to run experiment')
    parser.add_argument('--dataset', type=str, default='assist2009')
    parser.add_argument('--train_file', type=str, default='train.txt', help='train data saved file name')
    parser.add_argument('--valid_file', type=str, default='valid.txt')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--exp_property', type=str, default='interval_time')
    params = parser.parse_args()

    exp = Experiment(DKT, NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, device,
                     conf.dataset, conf.data_dir, conf.dataset_dirname[dataset], model_path)

    if params.exp_property == 'interval_time':
        from interval_time import output_processor
    elif params.exp_property == 'answer_accuracy':
        from answer_acc import output_processor
    elif params.exp_property == 'frequency':
        from frequency import output_processor

    exp.run(conf.epoch, conf.learning_rate, train_log_file, valid_log_file,
            params.train_file, params.valid_file, params.test_file, output_processor)
