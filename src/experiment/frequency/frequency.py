# coding: utf-8
# 2021/5/11 @ sone

import torch
from src import DKT as Model
from src import config as conf
from src.experiment.utils import Experiment
from src.experiment.frequency.stat import stat_question_freq

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

exp = Experiment(Model, NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, device,
                 conf.dataset, conf.data_dir, conf.dataset_dirname[dataset], model_path)
exp.run(conf.epoch, conf.learning_rate, log_train_file, log_valid_file, stat_question_freq, 'question_frequency',
        conf.train_filename, conf.valid_filename, conf.test_filename)
