# coding: utf-8
# 2021/5/10 @ sone
import os

dataset_name = {
    'assist2009': 'assistment-2009-2010-skill',
    'assist2012': 'assistment-2012-2013-non-skill',
    'assist2015': 'assistment-2015',
    'assist2017': 'assistment-2017',
}

dataset_dirname = {
    'assist2009': '2009_skill_builder_data_corrected',
    'assist2012': '2012-2013-data-with-predictions-4-final',
    'assist2015': '2015_100_skill_builders_main_problems',
    'assist2017': 'assistment_2017',
}

dataset_filename = {
    'assist2009': 'skill_builder_data_corrected.csv',
    'assist2012': '2012-2013-data-with-predictions-4-final.csv',
    'assist2015': '2015_100_skill_builders_main_problems.csv',
    'assist2017': 'assistment_2017.csv',
}

dataset_key = {
    'assist2009': {
        'order': 'order_id',
        'skill_id': 'skill_id',
        'user_id': 'user_id',
    },
    'assist2012': {
        'order': 'end_time',
        'skill_id': 'skill_id',
        'user_id': 'user_id',
    },
    'assist2015': {
        'order': 'log_id',
        'skill_id': 'sequence_id',
        'user_id': 'user_id',
    },
    'assist2017': {
        'order': 'endTime',
        'skill_id': 'skill',
        'user_id': 'studentId',
    },
}

num_questions = {
    'assist2009': 123,
    'assist2012': 265,
    'assist2015': 100,
    'assist2017': 102,
}

# data directory
data_dir = os.path.join('/my project location.../KT-bias', 'data')
# dataset = 'assist2012'
dataset = 'assist2009'

train_filename = 'train.txt'
valid_filename = 'valid.txt'
test_filename = 'test.txt'

log = 'log'
device = 'cpu'

# hyper-parameters
epoch = 10
learning_rate = 0.05
batch_size = 64
hidden_size = 10
num_layers = 3
seq_len = 100
