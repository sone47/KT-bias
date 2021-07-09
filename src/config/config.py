# coding: utf-8
# 2021/5/10 @ sone
import os

dataset_name = {
    'assist2009': 'assistment-2009-2010-skill',
    'assist2012': 'assistment-2012-2013-non-skill',
    'assist2015': 'assistment-2015'
}

dataset_dirname = {
    'assist2009': '2009_skill_builder_data_corrected',
    'assist2012': '2012-2013-data-with-predictions-4-final',
    'assist2015': '2015_100_skill_builders_main_problems'
}

dataset_filename = {
    'assist2009': 'skill_builder_data_corrected.csv',
    'assist2012': '2012-2013-data-with-predictions-4-final.csv',
    'assist2015': '2015_100_skill_builders_main_problems.csv'
}

dataset_key = {
    'assist2009': {
        'order': 'order_id',
    },
    'assist2012': {
        'order': 'end_time',
    },
    'assist2015': {
        'order': 'end_time',
    }
}

num_questions = {
    'assist2009': 123,
    'assist2012': 265,
    'assist2015': 0,  # placeholder
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
batch_size = 64
hidden_size = 10
num_layers = 2
seq_len = 100
