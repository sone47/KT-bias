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

dataset_order_field = {
    'assist2009': 'order_id',
    'assist2012': 'end_time',
    'assist2015': 'end_time'
}

num_questions = {
    'assist2009': 124,
    'assist2012': 266,
    'assist2015': 0
}

# data directory
data_dir = os.path.join('/my project location.../KT-bias', 'data')
# dataset = 'assist2012'
dataset = 'assist2009'

train_filename = 'train.txt'
valid_filename = 'valid.txt'
test_filename = 'test.txt'

batch_size = 64
hidden_size = 10
num_layers = 2
max_step = 50

log = 'log'
