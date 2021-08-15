import os
import argparse
import sys

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('./')
from src import config as conf


def prepare_dataset_conf(dataset):
    dataset_name = conf.dataset_name[dataset]
    dataset_dir = os.path.join(conf.data_dir, conf.dataset_dirname[dataset])
    dataset_path = os.path.join(dataset_dir, conf.dataset_filename[dataset])
    dataset_key = conf.dataset_key[dataset]
    return dataset_name, dataset_dir, dataset_path, dataset_key


def load_dataset(dataset_path, dataset_name, dataset_key):
    # download raw dataset
    if not os.path.exists(dataset_path):
        from EduData import get_data
        get_data(dataset_name, conf.data_dir)

    # read dataset and select columns
    cols = ['order', 'skill_id', 'user_id', 'correct']
    for i, c in enumerate(cols):
        if c not in dataset_key or not dataset_key[c]:
            print('Key \'%s\' is not exist in dataset %s. It will be removed.' % (c, dataset_name))
            cols.pop(i)
    use_cols = [dataset_key[c] for c in cols]
    data = pd.read_csv(dataset_path, usecols=use_cols) \
        .dropna(axis=0, subset=[dataset_key['skill_id']])

    data = data.rename(columns={dataset_key[c]: c for c in cols})
    data['correct'] = data['correct'].astype('int')

    if 'order' in cols:
        data = data.sort_values('order', ascending=True)
        data['order'] = data['order'].astype('int')
        # compute interval time
        new_data = pd.DataFrame()
        for s in tqdm(data.user_id.unique(), desc='loading dataset'):
            seq = pd.DataFrame(data[data.user_id == s])
            if len(seq) > 1:
                interval_time = seq.loc[:, 'order'].diff()
                interval_time.iloc[0] = interval_time.max()
                interval_time = interval_time.astype('int')
                seq['interval_time'] = interval_time
                new_data = pd.concat([new_data, seq])
        return new_data
    else:
        return data


def v2id(data, key):
    raw_data = data[key].unique()

    data2id = {q: i for i, q in enumerate(raw_data)}
    data[key] = data[key].map(data2id)


# parse mixture to sequences
def parse_all_seq(data, all_seq_id, keys):
    for k in keys:
        if k not in data.columns:
            raise KeyError('Key %s is not in dataset.' % k)

    all_sequences = []
    for seq_id in tqdm(all_seq_id, 'parse sequences'):
        seq = parse_seq(data[data.user_id == seq_id], keys=keys)
        all_sequences.append(seq)
    return all_sequences


def parse_seq(seq, keys):
    data = [seq[k] for k in keys]
    return data


# split data
def train_test_split(all_data, train_size=.7, shuffle=True):
    if shuffle:
        np.random.shuffle(all_data)
    boundary = round(len(all_data) * train_size)
    return all_data[: boundary], all_data[boundary:]


# convert sequences data to triple line txt data
def sequences2lines(seqs, target_path):
    with open(target_path, 'a', encoding='utf8') as f:
        for seq in tqdm(seqs, 'write into file: %s\n' % target_path):
            for values in seq:
                f.write(','.join([str(v) for v in values]) + '\n')


def save_txt_file(train_data, valid_data, test_data, train_path, valid_path, test_path):
    # save triple line format for subsequent tasks
    if not os.path.exists(train_path):
        sequences2lines(train_data, train_path)
    if not os.path.exists(valid_path):
        sequences2lines(valid_data, valid_path)
    if not os.path.exists(test_path):
        sequences2lines(test_data, test_path)


def decode_keys_arg(keys_str):
    keys = keys_str.split(',')

    if 'skill_id' in keys and keys.index('skill_id') != 0:
        keys.remove('skill_id')
    if 'skill_id' not in keys:
        keys.insert(0, 'skill_id')

    if 'correct' in keys and keys.index('correct') != 1:
        keys.remove('correct')
    if 'correct' not in keys:
        keys.insert(1, 'correct')

    return keys


def main():
    parser = argparse.ArgumentParser(description='Script to prepare txt data')
    parser.add_argument('--dataset', type=str, default='assist2009')
    parser.add_argument('--train_file', type=str, default='train.txt', help='train data saved file name')
    parser.add_argument('--valid_file', type=str, default='valid.txt')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--keys', type=str, default='skill_id,correct', help='keys the bias analysis requires')
    params = parser.parse_args()

    dataset_name, dataset_dir, dataset_path, dataset_key = prepare_dataset_conf(params.dataset)
    dataset = load_dataset(dataset_path, dataset_name, dataset_key)

    # count the number of question
    num_question = len(dataset.skill_id.unique())
    print("number of skills: %d" % num_question)

    v2id(dataset, 'skill_id')

    # [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
    sequences = parse_all_seq(dataset, dataset.user_id.unique(), keys=decode_keys_arg(params.keys))

    # train: valid: test = 6: 2: 2
    train_sequences, test_sequences = train_test_split(sequences, 0.8)
    train_sequences, valid_sequences = train_test_split(train_sequences, 0.75, False)

    train_data_path = os.path.join(dataset_dir, params.train_file)
    valid_data_path = os.path.join(dataset_dir, params.valid_file)
    test_data_path = os.path.join(dataset_dir, params.test_file)
    save_txt_file(train_sequences, valid_sequences, test_sequences, train_data_path, valid_data_path, test_data_path)


if __name__ == '__main__':
    main()
