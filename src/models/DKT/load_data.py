# coding: utf-8
# 2021/5/9 @ sone

import math
from os import path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class DataReader:
    def __init__(self, seq_len, num_question, device, n_unit=2, separate_char=','):
        self.seq_len = seq_len
        self.n_question = num_question
        self.device = device
        self.n_unit = n_unit
        self.separate_char = separate_char

    def get_data(self, data_path):
        has_feature = self.n_unit > 2
        q_data = np.array([])
        a_data = np.array([])
        features_data = np.array([])
        features = []
        with open(data_path, 'r') as d:
            for lineId, line in tqdm(enumerate(d), desc='loading data'):
                line = line.strip()
                if lineId % self.n_unit == 0:
                    Q = line.split(self.separate_char)
                    if len(Q[len(Q) - 1]) == 0:
                        Q = Q[:-1]
                elif lineId % self.n_unit == 1:
                    A = line.split(self.separate_char)
                    if len(A[len(A) - 1]) == 0:
                        A = A[:-1]
                else:
                    feature = line.split(self.separate_char)
                    if len(feature[len(feature) - 1]) == 0:
                        feature = feature[:-1]
                    features.append(feature)

                if lineId % self.n_unit == self.n_unit - 1:
                    length = len(Q)
                    question_sequence = np.array(Q, 'int')
                    answer_sequence = np.array(A, 'int')
                    if has_feature:
                        features = np.array(features, 'int')

                    mod = 0 if length % self.seq_len == 0 else (self.seq_len - length % self.seq_len)
                    if length % self.seq_len <= 5:
                        answer_sequence = answer_sequence[: -(length % self.seq_len)]
                        question_sequence = question_sequence[: -(length % self.seq_len)]
                        if has_feature:
                            features = features[:, : -(length % self.seq_len)]
                        mod = 0

                    zeros = np.zeros(mod)

                    question_sequence = np.append(question_sequence, zeros)
                    answer_sequence = np.append(answer_sequence, zeros)
                    if has_feature:
                        features = np.concatenate((features, np.zeros((len(features), mod))), axis=1)

                    q_data = np.append(q_data, question_sequence)
                    a_data = np.append(a_data, answer_sequence)
                    features_data = np.append(features_data, features)

                    features = []

        q_data = q_data.reshape([-1, self.seq_len]).astype(int)
        a_data = a_data.reshape([-1, self.seq_len]).astype(int)
        if has_feature:
            features_data = features_data.reshape([self.n_unit - 2, -1, self.seq_len])
        return q_data, a_data, features_data


class DKTDataset(Dataset):
    def __init__(self, q, a, features):
        self.q = q
        self.a = a
        self.features = features

    def __len__(self):
        return len(self.q)

    def __getitem__(self, index):
        q = self.q[index]
        a = self.a[index]
        return torch.tensor(q), torch.tensor(a), [feature[index] for feature in self.features]


def __get_data_loader(handler, data_path, batch_size, shuffle=False):
    q, a, interval = handler.get_data(data_path)
    dataset = DKTDataset(q, a, interval)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle)
    return data_loader


def load_data(train_data_path, valid_data_path, test_data_path, seq_len, batch_size, num_questions, n_unit, device):
    handler = DataReader(seq_len, num_questions, device=device, n_unit=n_unit)
    train_data_loader, valid_data_loader, test_data_loader = None, None, None
    if path.isfile(train_data_path):
        print('loading train data:')
        train_data_loader = __get_data_loader(handler, train_data_path, batch_size, shuffle=True)
    if path.isfile(valid_data_path):
        print('loading valid data:')
        valid_data_loader = __get_data_loader(handler, valid_data_path, batch_size, shuffle=False)
    if path.isfile(test_data_path):
        print('loading test data:')
        test_data_loader = __get_data_loader(handler, test_data_path, batch_size, shuffle=False)
    return train_data_loader, valid_data_loader, test_data_loader
