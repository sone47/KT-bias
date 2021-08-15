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
        q_data = np.array([])
        a_data = np.array([])
        interval_data = np.array([])
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
                    interval_time = line.split(self.separate_char)
                    if len(interval_time[len(interval_time) - 1]) == 0:
                        interval_time = interval_time[:-1]

                if lineId % self.n_unit == self.n_unit - 1:
                    length = len(Q)
                    question_sequence = np.array(Q, 'int')
                    answer_sequence = np.array(A, 'int')
                    interval_time = np.array(interval_time, 'int')

                    mod = 0 if length % self.seq_len == 0 else (self.seq_len - length % self.seq_len)
                    if length % self.seq_len <= 5:
                        answer_sequence = answer_sequence[: -(length % self.seq_len)]
                        question_sequence = question_sequence[: -(length % self.seq_len)]
                        interval_time = interval_time[: -(length % self.seq_len)]
                        zeros = np.zeros(0)
                    else:
                        zeros = np.zeros(mod)
                    question_sequence = np.append(question_sequence, zeros + self.n_question)
                    answer_sequence = np.append(answer_sequence, zeros)
                    interval_time = np.append(interval_time, zeros)

                    q_data = np.append(q_data, question_sequence)
                    a_data = np.append(a_data, answer_sequence)
                    interval_data = np.append(interval_data, interval_time)

        q_data = q_data.reshape([-1, self.seq_len]).astype(int)
        a_data = a_data.reshape([-1, self.seq_len]).astype(int)
        interval_data = interval_data.reshape([-1, self.seq_len])
        return q_data, a_data, interval_data


class DKTDataset(Dataset):
    def __init__(self, q, a, interval):
        self.q = q
        self.a = a
        self.interval = interval

    def __len__(self):
        return len(self.q)

    def __getitem__(self, index):
        q = self.q[index]
        a = self.a[index]
        interval = self.interval[index]
        return torch.tensor(q), torch.tensor(a), interval


def __get_data_loader(handler, data_path, batch_size, shuffle=False):
    q, a, interval = handler.get_data(data_path)
    dataset = DKTDataset(q, a, interval)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle)
    return data_loader


def get_data_loader(train_data_path, valid_data_path, test_data_path, seq_len, batch_size, num_questions, device):
    handler = DataReader(seq_len, num_questions, device=device, n_unit=3)
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
