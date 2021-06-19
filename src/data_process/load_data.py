# coding: utf-8
# 2021/5/9 @ sone

import itertools
import math
from os import path

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import tqdm


class DataReader:
    def __init__(self, data_path, seq_len, num_question):
        self.data_path = data_path
        self.seq_len = seq_len
        self.n_question = num_question

    def get_data(self):
        qa_data = np.array([])
        num_file_line = sum([1 for i in open(self.data_path, 'r')])
        with open(self.data_path, 'r') as d:
            for length, Q, A in tqdm.tqdm(itertools.zip_longest(*[d] * 3), desc='loading data',
                                          total=math.ceil(num_file_line / 3)):
                length = int(length)
                question_sequence = np.array(Q.strip().split(',')).astype(int)
                answer_sequence = np.array(A.strip().split(',')).astype(int)
                mod = 0 if length % self.seq_len == 0 else (self.seq_len - length % self.seq_len)
                fill_content = np.zeros(mod)
                answer_sequence = np.append(answer_sequence * self.n_question + question_sequence, fill_content)
                qa_data = np.append(qa_data, answer_sequence).astype(int)

        return qa_data.reshape([-1, self.seq_len])


class DKTDataset(Dataset):
    def __init__(self, qa_sequences, seq_len, num_questions, device):
        self.qa_sequences = qa_sequences
        self.seq_len = seq_len
        self.num_questions = num_questions
        self.device = device

    def __len__(self):
        # number of sequences
        return len(self.qa_sequences)

    def __getitem__(self, index):
        qa = self.qa_sequences[index]
        return torch.tensor(qa).to(self.device)


def __get_data_loader(data_path, seq_len, batch_size, num_questions, device, shuffle=False):
    handle = DataReader(data_path, seq_len, num_questions)
    qa_data = handle.get_data()
    dataset = DKTDataset(qa_data, seq_len, num_questions, device)
    data_loader = data.DataLoader(dataset, batch_size, shuffle)
    return data_loader


def get_data_loader(train_data_path, valid_data_path, test_data_path, seq_len, batch_size, num_questions, device):
    train_data_loader, valid_data_loader, test_data_loader = None, None, None
    if path.isfile(train_data_path):
        print(train_data_path)
        print('loading train data:')
        train_data_loader = __get_data_loader(train_data_path, seq_len, batch_size, num_questions,
                                              shuffle=True,device=device)
    if path.isfile(valid_data_path):
        print('loading valid data:')
        valid_data_loader = __get_data_loader(valid_data_path, seq_len, batch_size, num_questions,
                                              shuffle=False, device=device)
    if path.isfile(test_data_path):
        print('loading test data:')
        test_data_loader = __get_data_loader(test_data_path, seq_len, batch_size, num_questions,
                                             shuffle=False, device=device)
    return train_data_loader, valid_data_loader, test_data_loader
