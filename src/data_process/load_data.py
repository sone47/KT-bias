# coding: utf-8
# 2021/5/9 @ sone

import itertools
import math

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
    def __init__(self, qa_sequences, seq_len, num_questions):
        self.qa_sequences = qa_sequences
        self.seq_len = seq_len
        self.num_questions = num_questions

    def __len__(self):
        # number of sequences
        return len(self.qa_sequences)

    def __getitem__(self, index):
        qa = self.qa_sequences[index]
        return torch.tensor(qa.tolist())


def __get_data_loader(data_path, seq_len, batch_size, num_questions, shuffle=False):
    handle = DataReader(data_path, seq_len, num_questions)
    qa_data = handle.get_data()
    dataset = DKTDataset(qa_data, seq_len, num_questions)
    data_loader = data.DataLoader(dataset, batch_size, shuffle)
    return data_loader


def get_data_loader(train_data_path, valid_data_path, test_data_path, seq_len, batch_size, num_questions):
    print('loading train data:')
    train_data_loader = __get_data_loader(train_data_path, seq_len, batch_size, num_questions, True)
    print('loading valid data:')
    valid_data_loader = __get_data_loader(valid_data_path, seq_len, batch_size, num_questions, False)
    print('loading test data:')
    test_data_loader = __get_data_loader(test_data_path, seq_len, batch_size, num_questions, False)
    return train_data_loader, valid_data_loader, test_data_loader
