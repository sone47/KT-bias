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
    def __init__(self, data_path, max_step):
        self.data_path = data_path
        self.max_step = max_step

    def get_data(self):
        question_sequences = np.array([])
        answer_sequences = np.array([])
        num_file_line = sum([1 for i in open(self.data_path, 'r')])
        with open(self.data_path, 'r') as d:
            for length, ques, ans in tqdm.tqdm(itertools.zip_longest(*[d] * 3), desc='loading data', total=math.ceil(num_file_line / 3)):
                length = int(length)
                ques = np.array(ques.strip().split(',')).astype(np.int)
                ans = np.array(ans.strip().split(',')).astype(np.int)
                mod = 0 if length % self.max_step == 0 else (self.max_step - length % self.max_step)
                fill_content = np.zeros(mod) - 1
                ques = np.append(ques, fill_content)
                ans = np.append(ans, fill_content)
                question_sequences = np.append(question_sequences, ques).astype(np.int)
                answer_sequences = np.append(answer_sequences, ans).astype(np.int)
        return question_sequences.reshape([-1, self.max_step]), answer_sequences.reshape([-1, self.max_step])


class DKTDataset(Dataset):
    def __init__(self, ques, ans, max_step, num_questions):
        self.ques = ques
        self.ans = ans
        self.max_step = max_step
        self.num_questions = num_questions

    def __len__(self):
        # number of students
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        one_hot_data = self.one_hot(questions, answers)
        return torch.FloatTensor(one_hot_data.tolist())

    def one_hot(self, questions, answers):
        result = np.zeros(shape=[self.max_step, 2 * self.num_questions])
        for i in range(self.max_step):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + self.num_questions] = 1
        return result


def __get_data_loader(data_path, max_step, batch_size, num_questions, shuffle=False):
    handle = DataReader(data_path, max_step)
    ques, ans = handle.get_data()
    dataset = DKTDataset(ques, ans, max_step, num_questions)
    data_loader = data.DataLoader(dataset, batch_size, shuffle)
    return data_loader


def get_data_loader(train_data_path, valid_data_path, test_data_path, max_step, batch_size, num_questions):
    print('loading train data:')
    train_data_loader = __get_data_loader(train_data_path, max_step, batch_size, num_questions, True)
    print('loading valid data:')
    valid_data_loader = __get_data_loader(valid_data_path, max_step, batch_size, num_questions, False)
    print('loading test data:')
    test_data_loader = __get_data_loader(test_data_path, max_step, batch_size, num_questions, False)
    return train_data_loader, valid_data_loader, test_data_loader
