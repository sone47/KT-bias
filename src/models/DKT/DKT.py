# coding: utf-8
# 2021/5/7 @ sone

import logging

import numpy as np
import torch
import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers, device):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.n_question = num_questions
        self.n_one_hot = (num_questions + 1) * 2
        self.rnn = nn.LSTM(self.n_one_hot, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)
        self.sig = nn.Sigmoid()
        self.device = device

    def one_hot(self, qa):
        return nn.functional.one_hot(qa, self.n_one_hot)

    def forward(self, q, a) -> torch.Tensor:
        x = self.one_hot(a * self.n_question + q).float()
        out, _ = self.rnn(x)
        res = self.sig(self.fc(out))
        return res


def process_raw_pred(questions, answers, raw_pred, n_question) -> tuple:
    questions = questions[questions < n_question]
    valid_length = len(questions)
    questions = questions[1: valid_length]
    raw_pred = raw_pred[: valid_length - 1]
    pred = raw_pred.gather(1, questions.view(-1, 1)).flatten()
    truth = answers[1: valid_length]
    return truth, pred, questions


class DKT:
    def __init__(self, num_questions, hidden_size, num_layers, device) -> ...:
        super(DKT, self).__init__()
        self.dkt_model = Net(num_questions, hidden_size, num_layers, device).to(device)
        self.device = device
        self.n_question = num_questions

    def train(self, train_data, test_data=None, epoch: int = 5, lr=0.002, train_log_file='', test_log_file='',
              save_filepath=''):
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)
        best_auc = 0

        # prepare logging file
        if train_log_file:
            with open(train_log_file, 'w') as log_tf:
                log_tf.write('epoch, loss\n')
        if test_data and test_log_file:
            with open(test_log_file, 'w') as log_tf:
                log_tf.write('epoch, auc, acc\n')

        for e in range(epoch):
            losses = []
            for q_sequences, a_sequences, features in tqdm.tqdm(train_data, "Epoch %s" % e):
                integrated_pred = self.dkt_model(q_sequences, a_sequences)
                batch_size = q_sequences.size(0)
                loss = torch.Tensor([0.0]).to(self.device)
                for i in range(batch_size):
                    truth, pred, _ = process_raw_pred(q_sequences[i], a_sequences[i], integrated_pred[i], self.n_question)
                    if len(pred) > 0:
                        loss += loss_function(pred, truth.float())
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.mean().item() / batch_size)
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            if train_log_file:
                with open(train_log_file, 'a') as log_tf:
                    log_tf.write('{epoch},{loss: 8.5f}\n'.format(epoch=e, loss=sum(losses) / len(losses)))

            if test_data is not None:
                _, (auc, acc, mse) = self.eval(test_data, False)
                if auc > best_auc:
                    self.save(save_filepath)
                    best_auc = auc
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, MSE: %.6f" % (e, auc, acc, mse))
                if test_log_file:
                    with open(test_log_file, 'a') as log_tf:
                        log_tf.write('{epoch},{auc: 8.5f},{acc:3.3f}\n'.format(epoch=e, auc=auc, acc=100 * acc))

    def eval(self, test_data, evaluation: bool = True) -> tuple:
        if evaluation:
            self.dkt_model.eval()
        sequences = torch.tensor([]).to(self.device)
        all_features = [torch.tensor([]) for _ in range(1)]
        y_pred = torch.Tensor([]).to(self.device)
        y_truth = torch.Tensor([]).to(self.device)

        for question_sequences, answer_sequences, features in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(question_sequences, answer_sequences)
            batch_size = question_sequences.size(0)
            for i in range(batch_size):
                truth, pred, sequence = process_raw_pred(question_sequences[i], answer_sequences[i], integrated_pred[i],
                                                         self.n_question)
                valid_length = len(sequence)

                sequences = torch.cat((sequences, sequence.float()))
                y_pred = torch.cat([y_pred, pred.float()])
                y_truth = torch.cat([y_truth, truth.float()])
                for j, feature in enumerate(features):
                    all_features[j] = torch.cat((all_features[j], feature[i][1: valid_length + 1].float()))

        sequences = sequences.cpu().numpy()
        y_truth = y_truth.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        for i, _ in enumerate(all_features):
            all_features[i] = all_features[i].numpy()

        auc = roc_auc_score(y_truth, y_pred)
        accuracy = accuracy_score(y_truth, y_pred >= 0.5)
        mse = mean_squared_error(y_truth, y_pred)

        return (sequences, all_features, y_truth, y_pred), (auc, accuracy, mse)

    def save(self, filepath) -> ...:
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
