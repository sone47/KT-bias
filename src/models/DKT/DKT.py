# coding: utf-8
# 2021/5/7 @ sone

import logging

import numpy as np
import torch
import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


class Net(nn.Module):
    def __init__(self, num_questions, d_qa_vec, hidden_size, num_layers):
        super(Net, self).__init__()
        self.qa_emb = nn.Embedding(num_questions * 2, d_qa_vec)
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.LSTM(d_qa_vec, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x) -> torch.Tensor:
        h0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim),
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        )
        out, _ = self.rnn(self.qa_emb(x), h0)
        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(question_matrix, raw_pred, num_questions: int) -> tuple:
    question_matrix = question_matrix[question_matrix > 0]
    valid_length = len(question_matrix)
    valid_questions = (question_matrix % num_questions)[1:]
    if valid_length == 0:
        raw_pred = raw_pred[0: 0]
    else:
        raw_pred = raw_pred[: valid_length - 1]
    pred = raw_pred.gather(1, valid_questions.view(-1, 1)).flatten()
    truth = (question_matrix // num_questions)[1: valid_length]
    return truth, pred, valid_questions


class DKT:
    def __init__(self, num_questions, d_qa_vec, hidden_size, num_layers, device) -> ...:
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = Net(num_questions, d_qa_vec, hidden_size, num_layers).to(device)
        self.device = device

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, train_log_file='', test_log_file=''):
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)
        sequences = np.array([], int)

        # prepare logging file
        if train_log_file:
            with open(train_log_file, 'w') as log_tf:
                log_tf.write('epoch, loss\n')
        if test_data and test_log_file:
            with open(test_log_file, 'w') as log_tf:
                log_tf.write('epoch, auc, acc\n')

        for e in range(epoch):
            losses = []
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                integrated_pred = self.dkt_model(batch)
                batch_size = batch.shape[0]
                loss = torch.Tensor([0.0])
                for student in range(batch_size):
                    truth, pred, sequence = process_raw_pred(batch[student], integrated_pred[student],
                                                             self.num_questions)
                    if len(pred) > 0:
                        loss += loss_function(pred, truth.float())
                        sequences = np.concatenate((sequences, sequence))
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            if train_log_file:
                with open(train_log_file, 'a') as log_tf:
                    log_tf.write('{epoch},{loss: 8.5f}\n'.format(epoch=e, loss=sum(losses)/len(losses)))

            if test_data is not None:
                _, (auc, acc, rmse) = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (e, auc, acc, rmse))
                if test_log_file:
                    with open(test_log_file, 'a') as log_tf:
                        log_tf.write('{epoch},{auc: 8.5f},{acc:3.3f}\n'.format(epoch=e, auc=auc, acc=100 * acc))

        return sequences

    def eval(self, test_data) -> tuple:
        self.dkt_model.eval()
        sequences = np.array([], int)
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])

        for batch in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                truth, pred, sequence = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)

                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
                sequences = np.concatenate((sequences, sequence))

        y_truth = y_truth.detach().numpy()
        y_pred = y_pred.detach().numpy()

        return (
            sequences,
            y_truth,
            y_pred
        ), (
            roc_auc_score(y_truth, y_pred),
            accuracy_score(y_truth, y_pred >= 0.5),
            np.sqrt(mean_squared_error(y_truth, y_pred))
        )

    def save(self, filepath) -> ...:
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)