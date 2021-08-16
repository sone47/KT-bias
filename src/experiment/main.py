# coding: utf-8
# 2021/6/18 @ sone

import torch
from os import path
from src import config as conf
from src.experiment.utils import Experiment

if __name__ == '__main__':
    # parse argument from terminal
    import argparse

    parser = argparse.ArgumentParser(description='Script to run experiment')
    parser.add_argument('--model', type=str, default='dkt')
    parser.add_argument('--dataset', type=str, default='assist2009')
    parser.add_argument('--train_file', type=str, default='train.txt', help='train data saved file name')
    parser.add_argument('--valid_file', type=str, default='valid.txt')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--exp_property', type=str, default='interval_time', help='bias property test')
    parser.add_argument('--n_unit', type=int, default=2, help='the number of features in txt file')
    params = parser.parse_args()

    dataset = params.dataset
    model_name = params.model.lower()
    n_unit = params.n_unit

    n_question = conf.num_questions[dataset]
    n_problem = conf.n_problem[dataset]
    batch_size = conf.batch_size
    seq_len = conf.seq_len
    device = torch.device(conf.device)

    # prepare data file path
    train_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], params.train_file)
    valid_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], params.valid_file)
    test_data_path = path.join(conf.data_dir, conf.dataset_dirname[dataset], params.test_file)
    # prepare log file path
    train_log_file = conf.log + '-train-' + dataset + '.log'
    valid_log_file = conf.log + '-valid-' + dataset + '.log'

    if model_name == 'dkt':
        from src.models import DKT
        from src.models.DKT import load_data

        hidden_size = 256
        n_layer = 1

        model = DKT(n_question, hidden_size, n_layer, device=device)
        model_path = 'dkt-' + dataset + '.params'
        train_data, valid_data, test_data = load_data(train_data_path, valid_data_path, test_data_path,
                                                      seq_len, batch_size, n_question, device=device,
                                                      n_unit=n_unit)
    elif model_name == 'akt':
        from src.models import AKT
        from src.models.AKT import DATA, PID_DATA

        n_blocks = 1
        d_model = 256
        dropout = 0.05
        kq_same = 1
        l2 = 1e-5
        maxgradnorm = -1

        model = AKT(n_question, n_problem, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm)
        model_path = 'akt-' + dataset + '.params'
        if n_problem > 0:
            dat = PID_DATA(n_question=n_question, seqlen=seq_len, separate_char=',')
        else:
            dat = DATA(n_question=n_question, seqlen=seq_len, separate_char=',')
        train_data = dat.load_data(train_data_path, n_unit)
        valid_data = dat.load_data(valid_data_path, n_unit)
        test_data = dat.load_data(test_data_path, n_unit)
    else:
        raise ValueError('No model named %s.' % model_name)

    exp = Experiment(model, model_path)

    if params.exp_property == 'interval_time':
        from interval_time import output_processor
    elif params.exp_property == 'answer_accuracy':
        from answer_acc import output_processor
    elif params.exp_property == 'frequency':
        from frequency import output_processor
    else:
        raise KeyError('Wrong argument exp_property with %s' % params.exp_property)

    exp.run(train_data, valid_data, test_data, conf.epoch, conf.learning_rate, train_log_file, valid_log_file,
            output_processor)
