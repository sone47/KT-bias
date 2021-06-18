# coding: utf-8
# 2021/6/18 @ sone

from collections import Counter


def __parse_tl(filepath):
    questions = []
    answers = []
    with open(filepath, 'r') as f:
        for i, l in enumerate(f.readlines()):
            if i % 3 == 1:
                # parse questions line
                questions.extend(list(map(int, l.strip().split(','))))
            elif i % 3 == 2:
                # parse answers line
                answers.extend(list(map(int, l.strip().split(','))))
    return questions, answers


def stat_answer_acc(filepath):
    questions, answers = __parse_tl(filepath)
    c = Counter(questions)
    answer_acc = {}
    for i, q in enumerate(questions):
        if answers[i] == 1:
            answer_acc.setdefault(q, answer_acc.get(q, 0) + 1)

    for k, v in answer_acc.items():
        answer_acc[k] /= c[k]

    return answer_acc


if __name__ == '__main__':
    from src import config as conf
    from os import path
    dataset = conf.dataset
    dataset_name = conf.dataset_name[dataset]
    dataset_dir = path.join(conf.data_dir, conf.dataset_dirname[dataset])
    train_data_path = path.join(dataset_dir, conf.train_filename)
    acc = stat_answer_acc(train_data_path)
