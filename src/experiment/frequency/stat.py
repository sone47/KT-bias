# coding: utf-8
# 2021/6/19 @ sone

from collections import Counter


def __parse_tl(filepath):
    questions = []
    with open(filepath, 'r') as f:
        for i, l in enumerate(f.readlines()):
            if i % 3 == 1:
                # parse questions line
                questions.extend(list(map(int, l.strip().split(','))))
    for i, q in enumerate(questions):
        questions[i] = q - 1
    return questions


def stat_question_freq(filepath):
    questions = __parse_tl(filepath)
    freq = Counter(questions)
    sum_question = sum(freq.values())
    for k in freq.keys():
        freq[k] /= sum_question
    return freq


if __name__ == '__main__':
    from src import config as conf
    from os import path
    dataset = conf.dataset
    dataset_name = conf.dataset_name[dataset]
    dataset_dir = path.join(conf.data_dir, conf.dataset_dirname[dataset])
    data_path = path.join(dataset_dir, conf.test_filename)
    print(stat_question_freq(data_path))
