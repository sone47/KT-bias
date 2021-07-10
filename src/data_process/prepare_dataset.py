import os

import random
import pandas as pd
import tqdm

from src import config as conf

dataset = conf.dataset
dataset_name = conf.dataset_name[dataset]
dataset_dir = os.path.join(conf.data_dir, conf.dataset_dirname[dataset])
dataset_path = os.path.join(dataset_dir, conf.dataset_filename[dataset])
dataset_key = conf.dataset_key[dataset]

# download raw dataset
if not os.path.exists(dataset_path):
    from EduData import get_data
    get_data(dataset_name, conf.data_dir)

# read dataset and select columns
data = pd.read_csv(
    dataset_path,
    usecols=[dataset_key['order'], 'user_id', dataset_key['skill_id'], 'correct']
).dropna(axis=0, subset=[dataset_key['skill_id']])
data = data.rename(columns={
    dataset_key['order']: 'order_id',
    dataset_key['skill_id']: 'skill_id',
})
data['correct'] = data['correct'].astype('int')


# count the number of question
num_question = len(data.skill_id.unique())
print("number of skills: %d" % num_question)

# convert skill id to consecutive integers id
raw_question = data.skill_id.unique().tolist()

question2id = {q: i + 1 for i, q in enumerate(raw_question)}
data['skill_id'] = data['skill_id'].map(question2id)


# parse mixture to sequences
def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence'):
        q, a = parse_student_seq(data[data.user_id == student_id])
        all_sequences.append((q, a))
    return all_sequences


def parse_student_seq(student):
    student = student.drop_duplicates(subset='order_id')
    seq = student.sort_values('order_id')
    questions = seq.skill_id.tolist()
    answers = seq.correct.tolist()
    return questions, answers


# [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
sequences = parse_all_seq(data.user_id.unique())


# split data
def train_test_split(all_data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(all_data)
    boundary = round(len(all_data) * train_size)
    return all_data[: boundary], all_data[boundary:]


# train: valid: test = 6: 2: 2
train_sequences, test_sequences = train_test_split(sequences, 0.8)
train_sequences, valid_sequences = train_test_split(train_sequences, 0.75, False)


# convert sequences data to triple line txt data
def sequences2tl(seqs, target_path):
    with open(target_path, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(seqs, 'write into file: %s\n' % target_path):
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')


# save triple line format for other tasks
train_data_path = os.path.join(dataset_dir, conf.train_filename)
valid_data_path = os.path.join(dataset_dir, conf.valid_filename)
test_data_path = os.path.join(dataset_dir, conf.test_filename)
if not os.path.exists(train_data_path):
    sequences2tl(train_sequences, train_data_path)
if not os.path.exists(valid_data_path):
    sequences2tl(valid_sequences, valid_data_path)
if not os.path.exists(test_data_path):
    sequences2tl(test_sequences, test_data_path)
