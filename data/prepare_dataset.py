from EduData import get_data
import os

import random
import pandas as pd
import tqdm

dataset_dir = '2009_skill_builder_data_corrected'
dataset_filename = 'skill_builder_data_corrected.csv'
dataset_path = './' + dataset_dir + '/' + dataset_filename

# download raw dataset
if not os.path.exists(dataset_path):
    get_data('assistment-2009-2010-skill', './')

# read dataset and select columns
data = pd.read_csv(
    dataset_path,
    usecols=['order_id', 'user_id', 'sequence_id', 'skill_id', 'correct']
).fillna(0)

# convert skill id to consecutive integers id
raw_question = data.skill_id.unique().tolist()

question2id = {q: i for i, q in enumerate(raw_question)}
data['skill_id'] = data['skill_id'].map(question2id)

print("number of skills: %d" % len(raw_question))


# parse mixture to sequences
def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.user_id == student_id])
        all_sequences.extend(student_sequence)
    return all_sequences


def parse_student_seq(student):
    student = student.drop_duplicates(subset='order_id')
    sequence_ids = student.sequence_id.unique()
    res_sequences = []
    for seq_id in sequence_ids:
        seq = student[student.sequence_id == seq_id].sort_values('order_id')
        questions = seq.skill_id.tolist()
        answers = seq.correct.tolist()
        res_sequences.append((questions, answers))
    return res_sequences


# [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
sequences = parse_all_seq(data.user_id.unique())


# split data
def train_test_split(all_data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(all_data)
    boundary = round(len(all_data) * train_size)
    return all_data[: boundary], all_data[boundary:]


train_sequences, test_sequences = train_test_split(sequences)


# convert sequences data to triple line txt data
def sequences2tl(seqs, target_path):
    with open(target_path, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(seqs, 'write into file: '):
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')


# save triple line format for other tasks
train_data_path = './' + dataset_dir + '/train.txt'
test_data_path = './' + dataset_dir + '/test.txt'
if not os.path.exists(train_data_path):
    sequences2tl(train_sequences, train_data_path)
if not os.path.exists(test_data_path):
    sequences2tl(test_sequences, test_data_path)
