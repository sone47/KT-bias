import os

import random
import pandas as pd
import tqdm

from src import config as conf

dataset = conf.dataset
dataset_name = conf.dataset_name[dataset]
dataset_dir = os.path.join(conf.data_dir, conf.dataset_dirname[dataset])
dataset_path = os.path.join(dataset_dir, conf.dataset_filename[dataset])
order_field = conf.dataset_order_field[dataset]

# download raw dataset
if not os.path.exists(dataset_path):
    from EduData import get_data
    get_data(dataset_name, conf.data_dir)

# read dataset and select columns
data = pd.read_csv(
    dataset_path,
    usecols=[order_field, 'user_id', 'sequence_id', 'skill_id', 'correct']
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
    student = student.drop_duplicates(subset=order_field)
    sequence_ids = student.sequence_id.unique()
    res_sequences = []
    for seq_id in sequence_ids:
        seq = student[student.sequence_id == seq_id].sort_values(order_field)
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


# train: valid: test = 6: 2: 2
train_sequences, test_sequences = train_test_split(sequences, 0.8)
train_sequences, valid_sequences = train_test_split(train_sequences, 0.75, False)


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
train_data_path = os.path.join(dataset_dir, conf.train_filename)
valid_data_path = os.path.join(dataset_dir, conf.valid_filename)
test_data_path = os.path.join(dataset_dir, conf.test_filename)
if not os.path.exists(train_data_path):
    sequences2tl(train_sequences, train_data_path)
if not os.path.exists(valid_data_path):
    sequences2tl(valid_sequences, valid_data_path)
if not os.path.exists(test_data_path):
    sequences2tl(test_sequences, test_data_path)

# one-hot

# MAX_STEP = 50
# NUM_QUESTIONS = num_skill
#
#
# def encode_one_hot(sequences, max_step, num_questions):
#     question_sequences = np.array([])
#     answer_sequences = np.array([])
#     one_hot_result = []
#
#     for questions, answers in tqdm.tqdm(sequences, 'convert to one-hot format: '):
#         length = len(questions)
#         # append questions' and answers' length to an integer multiple of max_step
#         mod = 0 if length % max_step == 0 else (max_step - length % max_step)
#         fill_content = np.zeros(mod) - 1
#         questions = np.append(questions, fill_content)
#         answers = np.append(answers, fill_content)
#         # one hot
#         q_seqs = questions.reshape([-1, max_step])
#         a_seqs = answers.reshape([-1, max_step])
#         for (i, q_seq) in enumerate(q_seqs):
#             one_hot = np.zeros(shape=[max_step, 2 * num_questions])
#             for j in range(max_step):
#                 index = int(q_seq[j] if a_seqs[i][j] > 0 else q_seq[j] + num_questions)
#                 one_hot[j][index] = 1
#             one_hot_result = np.append(one_hot_result, one_hot)
#
#     return one_hot_result.reshape(-1, max_step, 2 * num_questions)
#
#
# train_one_hot_path = './' + dataset_dir + '/train_data.npy'
# test_one_hot_path = './' + dataset_dir + '/test_data.npy'
#
# # save one_hot data
# if not os.path.exists(train_one_hot_path):
#     train_data = encode_one_hot(train_sequences, MAX_STEP, NUM_QUESTIONS)
#     np.save(train_one_hot_path, train_data)
# if not os.path.exists(test_one_hot_path):
#     test_data = encode_one_hot(test_sequences, MAX_STEP, NUM_QUESTIONS)
#     np.save(test_one_hot_path, test_data)
