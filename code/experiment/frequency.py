# coding: utf-8
# 2021/5/11 @ sone

from sklearn.metrics import accuracy_score

from code import DKT, get_data_loader
from code import config as conf

dataset = conf.dataset

NUM_QUESTIONS = conf.num_questions[dataset]
BATCH_SIZE = conf.batch_size
HIDDEN_SIZE = conf.hidden_size
NUM_LAYERS = conf.num_layers
MAX_STEP = conf.max_step

dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)

train_data_path = './data/2009_skill_builder_data_corrected/train.txt'
valid_data_path = './data/2009_skill_builder_data_corrected/valid.txt'
test_data_path = './data/2009_skill_builder_data_corrected/test.txt'
train_loader, valid_loader, test_loader = get_data_loader(train_data_path, valid_data_path, test_data_path, MAX_STEP,
                                                          BATCH_SIZE, NUM_QUESTIONS)

dkt.train(train_loader, valid_loader, epoch=5)
dkt.save("dkt.params")

dkt.load("dkt.params")
(sequences, y_truth, y_pred), (auc, acc, rmse) = dkt.eval(test_loader)
print("auc: %.6f, accuracy: %.6f, RMSE: %.6f" % (auc, acc, rmse))


def get_questions_perf(question_sequences, truth, pred):
    questions_perf = {}
    for i in range(NUM_QUESTIONS):
        index = question_sequences == i
        q = question_sequences[index]
        t = truth[index]
        p = pred[index]
        if len(q) > 0:
            accu = accuracy_score(t, p >= 0.5)
            questions_perf[i] = accu
        else:
            # -1 represents that the question has never been predicted
            questions_perf[i] = -1
    return questions_perf


perf = get_questions_perf(sequences, y_truth, y_pred)
print(perf)
