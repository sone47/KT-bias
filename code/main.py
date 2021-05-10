# coding: utf-8
# 2021/5/7 @ sone

from models import DKT
from data_process import get_data_loader


NUM_QUESTIONS = 124
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1
MAX_STEP = 50

dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)

train_data_path = './data/2009_skill_builder_data_corrected/train.txt'
test_data_path = './data/2009_skill_builder_data_corrected/test.txt'
train_loader, test_loader = get_data_loader(train_data_path, test_data_path, MAX_STEP, BATCH_SIZE, NUM_QUESTIONS)

dkt.train(train_loader, epoch=5)
dkt.save("dkt.params")

dkt.load("dkt.params")
auc = dkt.eval(test_loader)
print("auc: %.6f" % auc)
