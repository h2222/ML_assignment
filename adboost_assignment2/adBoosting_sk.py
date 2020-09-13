from numpy import *
from csv import reader

""" 
data pre-processing
"""


def one_hot(set_y):
    for i in range(0, set_y.size):
        if set_y[i] == 'B':
            set_y[i] = -1
        else:
            set_y[i] = 1
    return set_y


with open('wdbc_data.csv', 'rt', encoding='UTF-8') as raw_data:
    readers = reader(raw_data, delimiter=',')
    # print(list(readers)[0])
    dataset = array(list(readers))

    train_set_x = dataset[:300, 2:]
    test_set_x = dataset[300:, 2:]

    train_set_y = one_hot(dataset[:300, 1:2]).reshape(-1)
    test_set_y = one_hot(dataset[300:, 1:2]).reshape(-1)

    # print(train_set_y.shape, type(train_set_x))

"""
use sklearn. AdaBoost model
get error rate
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# initialize model
model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=30),
    n_estimators=103,
    learning_rate=1.5,
    algorithm='SAMME',
    random_state=1
)

# initialize basic variable
N_train_samples = train_set_x.shape[0]
N_epoch = 50
N_batch = 30
N_classes = unique(train_set_y)
score_train = []
score_test = []
epoch = 0


while epoch < N_epoch:
    test_predict_y = []
    train_predict_y = []
    print('epoch:', epoch)

    # random samples
    rand_perm = random.permutation(train_set_x.shape[0])

    mini_batch_index = 0

    while True:
        # mini_batch
        # randomly pick a some number between mini_batch_idx and max batch number
        # and make a list to save those number
        # this list will tell the train_set_x what kind of samples should be trained
        idx_list = rand_perm[mini_batch_index: mini_batch_index + N_batch]

        # print(idx)
        model.fit(train_set_x[idx_list], train_set_y[idx_list])

        # adding the lower bound of mini_batch
        mini_batch_index += N_batch

        # if the lower bound of mini_batch is greater than all samples number
        # break and go to the next epoch
        if mini_batch_index >= N_train_samples:
            break

    # computing error rate and save it
    for test_predict in model.predict(test_set_x):
        test_predict_y.append(test_predict)

    for train_predict in model.predict(train_set_x):
        train_predict_y.append(train_predict)

    score_te = accuracy_score(test_predict_y, test_set_y)
    score_tr = accuracy_score(train_predict_y, train_set_y)

    score_test.append(score_te)
    score_train.append(score_tr)

    epoch += 1

# 1 - accuracy rate =  error rate
score_train = [1.0 - x for x in score_test]
score_test = [1.0 - x for x in score_test]
# print(score_train)


""" 
plot
referring the plot function implemented in adboosting_test
"""
from adboosting_test import plotF

plotF(score_train,
      title='relationship between numIter - ErrorRate (train set. implemented by sklearn)',
      xlabel='Number of iteration',
      ylabel='Error rate',
      ylim=0.5)

plotF(score_test,
      title='relationship between numIter - ErrorRate (test set. implemented by sklearn)',
      xlabel='Number of iteration',
      ylabel='Error rate',
      ylim=0.5)

