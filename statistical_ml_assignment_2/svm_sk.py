from adboosting_test import load_data
from sklearn.svm import SVC
from numpy import asarray

train_set_x, train_set_y, test_set_x, test_set_y = load_data('wdbc_data.csv')

train_set_y = asarray(train_set_y)
print(type(train_set_x))
print(type(train_set_y))

# soft-margin svm classifier
model = SVC(gamma='auto')

model.fit(train_set_x, train_set_y)

predict_y = model.predict(test_set_x).reshape(-1)
predict_y = predict_y.tolist()
print(predict_y)
print(test_set_y)

error = 0
for intem_p, item_y in zip(predict_y, test_set_y):
    if intem_p != item_y:
        error +=1

error_rate = float(error/len(test_set_y))
print(error_rate)