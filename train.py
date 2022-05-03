from data import x_test, x_train, y_test, y_train
from Neural_Network import *

train(20,x_train[0:2000],y_train[0:2000]) # 20 epochs, train on the first 2000 samples
acc = accuracy(x_test) # Accuracy tested on the test samples
print(acc)
