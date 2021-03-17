import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
start_time = time.time()
data = pd.read_csv("student-mat100k.csv", sep=";")

data_length = len(data)
train_X = data.head(int(data_length * 0.9))[["G1", "G2", "absences", "freetime"]]
train_Y = data.head(int(data_length * 0.9))[["G3"]]
train_X.to_numpy()
n, m = train_X.shape
X0 = np.ones((n,1))
train_X = np.hstack((X0, train_X))
train_Y.to_numpy()
thetas = np.linalg.inv(train_X.T.dot(train_X)).dot(train_X.T).dot(train_Y) # normal equation
print(thetas)
print("It took: ", time.time() - start_time)

split = int(data_length * 0.1)
test_data = data.tail(split)
test_data.to_numpy()

split = int(data_length * 0.1)
test_data = data.tail(split)
test_j0 = test_data["G1"]
test_j1 = test_data["G2"]
test_j2 = test_data["absences"]
test_j3 = test_data["freetime"]
test_ys = test_data["G3"]

def get_hypothesis(thetas, xs): # Example [2, 3, 5]
    # Make sure xs[0] = 1
    hyp = sum([thetaj * xj for thetaj, xj in zip(thetas, xs)])
    return hyp

def error(hypothesis, y):
    return hypothesis - y

def get_accuracy(thetas):
    dif = 0
    for x in test_data.index:
        xs = [1, test_j0[x], test_j1[x], test_j2[x], test_j3[x]]
        hypothesis = get_hypothesis(thetas, xs)
        # print(hypothesis, test_ys[x])
        dif += error(hypothesis, test_ys[x])**2
    return dif / len(test_data) ** 1/2

print("Accuracy:", get_accuracy(thetas))
