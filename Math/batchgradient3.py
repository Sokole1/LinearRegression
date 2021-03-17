import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pickle
import time

data = pd.read_csv("student-mat.csv", sep=";")

data_length = len(data)
train_data = data.head(int(data_length * 0.9))[["G1", "G2", "G3", "absences"]]
data_j0 = train_data["G1"]
data_j1 = train_data["G2"]
data_j2 = train_data["absences"]
data_ys = train_data["G3"]

split = int(data_length * 0.1)
test_data = data.tail(split)
test_j0 = test_data["G1"]
test_j1 = test_data["G2"]
test_j2 = test_data["absences"]
test_ys = test_data["G3"]

def get_z(b, m1, m2, max):
    points = []
    for x in range(max):
        points.append(m1*x + m2*x + b)
    return points

def get_hypothesis(thetas, xs): # Example [2, 3, 5]
    # Make sure xs[0] = 1
    hyp = sum([thetaj * xj for thetaj, xj in zip(thetas, xs)])
    return hyp

def error(hypothesis, y):
    return hypothesis - y

def partial_derivative_of_costfunction(thetas, j):
    sum = 0
    for x in range(len(data_ys)):
        xs = [1, data_j0[x], data_j1[x], data_j2[x]]
        hypothesis = get_hypothesis(thetas, xs)
        err = error(hypothesis, data_ys[x])
        sum += err * xs[j]
    
    return 1/len(data_ys) * sum

j0_change = []
j1_change = []
j2_change = []
t0s = []
t1s = []
t2s = []
def update_thetas(thetas, epochs, learning_rate):
    count = 0
    while True:
        count += 1
        t_change = 0
        for j in range(len(thetas)):
            change = partial_derivative_of_costfunction(thetas, j)
            thetas[j] = thetas[j] - learning_rate * change
            if j == 0:
                j0_change.append(change)
            elif j == 1:
                j1_change.append(change)
            elif j == 2:
                j2_change.append(change)
            t_change += abs(change)
        if count % 30 == 0 or t_change > 0.8:
            t0s.append(thetas[0])
            t1s.append(thetas[1])
            t2s.append(thetas[2])
        if t_change < 0.01: 
            break
    return thetas

def find_thetas():
    start_time = time.time()
    # Just set all to be 0
    thetas = [0,0,0,0]
    thetas = update_thetas(thetas, 100, 0.01)
    print("Coefficients are: ",thetas)
    print("It took: ", time.time() - start_time)
    return thetas

thetas = find_thetas()

def get_accuracy(thetas):
    dif = 0
    for x in test_data.index:
        xs = [1, test_j0[x], test_j1[x], test_j2[x]]
        hypothesis = get_hypothesis(thetas, xs)
        # print(hypothesis, test_ys[x])
        dif += error(hypothesis, test_ys[x])**2
    return dif / len(test_data) ** 1/2

print("Acc", get_accuracy(thetas))

iterations = [x for x in range(len(j0_change))]

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(2,2,1, projection="3d")
ax.scatter(t0s, t1s, t2s, label="First plot")
ax.set_title("Theta change over time")

ax = fig.add_subplot(2,2,4)
ax.plot(iterations, j0_change) 
ax.set_title("J0") 

ax = fig.add_subplot(2,2,2)
ax.plot(iterations, j1_change) 
ax.set_title("J1") 

ax = fig.add_subplot(2,2,3)
ax.plot(iterations, j2_change) 
ax.set_title("J2") 


plt.show() 






