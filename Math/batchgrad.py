import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3"]]
# print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Coefficients:", linear.coef_)
print("Intercept:", linear.intercept_)


print(acc)

def get_z(m1, m2, b, max):
    points = []
    for x in range(max):
        points.append(m1*x + m2*x + b)
    return points

m1 = linear.coef_[0]
m2 = linear.coef_[1]
b = linear.intercept_

ax = plt.axes(projection="3d")
ax.scatter(data["G1"], data["G2"], data["G3"], label="First plot")

xs = [x for x in range(20)]
ys = [x for x in range(20)]
zs = get_z(m1, m2, b, 20)

ax.plot(xs, ys, zs,"black")
print(max(data["G3"]))
ax.set(xlabel="G1", ylabel="G2", zlabel="G3")
ax.legend(loc="upper left")
ax.set_title("Monkey")

plt.show()








