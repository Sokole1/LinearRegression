# https://ozzieliu.com/2016/02/09/gradient-descent-tutorial/
# https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance 
# https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# df = pd.read_csv("data.txt", sep=",", header=None, names=["A", "B"])
data = pd.read_csv("YearPredictionMSD.txt", sep=",", header=None)

# data = data[["Time","CO","Humidity","Temperature","Flowrate","Heatervoltage","R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14"]]

## Split population and profit into X and y
y_df = pd.DataFrame(data[0])
for x in range(1,90):
    X_df = pd.DataFrame(data[x])
    style.use("ggplot")
    plt.scatter(X_df, y_df)
    plt.xlabel(x)
    plt.ylabel('Y')
    plt.show()
# X_f = pd.DataFrame(data)
# print(X_f.shape[1])


## Length, or number of observations, in our data
# m = len(y_df)


# plt.figure(figsize=(10,8))
