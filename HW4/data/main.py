import pandas as pd
import numpy as np

df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")


def l2norm(l, w):
    return np.dot(w, w) * l


def l2lossfunction(y, x, w, l):
    LSE = 0
    m, n = y.shape
    for i in range(m):
        LSE += (np.dot(w[i], x) - y[i]) ** 2 + l2norm(l, w[i])
    return LSE


def train(l, x, y):
    return 0
