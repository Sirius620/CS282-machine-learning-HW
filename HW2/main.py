import csv
import math

import numpy as np
import matplotlib.pyplot as plt

max_iter = 1000
theta = np.array([[0.] for i in range(1000)])
iter_cnt = 0
learning_rate = 0.0001


def diff(matrix1, matrix2, matrix3):
    return np.dot(matrix1, matrix2) - matrix3

def cost(matrixA, matrixX, matrixb):
    LSE = 0
    m,n = matrixb.shape
    for i in range(m):
        LSE+=(np.dot(matrixA[i],matrixX)-matrixb[i])**2
    return 0.5 * LSE

def read_data(path):
    f = open(path)
    reader = csv.reader(f)
    matrix = []
    for i in reader:
        temp = []
        for j in i:
            temp.append(float(j))
        matrix.append(temp)
    return matrix


if __name__ == '__main__':
    m1 = np.array(read_data("./data/A.csv"))
    m2 = np.array(read_data("./data/b.csv"))

    loss_list = []
    grad_list = []
    learning_rate_list = []

    while iter_cnt < max_iter:
        cur_loss = cost(m1,theta,m2)
        error = diff(m1, theta, m2)
        grad = np.dot(m1.T, error)
        temp1 = np.dot(grad.T,grad)
        l2grad = temp1**0.5
        grad_list.append(math.log10(l2grad[0][0]))
        temp2 = np.dot(grad.T,m1.T)
        temp2 = np.dot(temp2,m1)
        temp2 = np.dot(temp2,grad)
        learning_rate = temp1/temp2
        learning_rate_list.append(learning_rate[0][0])
        theta -= learning_rate * grad
        new_loss = cost(m1, theta, m2)
        loss_list.append(math.log10(new_loss[0]))
        print("in iter %d" % iter_cnt)
        print("the learning rate is:")
        print(learning_rate)
        print("the loss is:")
        print(new_loss)
        print("\n")
        iter_cnt += 1

    y = [i for i in range(max_iter)]

    fig1 = plt.figure(0)
    plt.title("learning_rate")
    plt.xlabel("iter")
    plt.ylabel("learning_rate")
    plt.ylim((0.0001, 0.001))
    plt.scatter(y,learning_rate_list)
    plt.savefig("./learningrate.jpg")
    fig2 = plt.figure(1)
    plt.title("objectvalue")
    plt.ylabel("log10(objective value)")
    plt.xlabel("iter")
    plt.scatter(y, loss_list)
    plt.savefig("./objectvalue.jpg")
    fig3 = plt.figure(2)
    plt.title("l2 norm of gradient")
    plt.ylabel("log10(l2 norm of gradiente)")
    plt.xlabel("iter")
    plt.scatter(y, grad_list)
    plt.savefig("./gradient.jpg")
    plt.show()


