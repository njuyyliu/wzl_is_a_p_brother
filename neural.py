# -*- coding:utf-8 -*-
"""
@author: LIU
@file:neural.py
@time:2017/4/4 20:09
"""
import numpy as np
import time
from sklearn.linear_model import LinearRegression


# ---read train data------------------------------------------------
def read_data(path):
    data_matrix = []
    data_label = []
    file_train = open(path)
    for line in file_train.readlines():
        line.strip()
        temp = list(map(float, line.split(',')))
        data_label.append(temp[len(temp)-1])
        del temp[(len(temp)-1)]
        data_matrix.append(temp)
    file_train.close()
    return np.matrix(data_matrix), data_label


# ------------------------------------------------------------------
def neuron(arr, label):
    num, d = np.shape(arr)
    w = np.zeros((1, d))
    alpha = 0.001
    for i in xrange(num):
        output_y = w * arr[i].T
        error = label[i] - output_y
        w -= alpha * error * arr[i]
    return w


def test(arr, w, label):
    num, d = np.shape(arr)
    err = 0
    for i in xrange(num):
        output_y = w * arr[i].T
        if output_y > 0:
            output_y = 1
        else:
            output_y = -1
        if output_y != label[i]:
            err += 1
    return err


def line_reg(arr, label, test_d, test_l):
    lin_reg = LinearRegression()
    lin_reg.fit(arr, label)
    output_y = lin_reg.predict(test_d)

    # sum_mean1 = 0
    sum_error = 0
    for i in range(len(output_y)):
        if output_y[i] > 0:
            output_y[i] = 1
        else:
            output_y[i] = -1
        if output_y[i] != test_l[i]:
            sum_error += 1
        # sum_mean1 += (output_y[i] - test_l[i]) ** 2
    # sum_erro1 = np.sqrt(sum_mean1 / 50)
    return sum_error


if __name__ == "__main__":
    start = time.clock()
    train_data, train_label = read_data('sonar-train.txt')
    test_data, test_label = read_data('sonar-test.txt')
    wn = neuron(train_data, train_label)
    error_num = test(test_data, wn, test_label)
    print "Line neuron error:"
    print " %0.5f " % (float(error_num) / float(test_data.shape[0]))

    line_error = line_reg(train_data, train_label, test_data, test_label)
    print "Line regression error:"
    print " %0.5f " % (float(line_error) / float(test_data.shape[0]))

    end = time.clock()
    # print " %0.5f s" % (end - start)
