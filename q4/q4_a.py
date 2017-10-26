"""
--------
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet
--------
"""
from __future__ import print_function
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
import numpy as np
import sys

def get_input_data(filename):
    data = []; label = []
    f = open(filename,'r')
    for line in f:
        line = line.strip()
        vals = line.split(',')
        label.append(int(vals[-1]))
        tmp = []
        for i in xrange(len(vals)-1):
            tmp.append(float(vals[i]))
        data.append(tmp)
    data = np.asarray(data); label = np.asarray(label)
    f.close()
    return data, label

def get_input_testdata(filename):
    data = []
    f = open(filename,'r')
    for line in f:
        line = line.strip()
        vals = line.split(',')
        tmp = []
        for i in xrange(len(vals)):
            tmp.append(float(vals[i]))
        data.append(tmp)
    data = np.asarray(data)
    f.close()
    return data


def calculate_accuracy(predicted, truth):
    correct = 0
    total = len(predicted)
    for i in xrange(total):
        if float(predicted[i]) == float(truth[i]): 
            correct += 1
    return float(correct) / float(total)
    

def findLabel(predVec):
    # print(predVec.shape,type(predVec))
    predVec = predVec - 0.5
    return np.heaviside(predVec,0)

def classifier(x_train, y_train, x_test, alpha):
    mod = Lasso(alpha = alpha)
    mod.fit(x_train, y_train)
    regOut_train = mod.predict(x_train)
    regOut_test = mod.predict(x_test)
    labels_train = findLabel(regOut_train)
    labels_test = findLabel(regOut_test)
    train_accuracy = calculate_accuracy(labels_train, y_train)
    # test_accuracy = calculate_accuracy(labels_test, y_test)
    # print("Training Accuracy for alpha = ",alpha," : " , accuracy)
    # print("Testing Accuracy for alpha = ",alpha," : " ,accuracy)
    # print(alpha, "\t", train_accuracy, "\t", test_accuracy)
    for pred in labels_test:
        print(int(pred))
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python q4_a.py [relative/path/to/train/file] [relative/path/to/test/file]")
        exit()

    x_train, y_train = get_input_data(sys.argv[1])
    x_test = get_input_testdata(sys.argv[2])
    alpha = 0.0000001
    # print("Alpha \t Training Accuracy \t Testing Accuracy")
    while(alpha <= 0.0000001):
        classifier(x_train, y_train, x_test, alpha)
        alpha*=10