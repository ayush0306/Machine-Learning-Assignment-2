
# coding: utf-8

# In[1]:


# get_ipython().magic(u'matplotlib inline')


# In[2]:


from numpy import genfromtxt
import numpy as np
from random import randint
from cStringIO import StringIO
import sklearn
import sklearn.linear_model as lm
import copy


# In[21]:


X_train = genfromtxt('notMNIST_train_data.csv', delimiter=',')
y_train = genfromtxt('notMNIST_train_labels.csv', delimiter=',')
X_test = genfromtxt('notMNIST_test_data.csv', delimiter=',')
y_test = genfromtxt('notMNIST_test_labels.csv', delimiter=',')
print len(X_test)



# # In[18]:
def findLabel(predVec):
    # print(predVec.shape,type(predVec))
    predVec = predVec - 0.5
    return np.heaviside(predVec,0)

# finalc = []
c = 0.0000001
while (c <= 10.0):
    print "l1"
    classifier = lm.LogisticRegression(penalty = 'l1', C=c)
    classifier.fit(X_train, y_train)
    yPredict_l1 = classifier.predict(X_test)
    yPredict_l1 = findLabel(yPredict_l1)
    print "lambda = ", 1.0/float(c)
    print "accuracy = ", sklearn.metrics.accuracy_score(y_test, yPredict_l1)
    # coef = classifier.coef_.reshape(784,1)
    # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,255), copy=False)
    # scaler.fit(coef)
    # scaler.transform(coef)
#     print coef
    # showarray(coef.reshape(28,28))
    # finalc.append(copy.deepcopy(coef))
    print "l2"
    classifier = lm.LogisticRegression(penalty = 'l2', C=c)
    classifier.fit(X_train, y_train)
    yPredict_l2 = classifier.predict(X_test)
    yPredict_l2 = findLabel(yPredict_l2)
    print "lambda = ", 1.0/float(c)
    print "accuracy = ", sklearn.metrics.accuracy_score(y_test, yPredict_l2)
#     coef = classifier.coef_.reshape(784,1)
#     scaler.fit(coef)
#     scaler.transform(coef)
# #     print coef
#     showarray(coef.reshape(28,28))
    # finc.append(copy.deepcopy(coef))
    c *= 10
# print final[4]-finalc[0]

