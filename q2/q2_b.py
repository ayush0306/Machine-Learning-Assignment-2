from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import keras
import sys

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
# from keras.layers.advanced_activations import LeakyReLU

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def makeImage(inp):
    tmp1 = inp[:,0,:,:]
    tmp2 = inp[:,1,:,:]
    tmp3 = inp[:,2,:,:]
    rgbArray = np.zeros((inp.shape[0],32,32,3), 'uint8')
    rgbArray[:,:,:,0] = tmp1
    rgbArray[:,:,:,1] = tmp2
    rgbArray[:,:,:,2] = tmp3
    return rgbArray

def findArgMax(y_predArr):
    toret = []
    for x in y_predArr :
        for i,num in enumerate(x):
            if(x[i]==1):
                toret.append(i)
                break
    return toret


if "__name__" != "__main__":

    '''
    Each of the batch files contains a dictionary with the following elements:

    1. data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    2. labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    The  batches.meta file contains a Python dictionary object. It has the label names.
    '''

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    b = unpickle(arg1+"batches.meta")
    label_names = b["label_names"]
    print(label_names)

    # x_train,x_test,y_train,y_test =

    a = unpickle(arg1+"data_batch_1")
    x_train = a["data"]
    # print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0],3,32,32)
    x_train = makeImage(x_train)
    y_train = a["labels"]
    # print(y_train[0:40])
    # print(label_names[y_train[0]])

    # plt.imshow(x_train[0])
    # plt.show()

    for i in xrange(2,6):
        filename = arg1+"data_batch_"+str(i)
        a = unpickle(filename)
        data = a["data"]
        data = data.reshape(data.shape[0],3,32,32)
        data = makeImage(data)
        labels = a["labels"]
        x_train = np.vstack((x_train,data))
        y_train = np.append(y_train,labels)

    a = unpickle(arg2)
    x_test = a["data"]
    x_test = x_test.reshape(x_test.shape[0],3,32,32)
    x_test = makeImage(x_test)
    # y_test = np.array(a["labels"])

    y_train = np.reshape(y_train,(y_train.shape[0],1))
    # y_test = np.reshape(y_test,(y_test.shape[0],1))
    num_classes = len(label_names)
    print(num_classes)
    # print(x_train.shape,y_test.shape)
    # print(len(y_train),len(y_test))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # print(x_train.shape, x_test.shape ,y_train.shape,)


    batch_size = 32
    num_classes = 10
    epochs = 50
    data_augmentation = True
    num_predictions = 20
    acts = ['relu']
    model_name = 'keras_cifar10_trained_model.h5'
    for i in acts:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(BatchNormalization())
        model.add(Activation(i))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation(i))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(i))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation(i))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation(i))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255


        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  shuffle=True,verbose=True)
        y_predArr = model.predict(x_test)
        outfile = open('q2_b_output.txt', 'w')
        for yPred in y_predArr :
            outfile.write(label_names[np.argmax(yPred)] + '\n')
