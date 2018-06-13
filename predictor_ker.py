# -*- coding: utf-8 -*-
# @Time    : 2018/5/29 10:11
# @Author  : LeonHardt
# @File    : predictor_ker.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation

# data_path = os.getcwd()+"/data_error/"
# x_train = np.loadtxt(data_path+"x_train_error93.txt", delimiter=',')
# y_train = np.loadtxt(data_path+"y_train_error93.txt", delimiter=',')
# x_test = np.loadtxt(data_path+"x_test_error93.txt", delimiter=',')
# # print("ready")
# im = Imputer(strategy="most_frequent")
# x_train = im.fit_transform(x_train)
# x_test = im.transform(x_test)
data_path = os.getcwd()+"/data/"

x_train = np.loadtxt(data_path+"x_train_most.txt", delimiter=',')
y_train = np.loadtxt(data_path+"y_train_filter.txt", delimiter=',')
x_test = np.loadtxt(data_path+"x_test_a_most.txt", delimiter=',')
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.reshape(-1, 1))


print(y_train)
print(y_train.shape)

model = Sequential()

model.add(Dense(input_dim=297, units=297, activation='relu'))
# model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# cw = {0: 1, 1: 100}
model.fit(x_train, y_train, epochs=20, batch_size=10000)

prob = model.predict_proba(x_test)
np.savetxt(os.getcwd()+"/prediction/ker200_1_error93_1.txt", prob, delimiter=',')
# model.save("merge_model")
