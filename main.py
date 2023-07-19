# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:47:30 2023

@author: Martin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Custom_dense import *
from WeightsOrthogonalityConstraint import *
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, InputSpec
import re
import os
cwd = os.getcwd()

clmns = [f'c{i}' for i in range(1, 102)]
prbs=pd.read_csv(fr"{cwd}\dataset\PRBS.csv",delimiter=";", names=clmns,header=None)


PRBS1_M = [f'Mdd{prbs.iloc[0,i]}Hz'for i in range(0,101)]
PRBS1_P = [f'Pdd{prbs.iloc[0,i]}Hz'for i in range(0,101)]
PRBS2_M = [f'Mqq{prbs.iloc[1,i]}Hz'for i in range(0,49)]
PRBS2_P = [f'Pqq{prbs.iloc[1,i]}Hz'for i in range(0,49)]
c = ['P','class','type']
columns = PRBS1_M+PRBS1_P+PRBS2_M+PRBS2_P+c

df_ccGFM = pd.read_csv(fr"{cwd}\\dataset\data_f_ccGFM_ana.csv",delimiter=";", names=columns,header=None)
df_ccGFM['type'] = 1
df_ccGFM2 = pd.read_csv(fr"{cwd}\\dataset\data_f_GFM_cc2.csv",delimiter=";", names=columns,header=None)
df_ccGFM2['type'] = 1
df_vcGFM = pd.read_csv(fr"{cwd}\\dataset\data_f_GFM.csv",delimiter=";", names=columns,header=None)
df_vcGFM['type'] = 2
df_GFL_PQ = pd.read_csv(fr"{cwd}\\dataset\data_f_GFL_PQ.csv",delimiter=";", names=columns,header=None)
df_GFL_PQ['type'] = 3
df_GFL_PV = pd.read_csv(fr"{cwd}\\dataset\data_f_GFL_PV.csv",delimiter=";", names=columns,header=None)
df_GFL_PV['type'] = 4
data = pd.concat([df_ccGFM, df_ccGFM2, df_vcGFM, df_GFL_PQ, df_GFL_PV])

data = pd.get_dummies(data,columns=['type'])
data = pd.get_dummies(data,columns=['class'])

keep_cols = [col for col in data if "type_" in col]
Y_type = data[keep_cols]
keep_cols = [col for col in data if "class" in col]
Y_class = data[keep_cols]
keep_cols = [col for col in data if "Hz" in col]
X= data[keep_cols]


# Data preprocessing for autoencoder reconstruction
scaler = MinMaxScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)

X_train, X_test = train_test_split(Xscaled, test_size=0.2, random_state=1)

nb_epoch = 1500
batch_size = 1000
input_dim = Xscaled.shape[1]
encoding_dim = 150
learning_rate = 1e-3

custom_reg = WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0)
encoder_1 = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_regularizer= custom_reg,
                  name='encoder_1') 
# encoder_2 = Dense(encoding_dim, activation="linear", input_shape=(100,), use_bias = True, name='encoder_2')

# decoder_1 = Dense(1000, activation="relu", input_shape=(encoding_dim,), use_bias = True, name='decoder')
# decoder_2 = Dense(input_dim, activation="relu", input_shape=(100,), use_bias = True, name='decoder_2')

decoder_1 = Custom_dense(input_dim, activation="linear", freeze_weights = encoder_1, use_bias = False, name='decoder_1')
# decoder_2 = Custom_dense(input_dim, activation="linear", freeze_weights = encoder_1, use_bias = False, name='decoder_2')

autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder_1)
autoencoder.add(Dropout(.05))
# autoencoder.add(encoder_2)
autoencoder.add(decoder_1)
# autoencoder.add(decoder_2)

autoencoder.compile(metrics=['mse'],
                    loss='mean_squared_error',
                    optimizer='adam')
autoencoder.summary()

autoenc_history = autoencoder.fit(X_train, X_train,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_data = (X_test, X_test),
                shuffle=True,
                verbose=1)


# Data preprocessing for encoder + classification
scaler = MinMaxScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Y_classif = Y_class
X_train_classif, X_test_classif, Y_train_classif, Y_test_classif = train_test_split(Xscaled, Y_classif, test_size=0.2, random_state=1)

encoder = Sequential(name = 'freezed_encoder')
encoder.add(encoder_1)
# encoder.add(encoder_2)
encoder.trainable = False

classifier = Sequential(name = 'classifier')
classifier.add(encoder)
classifier.add(Dense(len(Y_classif.columns), activation = 'softmax'))


classifier.compile(metrics=['accuracy'],
                    loss='categorical_crossentropy',
                    optimizer='sgd')
classifier.summary()


history = classifier.fit(X_train_classif, Y_train_classif,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_data = (X_test_classif, Y_test_classif),
                shuffle=True,
                verbose=1)
