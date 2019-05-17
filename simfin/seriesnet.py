from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv1D, Input, Add, Activation, Dropout

from keras.models import Sequential, Model

from keras.regularizers import l2

from keras.initializers import TruncatedNormal

from keras.layers.advanced_activations import LeakyReLU, ELU

from keras import optimizers


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                              seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)

        layer_out = Activation('selu')(layer_out)

        skip_out = Conv1D(1, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_in = Conv1D(1, 1, activation='linear', use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                               seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_out = Add()([residual, network_in])

        return network_out, skip_out

    return f


def DC_CNN_Model(length, vars):
    input = Input(shape=(length, vars))

    l1a, l1b = DC_CNN_Block(32, 2, 1, 0.001)(input)
    l2a, l2b = DC_CNN_Block(32, 2, 2, 0.001)(l1a)
    l3a, l3b = DC_CNN_Block(32, 2, 4, 0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32, 2, 8, 0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32, 2, 16, 0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32, 2, 32, 0.001)(l5a)
    l6b = Dropout(0.8)(l6b)  # dropout used to limit influence of earlier data
    l7a, l7b = DC_CNN_Block(32, 2, 64, 0.001)(l6a)
    l7b = Dropout(0.8)(l7b)  # dropout used to limit influence of earlier data

    l8 = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

    l9 = Activation('relu')(l8)

    l21 = Conv1D(1, 1, activation='linear', use_bias=False,
                 kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                 kernel_regularizer=l2(0.001))(l9)

    model = Model(input=input, output=l21)

    adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None,
                           decay=0.0, amsgrad=False)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])

    return model

length = 20
vars = 84


model = DC_CNN_Model(length, vars)
model.fit(X_train_split_seq, y_train_split_seq, epochs=3000)

for _ in range(2):
    print("hi")
