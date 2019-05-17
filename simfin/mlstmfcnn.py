import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
# from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
# from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
from tcn import compiled_tcn, TCN


import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("D:\Projects\simfin\simfin")

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, LSTM, Dropout

def squeeze_excite_block(input):
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se



classes = np.unique(y_train_seq)
le = LabelEncoder()
y_ind = le.fit_transform(y_train_seq.ravel())
recip_freq = len(y_train_seq) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
class_weight = recip_freq[le.transform(classes)]
print("Class weights : ", class_weight)

# One hot encoding
y_train_seq = to_categorical(y_train_seq, len(np.unique(y_train_seq)))
y_val_seq = to_categorical(y_val_seq, len(np.unique(y_val_seq)))

factor = 1. / np.cbrt(2)

model_checkpoint = ModelCheckpoint('tmp/mlstmfcnn.h5', verbose=1, mode='auto',
                                       monitor='loss', save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
callback_list = [model_checkpoint, reduce_lr]

optm = Adam(lr=1e-3)





i = Input(shape=(20, 364))

x = Permute((2, 1))(i)
x = Masking()(x)
x = AttentionLSTM(8)(x)
x = Dropout(0.8)(x)

# w = TCN()(i)

# z = LSTM(100)(i)

y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(i)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = squeeze_excite_block(y)

y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = squeeze_excite_block(y)

y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = GlobalAveragePooling1D()(y)

x = concatenate([x, y])

m = Dense(1, activation='sigmoid')(y)

model = Model(i, m)
model.summary()

# X_train_seq = np.transpose(X_train_seq, (0, 2, 1))
# X_val_seq = np.transpose(X_val_seq, (0, 2, 1))
# print(X_train_seq.shape)
# print(X_val_seq.shape)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_split_seq, y_train_split_seq,
          validation_data=(X_val_split_seq, y_val_split_seq),
          epochs=2000, verbose=2, batch_size=1024)




model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_seq, y_train_seq, batch_size=256, epochs=600, callbacks=callback_list, class_weight=class_weight,
          verbose=2,  validation_data=(X_val_seq, y_val_seq))



