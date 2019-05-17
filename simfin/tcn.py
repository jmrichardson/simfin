import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense
from tcn import compiled_tcn, TCN
from keras.preprocessing.sequence import TimeseriesGenerator

model = compiled_tcn(return_sequences=False,
                     num_feat=X.shape[2],
                     num_classes=2,
                     nb_filters=24,
                     kernel_size=8,
                     dilations=[2 ** i for i in range(9)],
                     nb_stacks=1,
                     max_len=X.shape[1],
                     use_skip_connections=True,
                     )

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=256)
# model.fit(X_train, y_train, epochs=10)

print(y_val)
a = model.predict(X_val)
out = pd.DataFrame(a)



i = Input(shape=(20, 84))
m = TCN()(i)
m = Dense(1, activation='sigmoid')(m)
model = Model(inputs=[i], outputs=[m])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_split_seq, y_train_split_seq,
          validation_data=(X_val_split_seq, y_val_split_seq),
          epochs=200, verbose=2, batch_size=64)
# model.fit(X, y, validation_data=(X_val, y_val), epochs=2, verbose=2)

a = model.predict(X_val)


plt.plot(p)
plt.plot(y)
plt.title('Monthly Milk Production (in pounds)')
plt.legend(['predicted', 'actual'])
plt.show()













batch_size, timesteps, input_dim = None, 20, 1


gen = TimeseriesGenerator(np.array(X_train_split), np.array(y_train_split), length=20)
print('Samples: %d' % len(gen))
range(len(gen))
for i in range(len(gen)):
    print(i)



def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train




lookback_window = 12




i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False)(i)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)