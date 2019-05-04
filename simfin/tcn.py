from keras.layers import Dense
from keras.models import Input, Model
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
from tcn import TCN

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense
from tcn import compiled_tcn

from tcn import TCN

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##


lookback_window = 12  # months.

X_data = np.array(X_train_split)
y_data = np.array(y_train_split)

x, y = [], []
for i in range(lookback_window, len(X_data)):
    x.append(X_data[i - lookback_window:i])
    y.append(y_data[i])


x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)


model = compiled_tcn(...)
model.fit(x, y) # Keras model.







i = Input(shape=(lookback_window, 83))
m = TCN()(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile('adam', 'mae')

print('Train...')
model.fit(x, y, epochs=100, verbose=2)

p = model.predict(x)

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