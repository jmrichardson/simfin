from keras import Input, Model
from keras.layers import Dense, LSTM, Dropout


i = Input(shape=(20, 84))
m = LSTM(100)(i)
m = Dense(1, activation='sigmoid')(m)
model = Model(inputs=[i], outputs=[m])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_split_seq, y_train_split_seq,
          validation_data=(X_val_split_seq, y_val_split_seq),
          epochs=200, verbose=2, batch_size=256)


