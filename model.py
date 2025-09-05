import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

def build_model(input_shape, rnn='lstm', units=64, layers=2, dropout=0.2):
    model = Sequential()
    RNNLayer = LSTM if rnn.lower() == 'lstm' else GRU

    for i in range(layers):
        return_seq = (i < layers - 1)
        if i == 0:
            model.add(RNNLayer(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(RNNLayer(units, return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
