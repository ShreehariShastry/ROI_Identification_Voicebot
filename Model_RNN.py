import numpy as np

# https://www.tensorflow.org/guide/keras/rnn
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from Evaluation import evaluation


def Model_RNN(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 2]
    out, model = RNN_train(train_data, train_target, test_data, sol)  # RNN
    out = np.reshape(out, test_target.shape)
    return out



def Model__RNN(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 2]
    out, model = RNN_train(train_data, train_target, test_data, sol)  # RNN
    out = np.reshape(out, test_target.shape)
    pred = np.round(out)

    Eval = evaluation(pred, test_target)
    return np.asarray(Eval).ravel()



def RNN_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(5), input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=int(sol[0]), batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model

