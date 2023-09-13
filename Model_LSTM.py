import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from Evaluation import evaluation



def Model_LSTM(trainX, trainY, testX, test_y):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(trainX, trainY, epochs= 5, batch_size=1, verbose=2)
    testPredict= model.predict(testX).ravel()
    act = test_y.reshape(len(test_y), 1)
    pred = testPredict
    Eval = evaluation(pred, test_y)
    return Eval

