import numpy as np
from Evaluation import evaluation
from NeuralNetwork import train_nn



def Model_NN(train_data,train_target,test_data,test_target):
    pred = np.zeros(test_target.shape).astype('int')
    for i in range(2):#train_target.shape[1]):
        pred[:, i] = train_nn(train_data,train_target[:, i],test_data)  # NN
    Eval = evaluation(pred, test_target)
    return np.asarray(Eval).ravel()