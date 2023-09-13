import os
import pandas as pd
import numpy as np
from gtts import gTTS
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import joblib
import string
from Model_DNN import Model_DNN
from Model_LSTM import Model_LSTM
from Model_NN import Model_NN
from Model_RNN import Model_RNN
from Model_SVM import Model_SVM
from Plot_Results import Plot_Results, Plot_Confusion, Plot_ROC
from GUI import GUI


def text_to_speech(text, output_mp3):
    tts = gTTS(text)
    tts.save(output_mp3)


def train_svm_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm_classifier = MultiOutputClassifier(SVC(kernel='linear'))
    svm_classifier.fit(X_train, y_train)
    return svm_classifier, X_test, y_test


def evaluate_model(svm_classifier, X_test):
    y_pred = svm_classifier.predict(X_test)
    return y_pred


def save_svm_model(svm_classifier, save_model_path):
    joblib.dump(svm_classifier, save_model_path)


def load_svm_model(load_model_path):
    return joblib.load(load_model_path)


def preprocess_sentence(sentence):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    processed_sentence = sentence.translate(translator).lower().strip()
    return processed_sentence


## Read Data
an = 0  # set 1 to recompute
if an == 1:
    file_name = 'Dataset.xlsx'
    df = pd.read_excel(file_name, sheet_name='Sheet1')
    df = df.fillna(0)
    Query = df['Question'].to_numpy()
    Classes = df[df.columns.values[2:]].to_numpy().astype('int')
    np.save('Indents', df.columns.values[2:])
    np.save('Targets', Classes)
    np.save('Query', Query)

## Read Answer
an = 0
if an == 1:
    file_name = 'Answer.xlsx'
    df = pd.read_excel(file_name, sheet_name='Sheet1')
    Answer = df['Answer'].to_numpy()
    np.save('Answer.npy', Answer)

## Preprocessing
an = 0
if an == 1:
    Preprocessed = []
    Query = np.load('Query.npy', allow_pickle=True)
    for i in range(len(Query)):
        prep = preprocess_sentence(Query[i])
        Preprocessed.append(str(prep))
    np.save('Preprocessed.npy', list(Preprocessed))

## Generate Audio files in .mp3 format
an = 0
if an == 1:
    Query = np.load('Query.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Indent = np.load('Indents.npy', allow_pickle=True)
    Audio = []
    features = []
    target_length = 200000
    for i in range(len(Query)):
        print(i + 1)
        ind = np.where(Target[i] == 1)

        # Directory to save the output file
        output_directory = 'MP3Files/' + Indent[ind][0]

        # Create the directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_file_mp3 = './' + output_directory + '/' + str(i + 1) + '.mp3'
        text_to_speech(Query[i], output_file_mp3)
        audio, sr = librosa.load(output_file_mp3)

        data = audio.reshape(1, -1)
        # Pad or truncate the features to a fixed length (target_length)
        if data.shape[1] < target_length:
            padded_data = np.pad(data, ((0, 0), (0, target_length - data.shape[1])), mode='constant')
            features.append(padded_data.ravel())
        elif data.shape[1] > target_length:
            truncated_data = data[:, :target_length]
            features.append(truncated_data.ravel())
        else:
            features.append(data.ravel())
        Audio.append(audio)
    np.save('Audios.npy', Audio)
    np.save('Features.npy', features)

## SVM Training and save the model(Input --> Audio, Output --> Indent)
an = 0
if an == 1:
    Features = np.load('Features.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Indent = np.load('Indents.npy', allow_pickle=True)

    Train = 0
    # Save the SVM classifier model
    save_model_path = "svm_model.joblib"
    if Train:
        # Train the SVM model
        svm_classifier, X_test, y_test = train_svm_model(Features, Target)

        save_svm_model(svm_classifier, save_model_path)
    else:
        # Load the SVM classifier model from the file
        svm_classifier = load_svm_model(save_model_path)
        X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=42)

    predict = evaluate_model(svm_classifier, X_test)
    predicted_class = []
    for i in range(len(predict)):
        ind = np.where(predict[i] == 1)
        if len(ind[0]) == 0:
            predicted_class.append(Indent[0])
        else:
            # Directory to save the output file
            predicted_class.append(Indent[ind][0])

    actual_class = []
    for i in range(len(y_test)):
        ind = np.where(y_test[i] == 1)
        actual_class.append(Indent[ind][0])

    print(actual_class)
    print(predicted_class)

## Evaluation
an = 0
if an == 1:
    Features = np.load('Features.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Eval_all = []
    k_fold = 5
    for i in range(k_fold):
        Eval = np.zeros((5, 15), dtype=object)
        Total_Index = np.arange(Features.shape[0])
        Test_index = np.arange(((i - 1) * (Features.shape[0] / k_fold)) + 1, i * (Features.shape[0] / k_fold))
        Train_Index = np.setdiff1d(Total_Index, Test_index)

        train_data = Features[Train_Index, :]
        train_target = Target[Train_Index, :]
        test_data = Features[Test_index, :]
        test_target = Target[Test_index, :]

        Eval[0, :] = Model_NN(train_data, train_target, test_data, test_target)
        Eval[1, :] = Model_DNN(train_data, train_target, test_data, test_target)
        Eval[2, :] = Model_RNN(train_data, train_target, test_data, test_target)
        Eval[3, :] = Model_LSTM(train_data, train_target, test_data, test_target)
        Eval[4, :] = Model_SVM(train_data, train_target, test_data, test_target)

        Eval_all.append(Eval)
    np.save('Eval_all.npy', Eval_all)

Plot_Results()
Plot_Confusion()
Plot_ROC()
GUI()
