from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from Evaluation import evaluation


def Model_SVM(X_train, y_train, X_test, y_test):
    svm_classifier = MultiOutputClassifier(SVC(kernel='linear'))
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    Eval = evaluation(y_pred, y_test)
    return Eval
