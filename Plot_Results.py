import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve


def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

def Plot_Results():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1-Score',
             'MCC']
    Graph_Term = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Classifier = ['TERMS', 'NN', 'DNN','RNN','LSTM','SVM']

    value = eval[3, :, 4:]
    value[:, :-1] = value[:, :-1] * 100


    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[j, :])
    print('-------------------------------------------------- ',
          'Classifier Comparison --------------------------------------------------')
    print(Table)

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    for a in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, Graph_Term[a] + 4] * 100


        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 0], color='green', width=0.10, label="NN")
        ax.bar(X + 0.10, Graph[:, 1], color='red', width=0.10, label="DNN")
        ax.bar(X + 0.20, Graph[:, 2], color='dodgerblue', width=0.10, label="RNN")
        ax.bar(X + 0.30, Graph[:, 3], color='magenta', width=0.10, label="LSTM")
        ax.bar(X + 0.40, Graph[:, 4], color='cyan', width=0.10, label="SVM")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('k fold')
        plt.ylabel(Terms[Graph_Term[a]])
        #plt.legend(loc='best')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/bar_%s.png" % (Graph_Term[a])
        plt.savefig(path1)
        plt.show()



def Plot_Confusion():
    # Confusion Matrix
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    value = Eval[3, 4, :5]
    val = np.asarray([0, 1, 1])
    data = {'y_Actual': [val.ravel()],
            'y_Predicted': [np.asarray(val).ravel()]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
    value = value.astype('int')

    confusion_matrix.values[0, 0] = value[1]
    confusion_matrix.values[0, 1] = value[3]
    confusion_matrix.values[1, 0] = value[2]
    confusion_matrix.values[1, 1] = value[0]

    sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
    sn.plotting_context()
    path1 = './Results/Confusion.png'
    plt.savefig(path1)
    plt.show()

def Plot_ROC():
    lw = 2
    cls = ['NN', 'DNN','RNN','LSTM','SVM']
    colors = cycle(["lime", "orange", "hotpink", "dodgerblue", "cyan", ])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('roc_score.npy', allow_pickle=True)[3, i].astype('float')
        Actual = np.load('roc_act.npy', allow_pickle=True)[3, i].astype('int')
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())

        auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:,
                                                   -1].ravel())

        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label="{0}".format(cls[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path1 = "./Results/roc.png"
    plt.savefig(path1)
    plt.show()


if __name__ == '__main__':
    Plot_Results()
    Plot_Confusion()
    Plot_ROC()