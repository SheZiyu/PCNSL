import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
from sklearn.model_selection import RepeatedStratifiedKFold


def readcsv(file):
    df = pd.read_csv(file, encoding='unicode_escape')
    # print(df)
    x = df[["age >60", "Prot CSF >ULN", "LDH >ULN", "Prot CSF >ULN", "Deep lesions", "IELSG score"]].values
    y = df["OS 1 year"].values
    # csvfile = open(files, 'r')
    # plots = csv.reader(csvfile, delimiter=',')
    # x = []
    # y = []
    # for row in plots:
    #    y.append((row[2]))
    #    x.append((row[1]))
    # print(y)
    return x, y


X, y = readcsv("C:\\Users\\shezi\\Desktop\\pcnsl.csv")
# print(Y)
# print(np.any(np.isnan(X)))
# print(np.all(np.isfinite(X)))
# print(np.where(np.isnan(X)))
# plt.plot(x1, y1, color="red", label="Train_loss")

# print(X)
# print(y)
n_features = X.shape[1]

C = 10
kernel = 1.0 * RBF([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # for GPC

# Create different classifiers.
classifiers = {
    "L1 logistic": LogisticRegression(
        C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (Multinomial)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (OvR)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
    ),
    "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
    "GPC": GaussianProcessClassifier(kernel),
}

n_classifiers = len(classifiers)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=12)
for index, (name, classifier) in enumerate(classifiers.items()):
    Accuracy = []
    AUC = []
    Precision = []
    Recall = []
    F1score = []
    for train_index, test_index in rskf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        p, r, f, s = precision_recall_fscore_support(y_test, y_pred, average='macro')

        y_pred_new = classifier.predict_proba(X_test)
        y_pred_new = pd.DataFrame(y_pred_new)
        y_pred_new = y_pred_new.iloc[:, 1]
        y_pred_new = list(y_pred_new)
        ROC = roc_auc_score(y_test, y_pred_new)

        Accuracy.append(accuracy)
        Precision.append(p)
        Recall.append(r)
        F1score.append(f)
        AUC.append(ROC)

    print(Accuracy)
    print(Precision)
    print(Recall)
    print(F1score)
    print(AUC)
    print("Accuracy for %s: %0.1f%% " % (name, np.mean(Accuracy) * 100))

    print("Precision for %s: %0.1f%% " % (name, np.mean(Precision) * 100))
    print("Recall for %s: %0.1f%% " % (name, np.mean(Recall) * 100))
    print("F1-score for %s: %0.1f%% " % (name, np.mean(F1score) * 100))

    print("AUC for %s: %0.1f%% " % (name, np.mean(AUC) * 100))
    print(2 * "\n")



