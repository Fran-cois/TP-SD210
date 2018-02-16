from sklearn import datasets
from sklearn import svm
from random import randint
from skelarn.datasets import fetch_lfw_people

def getTrainingAndTest(X,y):
    LEN = len(X)
    X_TRAINING =[]
    y_TRAINING = []
    X_TEST = []
    y_TEST = []
    i = 0
    while i < LEN:
        X_TRAINING.append(X[i])
        y_TRAINING.append(y[i])
        X_TEST.append(X[i+1])
        y_TEST.append(y[i+1])
        i += 2
    return X_TRAINING,y_TRAINING, X_TEST, y_TEST

def getError(y1,y2):
    sum_ = 0
    len_ = len(y1)
    for i in range(len_):
        if y1[i] == y2[i]:
            sum_ += 1
    return sum_, len_
def linearSVC(X,y):
    clf = svm.LinearSVC()
    clf.fit(X,y)
    return clf
def linearKernelSVC(X,y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X,y)
    return clf
def polynomialSVC(X,y,deg=5):
    clf = svm.SVC(kernel='poly', degree=deg)
    clf.fit(X,y)
    return clf
def rbfSVC(X,y):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X,y)
    return clf
def sigmoidSVC(X,y):
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X,y)
    return clf


def printError(clf,label,X_TEST,y_TEST):
    sum_ , len_ = getError(clf.predict(X_TEST),y_TEST)
    print label,"err:" ,sum_, "/", len_

def CompareClassifier():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[y != 0, :2]
    y = y[y != 0]

    X_TRAINING,y_TRAINING, X_TEST, y_TEST = getTrainingAndTest(X,y)

    clfLinear = linearSVC(X_TRAINING,y_TRAINING)
    clfPolynomial = polynomialSVC(X_TRAINING,y_TRAINING)
    clfLinear2 = linearKernelSVC(X_TRAINING,y_TRAINING)
    clfRBF = rbfSVC(X_TRAINING,y_TRAINING)
    clfsigmoid = sigmoidSVC(X_TRAINING,y_TRAINING)

    printError(clfLinear,"Linear",X_TEST,y_TEST)
    printError(clfPolynomial,"Polynomial",X_TEST,y_TEST)
    printError(clfLinear2,"clfLinear2",X_TEST,y_TEST)
    printError(clfRBF,"clfRBF",X_TEST,y_TEST)
    printError(clfsigmoid,"clfsigmoid",X_TEST,y_TEST)


if __name__ == '__main__':
    CompareClassifier()
