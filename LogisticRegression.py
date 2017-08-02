# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,model_selection

def load_data():
    iris = datasets.load_iris()
    x_train=iris.data
    y_trian=iris.target
    return model_selection.train_test_split(x_train,y_trian,test_size=0.25,random_state=0,stratify=y_trian)

def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train,y_train)
    print('Coefficients:%s, intercept %s' % (regr.coef_, regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))

def test_LogisticRegression_multinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s' % (regr.coef_, regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))

def test_LogisticRegression_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2,4,num=100)
    scores=[]
    for C in Cs:
        regr= linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test,y_test))
    plt.figure(0)
    plt.plot(Cs, scores)
    plt.xlabel("C")
    plt.ylabel(r"score")
    plt.xscale('log')
    plt.title("LogisticRegression")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    test_LogisticRegression_C(X_train, X_test, y_train, y_test)

