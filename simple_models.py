#!/usr/bin/env python3

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def logistic_regression(X, y, X_test):

    # Отбор признаков
    model = ExtraTreesClassifier()
    model.fit(X, y)
    # display the relative importance of each attribute
    print(model.feature_importances_)

    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 3)
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)
    print("\n\n")

    print('Логистическая регрессия')
    model = LogisticRegression()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    #print("Проверка:")
    predicted_test = model.predict(X_test)
    #print(metrics.classification_report(y_test, predicted_test))
    #print("\n\n")
    return predicted_test

def other_models(X, y, X_test, Y_test):

    print("Наивный Байес")
    model = GaussianNB()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("Проверка:")
    predicted_test = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_test))
    print("\n\n")

    print("K-ближайших соседей")
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("Проверка:")
    predicted_test = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_test))
    print("\n\n")

    print("Деревья решений")
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("Проверка:")
    predicted_test = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_test))
    print("\n\n")

    print("Метод опорных векторов")
    from sklearn.svm import SVC
    # fit a SVM model to the data
    model = SVC()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("Проверка:")
    predicted_test = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_test))
    print("\n\n")