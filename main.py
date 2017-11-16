#!/usr/bin/env python3

import csv
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
types = []

X = np.empty((0, 6), float)
y = np.array([])

X_test = np.empty((0, 6), float)
y_test = np.array([])

tickets = set()
cabins = set()
embarked = set()

with open('data/train.csv', newline='') as f:
    file = csv.reader(f, delimiter=',', quotechar='"')

    for i, line in enumerate(file):
        if i == 0:
            types = line
            continue

        x = np.array([])
        y_val = 0

        for k, v in enumerate(line):
            t = types[k]

            if t == "Name":
                continue
            elif t == "Ticket":
                tickets.add(v)
                continue
            elif t == "Cabin":
                cabins.add(v)
                continue
            elif t == "Embarked":
                embarked.add(v)
                continue

            if v == '':
                x = np.append(x, -1)
                continue

            if t == "Survived":
                y_val = np.append(x, int(1 if int(v) == 1 else -1))
            if t == "Pclass" or t == "SibSp" or t == "Parch":
                x = np.append(x, int(v))
            elif t == "Sex":
                x = np.append(x, int(v == "male"))
            elif  t == "Age" or t == "Fare":
                x = np.append(x, float(v))

        if len(x) == 0:
            continue

        if i < 500:
            X = np.append(X, [x], axis=0)
            y = np.append(y, y_val)
        else:
            X_test = np.append(X_test, [x], axis=0)
            y_test = np.append(y_test, y_val)

# Survived
# Pclass,Sex,Age,SibSp,Parch,Fare
#X = preprocessing.normalize(X, axis=0)
#X_test = preprocessing.normalize(X_test, axis=0)

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
print("Проверка:")
predicted_test = model.predict(X_test)
print(metrics.classification_report(y_test, predicted_test))
print("\n\n")

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