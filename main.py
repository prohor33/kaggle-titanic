#!/usr/bin/env python3

import csv
import numpy as np
from simple_models import *
import pandas as pd


def load_file(filename):
    # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    types = []

    X = np.empty((0, 6), float)
    y = np.array([])
    passenger_ids = np.array([])

    tickets = set()
    cabins = set()
    embarked = set()

    with open(filename, newline='') as f:
        file = csv.reader(f, delimiter=',', quotechar='"')

        for i, line in enumerate(file):
            if i == 0:
                types = line
                continue

            x = np.array([])
            y_val = 0
            passenger_id = 0

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
                elif t == "PassengerId":
                    passenger_id = int(v);

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

            X = np.append(X, [x], axis=0)
            y = np.append(y, y_val)
            passenger_ids = np.append(passenger_ids, passenger_id)

        return X, passenger_ids, y

#X_train, X_test, y_train, y_test = train_test_split(eyes.data, eyes.target, test_size=0.25, random_state=0)

X, _, y = load_file('data/train.csv')
X_test, passenger_ids_test, _ = load_file('data/test.csv')

y_test = logistic_regression(X, y, X_test)

passenger_ids_test = pd.DataFrame(passenger_ids_test, columns=['PassengerId'])
y_test = pd.DataFrame(y_test, columns=['Survived'])

def convert_survived(x):
    if x < 0:
        return 0
    else:
        return 1

#y_test = y_test.apply(convert_survived)

#print(passenger_ids_test)

result = pd.concat([passenger_ids_test, y_test], axis=1)

result['PassengerId'] = result['PassengerId'].astype(int)
result['Survived'] = result['Survived'].apply(convert_survived)

result.to_csv('data/out.csv', sep=',', index=False, mode = 'w')


# Survived
# Pclass,Sex,Age,SibSp,Parch,Fare
#X = preprocessing.normalize(X, axis=0)
#X_test = preprocessing.normalize(X_test, axis=0)



