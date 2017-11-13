#!/usr/bin/env python3

import csv
import numpy as np

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
types = []
train = []
test = []

tickets = set()
cabins = set()
embarked = set()

with open('data/train.csv', newline='') as f:
    file = csv.reader(f, delimiter=',', quotechar='"')

    for i, line in enumerate(file):
        if i == 0:
            types = line
            continue

        args = np.array([])

        for k, x in enumerate(line):
            t = types[k]

            if t == "Name":
                continue
            elif t == "Ticket":
                tickets.add(x)
                continue
            elif t == "Cabin":
                cabins.add(x)
                continue
            elif t == "Embarked":
                embarked.add(x)
                continue

            if x == '':
                args = np.append(args, -1)
                continue

            if t == "Survived":
                args = np.append(args, int(1 if int(x) == 1 else -1))
            if t == "Pclass" or t == "SibSp" or t == "Parch":
                args = np.append(args, int(x))
            elif t == "Sex":
                args = np.append(args, int(x == "male"))
            elif  t == "Age" or t == "Fare":
                args = np.append(args, float(x))

        if len(args) == 0:
            continue

        if i < 500:
            train.append(args)
        else:
            test.append(args)

# Survived
# Pclass,Sex,Age,SibSp,Parch,Fare

#print(train)

w = []
for x in train[0][1:]:
    w.append(0.0)

tetta  = 100
for t in train:
    x = t[1:]
    y = t[0]
    model = np.dot(w, x)
    if model * y <= 0:
        w += np.dot(x, tetta * y)
        #print('+ w = ', w, ', y = ', y, ', model = ', model, ', x = ', x)
    #else:
        #print('  w = ', w, ', y = ', y, ', model = ', model, ', x = ', x)
    #w -= np.dot(args, tetta * (model - answer))
    #print('new w = ', w, '\n')

all = 0
ok = 0
for t in test:
    x = t[1:]
    y = t[0]
    model = np.dot(w, x)
    if model * y <= 0:
        print('+ w = ', w, ', y = ', y, ', model = ', model, ', x = ', x)
    else:
        print('  w = ', w, ', y = ', y, ', model = ', model, ', x = ', x)
        ok += 1
    all += 1

print('result: ', ok, '/', all, ' = ', int(ok / all * 100), '%')

#for t in train:
#    print(t)
print('Hello World!')