
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

    training_data = pd.read_csv("D:/西二Python/DigitalRecognizerData/train.csv")
    testing_data = pd.read_csv("D:/西二Python/DigitalRecognizerData/test.csv")
    print(training_data.columns)
    train, validation = train_test_split(training_data)
    X_columns = list(training_data.columns)
    X_columns.remove("label")

    train_y = train["label"].values
    train_X = train[X_columns].values

    val_y = validation["label"].values
    val_X = validation[X_columns].values

    print(np.max(train_X))

    train_X = train_X / 255
    val_X = val_X / 255
    print(np.max(train_X))

    clf = MLPClassifier(hidden_layer_sizes=(100))
    clf.fit(train_X, train_y)

    predictions = clf.predict(val_X)
    from sklearn.metrics import accuracy_score

    acc_score = accuracy_score(val_y, predictions)
    print("Accuracy Score: {0}".format(acc_score))

    n_values = testing_data.shape[0]  # number of rows
    test = testing_data.values / 255
    test

    test_preds = clf.predict(test)
    index = [i for i in range(1, n_values + 1)]
    df = pd.DataFrame({"ImageID": index, "Label": test_preds})

    df.to_csv("submission.csv", index=False)
    df
