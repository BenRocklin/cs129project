import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression


def trainLogReg(X_train, y_train, X_test, y_test):
    logisticRegression = LogisticRegression()
    logisticRegression.fit(X_train, y_train)

def trainNeuralNet(X_train, y_train, X_test, y_test):
    print("\nTraining neural net.")
    nn = Sequential()
    nn.add(Dense(10, activation='sigmoid'))
    nn.add(Dense(10, activation='sigmoid'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    hnn = nn.fit(X_train, y_train, epochs=2, batch_size=500)
    print("Done training.")
    print("Training set accuracy: %f" % np.sqrt(hnn.history['accuracy'][-1]))
    print("Test set accuracy:     %f" % np.sqrt(nn.evaluate(x=X_test, y=y_test, verbose=0)[1]))

def trainModels(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
    trainLogReg(X_train, y_train, X_test, y_test)
    #trainNeuralNet(X_train, y_train, X_test, y_test)

