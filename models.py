import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K

def trainLogReg(X_train, y_train, X_test, y_test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)

    y_pred = logisticRegr.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    print("\n\n")
    print("***LOG REG RESULTS***")
    print("Training set accuracy: %f" % logisticRegr.score(X_train, y_train))
    print("Test set accuracy:     %f" % logisticRegr.score(X_test, y_test))
    print("Test set true positives:     %i" % tp)
    print("Test set false positives:     %i" % fp)
    print("Test set false negatives:     %i" % fn)
    print("Test set true negatives:     %i" % tn)
    print("Test set precision: %f" % (precision))
    print("Test set recall: %f" % (recall))
    print("\n\n")



def trainNeuralNet(X_train, y_train, X_test, y_test):
    import os

    # THIS IS NEEDED TO PREVENT CRASHING
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    nn = Sequential()
    nn.add(Dense(10, activation='sigmoid'))
    nn.add(Dense(10, activation='sigmoid'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])#,f1_m,precision_m, recall_m])
    hnn = nn.fit(X_train, y_train, epochs=2, batch_size=500)
    
    # Test set values
    #loss, accuracy,f1_score, precision, recall = nn.evaluate(X_test, y_test, batch_size=500, verbose=0)
    loss, accuracy  = nn.evaluate(X_test, y_test, batch_size=500, verbose=0)

    y_pred = nn.predict_classes(X_test)

    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    print("\n\n")
    print("***Neural Net RESULTS***")
    print("Training set accuracy: %f" % np.sqrt(hnn.history['acc'][-1]))
    print("Test set accuracy:     %f" % np.sqrt(accuracy))
    print("Test set true positives:     %i" % tp)
    print("Test set false positives:     %i" % fp)
    print("Test set false negatives:     %i" % fn)
    print("Test set true negatives:     %i" % tn)
    print("Test set precision: %f" % (precision))
    print("Test set recall: %f" % (recall))

    # print(hnn.history)  Can be used to find keys
    #    hnn[loss]   hnn[acc]   hnn[f1_m]   hnn[precision_m]    hnn[recall_m]
    #print("Training set recall: %f" % np.sqrt(hnn.history['recall_m'][-1]))
    #print("Training set precision: %f" % np.sqrt(hnn.history['precision_m'][-1]))
    #print("Training set f1_score: %f" % np.sqrt(hnn.history['f1_m'][-1]))
    #print("Test set f1_score: %f" % np.sqrt(f1_score))



def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def trainModels(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
    trainLogReg(X_train, y_train, X_test, y_test)
    trainNeuralNet(X_train, y_train, X_test, y_test)