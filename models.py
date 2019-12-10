import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K

def trainLogReg(X_train, y_train):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    print("Log reg training set accuracy: %f" % logisticRegr.score(X_train, y_train))
    return logisticRegr

def trainNeuralNet(X_train, y_train):
    import os

    # THIS IS NEEDED TO PREVENT CRASHING
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    nn = Sequential()
    nn.add(Dense(5, activation='relu'))
    # nn.add(Dense(120, activation='relu'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])#,f1_m,precision_m, recall_m])
    #nn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc',f1_m])#,precision_m, recall_m])
    hnn = nn.fit(X_train, y_train, epochs=30)

    print("Neural net raining set accuracy: %f" % np.sqrt(hnn.history['acc'][-1]))
    return nn

    # print(hnn.history)  Can be used to find keys
    #    hnn[loss]   hnn[acc]   hnn[f1_m]   hnn[precision_m]    hnn[recall_m]
    #print("Training set recall: %f" % np.sqrt(hnn.history['recall_m'][-1]))
    #print("Training set precision: %f" % np.sqrt(hnn.history['precision_m'][-1]))
    #print("Training set f1_score: %f" % np.sqrt(hnn.history['f1_m'][-1]))
    #print("Test set f1_score: %f" % np.sqrt(f1_score))

def evaluateLogReg(model, X_test, y_test, save_path):
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    print("\n\n")
    print("***LOG REG RESULTS***")
    plot_confusion_matrix(y_test, y_pred, classes=["Non-striped", "Striped"], save_path=save_path)
    print("Test set accuracy:     %f" % model.score(X_test, y_test))
    print("Test set true positives:     %i" % tp)
    print("Test set false positives:     %i" % fp)
    print("Test set false negatives:     %i" % fn)
    print("Test set true negatives:     %i" % tn)
    print("Test set precision: %f" % (precision))
    print("Test set recall: %f" % (recall))
    print("\n\n")

def evaluateNN(model, X_test, y_test, save_path):
    # Test set values
    ##loss, accuracy,f1_score, precision, recall = model.evaluate(X_test, y_test, batch_size=500, verbose=0)
    #loss, accuracy, f1_score = model.evaluate(X_test, y_test, batch_size=500, verbose=0)
    loss, accuracy  = model.evaluate(X_test, y_test, batch_size=500, verbose=0)

    y_pred = model.predict_classes(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    print("\n\n")
    print("***Neural Net RESULTS***")
    plot_confusion_matrix(y_test, y_pred, classes=["Non-striped", "Striped"], save_path=save_path)
    print("Test set accuracy:     %f" % np.sqrt(accuracy))
    print("Test set true positives:     %i" % tp)
    print("Test set false positives:     %i" % fp)
    print("Test set false negatives:     %i" % fn)
    print("Test set true negatives:     %i" % tn)
    print("Test set precision: %f" % (precision))
    print("Test set recall: %f" % (recall))

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

def plot_confusion_matrix(y_true, y_pred, classes, save_path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(save_path)
    else:
        print(save_path)

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    return ax


def trainModels(features, labels):
    X_set, X_test, y_set, y_test = train_test_split(features, labels, test_size=.2, random_state=420)
    X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=.2, random_state=69)
    logReg = trainLogReg(X_train, y_train)
    evaluateLogReg(logReg, X_val, y_val, "log-reg-validation.png")
    evaluateLogReg(logReg, X_test, y_test, "log-reg-test.png")
    nn = trainNeuralNet(X_train, y_train)
    evaluateNN(nn, X_val, y_val, "nn-validation.png")
    evaluateNN(nn, X_test, y_test, "nn-test.png")