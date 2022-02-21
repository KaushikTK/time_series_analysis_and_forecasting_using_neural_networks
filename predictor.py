from tensorflow.keras.models import load_model
from matplotlib import pyplot
from keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def loadTrainedModel(modelPath): return load_model(modelPath)

def plot(preds, actual, plotName):
    pyplot.plot(preds, label='prediction', color='b', linestyle='--')
    pyplot.plot(actual, label='actual', color='r')
    pyplot.legend()
    pyplot.savefig(plotName + '.png')
    pyplot.show()

def trainLSTM(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, epochs=100, batch_size=70, validation_data=(X_test, Y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=20)], shuffle=False, verbose=2)
    return model

def perform_minmax_scaling(values, low=0, high=1):
    global SCALER
    SCALER = MinMaxScaler(feature_range=(low, high))
    values = SCALER.fit_transform(values.reshape(-1, 1))
    data = []
    for i in values: data.append(i[0])
    return data

def split_dataset(data, trainSize=0.8):
    train_size = int(len(data)*trainSize)
    return data[:train_size], data[train_size:len(data)]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

def main():
    model = loadTrainedModel('Mahindra_Bidirectional_LSTMmodel.h5')
    


main()