from pandas import read_csv, DataFrame
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from tensorflow.keras.models import load_model


SCALER = MinMaxScaler()
undifferenced = []

def readCSV(path:str, companies):
    df = read_csv(path)
    vals = {}
    for i in companies: vals[i] = list(df[i].dropna())
    return DataFrame(vals, columns=vals.keys())


def dickeyFullerTest(data):
    return adfuller(data)


def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)): diff.append(dataset[i] - dataset[i-interval])
    return diff


def inverse_difference(last_obs, value):
    return last_obs + value


def perform_inverse_transform(values):
    global SCALER
    return SCALER.inverse_transform(values)


def check_dickeyFuller_for_different_intervals(data, start = 1, end = 50):
    for i in range(start, end, 1):
        vals = difference(data, i)
        print(str(i) + ': ' + str(dickeyFullerTest(vals)[1]))


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


def createLSTMmodel(inputShape, model_type='stacked'):
    model = createStackedLSTMmodel(inputShape)
    if(model_type == 'bidirectional'): model = createBidirectionalLSTMmodel(inputShape)
    #elif(model_type == 'cnn'): model = 

    return model


def createStackedLSTMmodel(inputShape):
    model = Sequential()
    model.add(LSTM(100, input_shape=inputShape,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


def createBidirectionalLSTMmodel(inputShape):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=inputShape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


def createCnnLSTMmodel(inputShape):
    model = Sequential()
    #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

def trainLSTM(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, epochs=100, batch_size=70, validation_data=(X_test, Y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=20)], shuffle=False, verbose=2)
    return model


def justTrain(model, X, Y):
    model.fit(X, Y, epochs=50, batch_size=30, shuffle=False, verbose=2)
    return model


def plot(preds, actual, plotName):
    pyplot.plot(preds, label='prediction', color='b', linestyle='--')
    pyplot.plot(actual, label='actual', color='r')
    pyplot.legend()
    pyplot.savefig(plotName + '.png')
    pyplot.show()


def plotExtrapolateData(data, plotName):
    pyplot.plot(data, label='prediction', color='r')
    pyplot.savefig(plotName + '.png')


def TRAIN():
    companies = ['MAHINDRA','WIPRO','INFOSYS','EXIDE']
    data = readCSV('dataset.csv', companies)

    vals = data.MAHINDRA.values

    #print('p value without differencing: ' + str(dickeyFullerTest(vals)[1]))
    #check_dickeyFuller_for_different_intervals(vals, 1, 50)

    #global undifferenced
    #undifferenced = np.array(vals)

    #stationaryData = np.array(difference(vals, 1)) # differencing performed
    stationaryData = np.array(vals) # no differencing performed
    
    scaledStationaryData = perform_minmax_scaling(stationaryData, 0, 1)

    train, test = split_dataset(scaledStationaryData, 0.9)

    look_back = 15

    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    model = createLSTMmodel((look_back,1), 'bidirectional')
    #print(model.summary())

    model = trainLSTM(model, X_train, Y_train, X_test, Y_test)

    train_prediction = perform_inverse_transform(model.predict(X_train))
    test_prediction = perform_inverse_transform(model.predict(X_test))

    Y_train = perform_inverse_transform(Y_train)
    Y_test = perform_inverse_transform(Y_test)

    # do this if u have done differencing
    #Y_train = [inverse_difference(undifferenced[i], Y_train[i]) for i in range(len(Y_train))]
    #Y_test = [inverse_difference(undifferenced[len(Y_train)+i], Y_test[i]) for i in range(len(Y_test))]

    #train_prediction = [inverse_difference(undifferenced[i], train_prediction[i]) for i in range(len(train_prediction))]
    #m = [train_prediction[-1]]
    #for i in range(len(test_prediction)): m.append(m[-1]+test_prediction[i])
    #test_prediction = m[1:]



    #preds = np.concatenate((train_prediction, test_prediction))
    #actual = np.concatenate((Y_train, Y_test))
    preds, actual = test_prediction, Y_test

    plot(preds, actual, 'Mahindra_Bidirectional')

    model.save('Mahindra_Bidirectional_LSTMmodel.h5')

    return


def TRAIN_ON_TEST_DATA():
    model = load_model('Mahindra_Bidirectional_LSTMmodel.h5')
    vals = readCSV('dataset.csv', ['MAHINDRA']).MAHINDRA.values

    stationaryData = np.array(vals) # no differencing performed
    scaledStationaryData = perform_minmax_scaling(stationaryData, 0, 1)
    _, test = split_dataset(scaledStationaryData, 0.9)
    look_back = 15
    X_test, Y_test = create_dataset(test, look_back)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    model = justTrain(model, X_test, Y_test)

    model.save('FullyTrainedMahindraBidirectional.h5')

    print(X_test[-1])


def EXTRAPOLATE():
    model = load_model('FullyTrainedMahindraBidirectional.h5')

    #X = [[0.95205623, 0.92026857, 0.9284865, 0.93698419, 0.95177647, 0.9470905, 0.93894251, 0.95219611, 0.94146034, 0.89834243, 0.9048818, 0.90264373, 0.85134285, 0.7951112, 0.78633375]]

    #X = np.array(X)

    #pred = model.predict(X)

    #print(pred[0][0])

    preds = []
    X = [0.95205623, 0.92026857, 0.9284865, 0.93698419, 0.95177647, 0.9470905, 0.93894251, 0.95219611, 0.94146034, 0.89834243, 0.9048818, 0.90264373, 0.85134285, 0.7951112, 0.78633375]
    for _ in range(0,180,1):
        pred = model.predict(np.array([X]))
        preds.append(pred[0][0])
        X.append(pred[0][0])
        X = X[1:]

    data = readCSV('dataset.csv', ['MAHINDRA'])
    vals = np.array(data.MAHINDRA.values)

    global SCALER
    SCALER = MinMaxScaler(feature_range=(0, 1)).fit(vals.reshape(-1, 1))

    x = []
    for i in preds: x.append(np.array(i))
    x = np.array(x)

    values = SCALER.inverse_transform(x.reshape(-1, 1))
    print(values)
    #plotExtrapolateData(preds,'mahindra_for_next_6mo_normalised')
    plotExtrapolateData(list(values),'mahindra_for_next_6mo')



EXTRAPOLATE()
