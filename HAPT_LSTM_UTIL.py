# Filename: HAPT_LSTM_UTIL.py
# Dependencies: keras, numpy, os, pandas, sklearn, tensorflow
# Author: Jean-Michel Boudreau
# Date: June 5, 2019

# Import libraries
import os
import numpy as np
import pandas as pd
import keras.layers as kl
import keras.models as km
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# Function to train LSTM RNN
def train_LSTM():
    # Specify some data file names.
    file_name = "HAPT.data.txt"
    datafile = 'HAPT_LSTM.data'
    modelfile = 'HAPT_LSTM.model.h5'
    timesteps = 50

    # If the data have already been processed, then don't do it again,
    # just read it in.
    if (not os.path.isfile('data/' + datafile + '.x_test.npy')):

        # Load in the data
        data = pd.read_csv(file_name)

        # Feature scaling to avoid bias in a single/few feature that are large
        # in magnitude as well as speed up convergence.
        feature_list = data.columns.drop("ID")
        scaler = MinMaxScaler()
        data[feature_list] = scaler.fit_transform(data[feature_list])

        # Seperate the data into 50-timestep 2D arrays such that each instance
        # in the array have an identical label ("ID" column). Each array is
        # generated by sliding down one instance in the complete data and
        # extracting the next 50 timesteps IF all 50 have the same label. Split
        # data into features (x_data) and targets (y_data).
        x_data = []
        y_data = []
        for i in range(0, len(data) - timesteps):
            group_data = data[i: i + timesteps]
            if group_data.ID.nunique() == 1:
                group_x = group_data.drop(labels=['ID'], axis=1)
                group_y = group_data['ID'][i]
                x_data.append([group_x.values])
                y_data.append(group_y)
        # One hot encode the categorical target
        y_data = to_categorical(y_data)
        y_data = np.delete(y_data, 0, 1)

        # Segregate 10 percent of the 50-timestep 2D arrays for test purposes.
        # Use remaining percent for training.
        x_test = []
        y_test = []
        x_train = []
        y_train = []
        for i in range(0, len(x_data)):
            remain = i % 10
            if remain == 0:
                x_test.append(x_data[i])
                y_test.append(y_data[i])
            else:
                x_train.append(x_data[i])
                y_train.append(y_data[i])

        # Reshape arrays for feeding into keras
        x_test = np.array(x_test).reshape((-1, timesteps, 6))
        y_test = np.array(y_test).reshape((-1, 12))
        x_train = np.array(x_train).reshape((-1, timesteps, 6))
        y_train = np.array(y_train).reshape((-1, 12))

        # The processing of the data takes a fair amount of time.  Save
        # the data so we don't have to do this again.
        print('Saving processed data.')
        np.save('data/' + datafile + '.x_test.npy', x_test)
        np.save('data/' + datafile + '.y_test.npy', y_test)
        np.save('data/' + datafile + '.x_train.npy', x_train)
        np.save('data/' + datafile + '.y_train.npy', y_train)

    else:

        # If the data already exists, then use it.

        print('Reading processed data.')
        x_train = np.load('data/' + datafile + '.x_train.npy')
        y_train = np.load('data/' + datafile + '.y_train.npy')

    # If this is our first rodeo, build the model.
    if (not os.path.isfile('data/' + modelfile)):
        print('Building network.')
        model = km.Sequential()
        model.add(kl.LSTM(10, input_shape=(timesteps, 6)))
        model.add(kl.Dense(12, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    else:

        # Otherwise, use the previously-saved model as our starting point
        # so that we can continue to improve it.

        print('Reading model file.')
        model = km.load_model('data/' + modelfile)

    # Fit!  Begin elevator music...
    print('Training Network')
    fit = model.fit(x_train, y_train, epochs=10, batch_size=50)

    # Save the model so that we can use it as a starting point.
    model.save('data/' + modelfile)

# Function to test LSTM RNN
def test_LSTM():
    # Specify some data file names.
    datafile = 'HAPT_LSTM.data'
    modelfile = 'HAPT_LSTM.model.h5'

    # Load the model
    print('Reading model file.')
    model = km.load_model('data/' + modelfile)

    # Load test data
    x_test = np.load('data/' + datafile + '.x_test.npy')
    y_test = np.load('data/' + datafile + '.y_test.npy')

    # Get accuracy of model on test set
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
