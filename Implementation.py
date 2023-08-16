"""
This central module contains functions which execute all steps of the machine learning and deep learning models used in this project.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn import metrics


def LR_Model(x_train, y_train, x_test, y_test, ytest_range):
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    lm_predict = lm.predict(x_test)

    residuals = y_test - lm_predict

    rmse = np.sqrt(metrics.mean_squared_error(y_test, lm_predict))
    nrmse = rmse / ytest_range
    r2 = metrics.r2_score(y_test, lm_predict)

    print(f"RMSE: {rmse}; NRMSE: {nrmse}; R2: {r2}")

    # Residual Histogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Residual Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Residual Box Plot
    plt.subplot(1, 2, 2)
    plt.boxplot(residuals)
    plt.title('Residual Box Plot')
    plt.ylabel('Residuals')

    plt.tight_layout()
    plt.show()

    return lm_predict



def MLPModel(x_train, y_train, x_test, y_test, hidden_layer_size, epochs, ytest_range, l2_strength=0.01):
    num_features = x_train.shape[1]
    mlp = Sequential()
    mlp.add(Dense(num_features, input_dim=num_features, activation='relu', kernel_regularizer=l2(l2_strength)))
    for i in hidden_layer_size:
        mlp.add(Dense(i, activation='relu', kernel_regularizer=l2(l2_strength)))
    mlp.add(Dense(1, activation=None))
    mlp.compile(loss='mean_squared_error', optimizer='adam')

    mlp.fit(x_train, y_train, epochs = epochs, validation_split=0.2, verbose =2)
    mlp_pred = mlp.predict(x_test)
    
    rmse = np.sqrt(metrics.mean_squared_error(y_test, mlp_pred))
    nrmse = rmse / ytest_range
    r2 = metrics.r2_score(y_test, mlp_pred)

    print(f"RMSE: {rmse}; NRMSE: {nrmse}; R2: {r2}")
    
    return mlp_pred



def RMLPModel(x_train, y_train, x_test, y_test, hidden_layer_size, epochs, ytest_range, l2_strength=0.01):
    num_features = x_train.shape[1]
    num_layers = len(hidden_layer_size)

    rmlp = Sequential()

    for index in range(num_layers):
        rmlp.add(Dense(hidden_layer_size[index], activation='relu', input_dim=num_features,
                       kernel_regularizer=l2(l2_strength)))

        if index > 0:
            for prev_index in range(layer):
                rmlp.add(Dense(hidden_layer_size[index], activation='relu',
                               kernel_regularizer=l2(l2_strength)))

    rmlp.add(Dense(1, activation=None))

    rmlp.compile(loss='mean_squared_error', optimizer='adam')

    rmlp.fit(x_train, y_train, epochs=epochs, validation_split=0.2, verbose=2)

    rmlp_pred = rmlp.predict(x_test)

    rmse = np.sqrt(metrics.mean_squared_error(y_test, rmlp_pred))
    nrmse = rmse / ytest_range
    r2 = metrics.r2_score(y_test, rmlp_pred)

    print(f"RMSE: {rmse}; NRMSE: {nrmse}; R2: {r2}")

    return rmlp_pred
    