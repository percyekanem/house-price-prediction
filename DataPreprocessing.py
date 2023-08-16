"""
This central module contains functions used in the data preprocessing steps of this project. The selectPC function identifies the principal components of a dataset and returns a dataset with reduced dimensionality up to a threshold. The crossval function uses the k-fold method to cross-validate the performance of a model on a dataset using the rmse values.
"""
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def selectPC(data, threshold):
    pca = PCA()
    pca.fit(data)
    variance = np.cumsum(pca.explained_variance_ratio_) #cumulative variance of all the data features.

    features = np.argmax(variance >= threshold)+1 #argmax finds the first index where the condition is met. +1 bc index starts at 0.
    final_pca = PCA(n_components=features)

    final = final_pca.fit_transform(data)

    print(f"At least {int(threshold*100)}% of the variation in the dataset can be explained by {features} feature(s).")
    return final


def crossval(model, x, y, k):
    cv = KFold(n_splits=k, shuffle=True, random_state=26, )

    predicted_values = cross_val_predict(model, x, y, cv=cv)

    residuals = y - predicted_values

    rmse_scores = np.sqrt(metrics.mean_squared_error(y, predicted_values))

    print("Cross-Validated Root Mean Squared Error:", rmse_scores)

    
    plt.figure(figsize=(10, 4))
    
    # Residual Histogram
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
