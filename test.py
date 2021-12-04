#! /usr/bin/env python

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from src import cluster


def main():
    # Load Data
    data = load_digits().data
    pca = PCA(2)
    df = pca.fit_transform(data)
    print(data.shape)
    print(df.shape)

    print(df[1])
    # df.shape
    # Initialize the class object
    kmeans = KMeans(n_clusters=10)

    # predict the labels of clusters.
    label = kmeans.fit_predict(df)
    print(label[1])
    exit()

    cluster.plot(df, label)

    # print(set(label))

    # # filter rows of original data
    # filtered_label0 = df[label == 0]

    # # plotting the results
    # plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
    # # plt.show()

    # # filter rows of original data
    # filtered_label2 = df[label == 2]

    # filtered_label8 = df[label == 8]

    # # Plotting the results
    # plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='red')
    # plt.scatter(filtered_label8[:, 0], filtered_label8[:, 1], color='black')

    # plt.show()

    # Getting unique labels

    # plotting the results:

    # Getting the Centroids


if __name__ == "__main__":
    main()
