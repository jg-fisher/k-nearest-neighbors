"""
Supervised algorithm for regression or classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import operator

x = np.array([[1,1],
    [2,2],
    [3,3],
    [2, 2]])

y = np.array([1, 2, 3, 2])

class DataPoint:
    def __init__(self, x, label, dist=None):
        self.x = x
        self.label = label
        self.dist = dist


class KNN:
    def __init__(self, x, y, k=2):
        self.x = x
        self.y = y
        self.k = k

        self.data = [DataPoint(x, y) for x, y in zip(self.x, self.y)]

        self.n_plot = 0

    def euclidean_distance(self, x, y):
        """
        Calculates euclidean distance between x and y
        """
        dist = np.linalg.norm(x-y)
        return dist

    def predict(self, x):
        """
        Predicts class label for x.
        """

        # calculate the distance for each point
        for p in self.data:
            p.dist = self.euclidean_distance(x, p.x)

        # k nearest neighbors
        self.k_neighbors = sorted(self.data, key=lambda p: p.dist)[:3]

        # output probabilites for k nearest neighbors
        n = { p.label:0 for p in self.data }
        for p in self.k_neighbors:
            n[p.label] += 1

        # class with max probability
        return max(n.items(), key=operator.itemgetter(1))[0]

    def vis(self, show=False):

        x_cor = [x[0] for x in self.x]
        y_cor = [y[1] for y in self.x]

        plt.figure(self.n_plot)
        plt.scatter(x_cor, y_cor)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('KNN')
        plt.show() if show else 0

if __name__ == '__main__':
    knn = KNN(x, y)
    print('Predicting: [1, 1] Output: {}'.format(knn.predict([1,1])))
    #knn.vis(show=True)
