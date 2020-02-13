import numpy as np


def norm(x):
    temp = x.T - np.mean(x.T, axis=0)
    temp = temp / np.std(temp, axis=0)
    return temp.T


def Extract_Features(xs, ys, zs):

    n = len(xs)  # x.shape[0]
    feature_vector = np.hstack((np.mean(xs), np.mean(
        ys), np.mean(zs), np.std(xs), np.std(ys), np.std(zs)))
    feature_vector = np.hstack((feature_vector, np.mean(abs(
        xs - np.mean(xs))), np.mean(abs(ys - np.mean(ys))), np.mean(abs(zs - np.mean(zs)))))
    feature_vector = np.hstack((feature_vector, np.mean(
        np.sqrt(np.power(xs, 2) + np.power(ys, 2) + np.power(zs, 2)))))
    feature_vector = np.hstack((feature_vector, np.divide(np.histogram(xs, 10), n)[
                               0], np.divide(np.histogram(ys, 10), n)[0], np.divide(np.histogram(zs, 10), n)[0]))
    return feature_vector
