import Project2_Q1_backup

import numpy as np
import matplotlib.pyplot as plt
import math


# Functions written in Part 2:


def mse(X, m, b, species):
    iris_classifer = Project2_Q1_backup.simple_classifier(m=m, b=b)
    mse_sum = 0
    for i in range(len(X[0, :])):
        x_one = X[0, i]
        x_two = X[1, i]
        prediction = iris_classifer.classify(x_one, x_two)
        ground_truth = None
        if species[i] == "versicolor":
            ground_truth = 0
        elif species[i] == "virginica":
            ground_truth = 1
        mse_sum += (ground_truth - prediction)**2

    return mse_sum/len(X[0, :])


def gradient_mse(X_augmented, w, species):
    gradient_sum = np.zeros(3)
    for i in range(len(X_augmented[0, :])):
        x_vector = X_augmented[:, i]  # column data vector
        y = None
        if species[i] == "versicolor":
            y = 0.0
        elif species[i] == "virginica":
            y = 1.0

        sigmoid = 1 / (1 + math.exp(-np.dot(w, x_vector)))
        gradient_sum[0] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * x_vector[0]
        gradient_sum[1] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * x_vector[1]
        gradient_sum[2] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * x_vector[2]
    return gradient_sum


def plot_iris_data_with_two_decision_boundaries(petal_length, petal_width, species, m_one, b_one, m_two, b_two):
    versicolor_petal_length = []
    versicolor_petal_width = []
    virginica_petal_length = []
    virginica_petal_width = []
    for l, w, s in zip(petal_length, petal_width, species):
        if s == 'versicolor':
            versicolor_petal_length.append(l)
            versicolor_petal_width.append(w)
        elif s == 'virginica':
            virginica_petal_length.append(l)
            virginica_petal_width.append(w)
    # drawing the line
    x_ones = np.linspace(0, 7.5, 75)
    x_twos_one = []
    x_twos_two = []
    iris_decision_boundary_one = Project2_Q1_backup.decision_boundary(m_one, b_one)
    iris_decision_boundary_two = Project2_Q1_backup.decision_boundary(m_two, b_two)
    for x_one in x_ones:
        x_twos_one.append(iris_decision_boundary_one.get_x_two(x_one))
        x_twos_two.append(iris_decision_boundary_two.get_x_two(x_one))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    plt.plot(x_ones, x_twos_one, color='black')
    plt.plot(x_ones, x_twos_two, color='red')
    plt.fill_between(x_ones, x_twos_two, color='indigo', alpha=0.3)
    plt.fill_between(x_ones, x_twos_two, np.max(x_twos_two), color='orchid', alpha=0.2)
    plt.fill_between(x_ones, x_twos_one, color='indigo', alpha=0.2)
    plt.fill_between(x_ones, x_twos_one, np.max(x_twos_one), color='orchid', alpha=0.2)
    # plt.fill_between(x_ones, x_twos, color='indigo', alpha=0.15) # np.max(x_twos_two)
    plt.title("Iris Data")
    plt.ylabel("Petal Width in cms [x\u2082]")
    plt.xlabel("Petal Length in cms [x\u2081]")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()
    plt.show()
