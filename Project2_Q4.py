import Project2_Q1_backup
import Project2_Q3

import numpy as np
import matplotlib.pyplot as plt
import math
import random


def fit(classifier, step_size, X_augmented, petal_length, petal_width, species, progress_output=False):
    threshold = 0.25
    gradient = 0
    previous_gradient = 0

    # Parameters for Graphing/Output:
    num_iterations = 0
    weights_store = []
    mse_store = []
    while True:
        num_iterations += 1
        gradient = Project2_Q3.gradient_mse(X_augmented, classifier.get_weights(), species)
        new_weights = classifier.get_weights()-step_size*gradient
        if progress_output:
            mse_store.append(Project2_Q3.mse(X_augmented[1:3, :], new_weights[1:3], new_weights[0], species))  # for graphing
        weights_store.append(new_weights)  # for graphing
        classifier.set_weights(new_weights)
        if np.linalg.norm(gradient) < threshold:  # convergence condition
            break
        previous_gradient = gradient
    print(num_iterations)
    if progress_output:  # is true
        plot_locations = {'Initial': 0, 'Middle': math.floor(num_iterations/2), 'Final': num_iterations-1}
        for key in plot_locations:
            Project2_Q1_backup.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, fill=False, subtitle=key,
                                                               w=weights_store[plot_locations[key]])
            if key != 'Initial':
                plot_loss_over_iterations(mse_store[0:plot_locations[key]], plot_locations[key], subtitle=key)

    return classifier.get_weights()


def plot_loss_over_iterations(mse_store, num_iterations, subtitle = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(range(1, num_iterations+1), mse_store, color='indigo')
    if subtitle == None:
        plt.title("MSE vs. Number of iterations")
    else:
        plt.title("MSE (loss) vs. Number of iterations: {st}".format(st=subtitle))
    plt.ylabel("MSE")
    plt.xlabel("Number of iterations")
    # plt.xlim(0, 7.5)
    plt.show()


def plot_decision_boundaries_over_iterations(petal_length, petal_width, species, weights_store, num_iterations, skip_size=50):
    # Creating two datasets to plot
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

    # Setting up figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Plotting data points:
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    # Drawing Multiple Decision Boundaries:
    for i in range(len(weights_store)):
        current_weights_vector = weights_store[i]
        if i % skip_size == 0:
            x_ones = np.linspace(0, 7.5, 75)
            x_twos = []
            iris_decision_boundary = Project2_Q1_backup.decision_boundary(w=current_weights_vector)
            for x_one in x_ones:
                x_twos.append(iris_decision_boundary.get_x_two(x_one))
            plt.plot(x_ones, x_twos, color='black', alpha=0.05)
    # Labeling:
    plt.title("Iris Data")
    plt.ylabel("Petal Width (cm) [x\u2082]")
    plt.xlabel("Petal Length (cm) [x\u2081]")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()

    plt.show()


def random_weights():
    x_one_min = 4.0
    x_one_max = 30.0
    x_two_min = 0.9
    x_two_max = 2.5
    """My way of ensuring that the random weights still plot is by making sure one of the intercepts is within its
    range:"""
    w_random = np.zeros(3)
    x_two_intercept = random.uniform(x_two_min-1, x_two_max+1)
    w_random[0] = random.uniform(-10, -1)
    w_random[2] = -w_random[0]/x_two_intercept
    if w_random[2] > 0:
        w_random[1] = random.uniform(abs(w_random[0])/x_one_max, abs(w_random[0])/x_one_min)
    else:
        w_random[1] = -random.uniform(abs(w_random[0]) / x_one_max, abs(w_random[0]) / x_one_min)
    # multiplying by a random scalar factor:
    # w_random = w_random * random.uniform(1, 10)
    return w_random