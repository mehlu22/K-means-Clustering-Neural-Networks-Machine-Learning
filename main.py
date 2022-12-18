import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random
import seaborn as sns

#Weights vector
w = np.array([-7.1, 0.8, 1.8]).T
stepSize = 0.025

iris=pd.read_csv("irisdata.csv")
df =pd.DataFrame(iris)

df1 = df[df['species'].str.contains("versicolor")]
df2 = df[df['species'].str.contains("virginica")]
frames = [df1, df2]
df = pd.concat(frames)

df3 = pd.DataFrame(df,columns=['petal_length'])
df4 = pd.DataFrame(df,columns=['petal_width'])

X =np.array(pd.DataFrame(df,columns=['petal_length', 'petal_width', 'species']).to_numpy())
# Stores petal lengths in cms
X1 =np.array(pd.DataFrame(df,columns=['petal_length']).to_numpy())
# Stores petal widths in cms
Y1 =np.array(pd.DataFrame(df,columns=['petal_width']).to_numpy())
#Stores the various species types.
Z1 =np.array(pd.DataFrame(df,columns=['species']).to_numpy())

A1 = np.zeros((2, len(X1)))

# The length and width is stored in the form of a vector
vectorizedX = np.zeros((3, len(X1)))
for i in range(len(X1)):
    vectorizedX[0, i] = 1
    vectorizedX[1, i] = X1[i]
    vectorizedX[2, i] = Y1[i]

# function to get all the needed data for versicolor and virginica
def getData():
    versicolorLengths = []
    versicolorWidths = []
    virginicaLengths = []
    virginicaWidths = []
    for s in range(len(Z1)):
        if Z1[s] == 'virginica':  
            virginicaLengths.append(X1[s])
            virginicaWidths.append(Y1[s])
        if Z1[s] == 'versicolor':
            versicolorLengths.append(X1[s])
            versicolorWidths.append(Y1[s])
    return versicolorLengths, versicolorWidths, virginicaLengths, virginicaWidths

for i in range(len(X1)):
            A1[0, i] = X1[i]
            A1[1, i] = Y1[i]

#Weight vector is stored and value of xOne and xTwo is returned
class DecisionBoundary:
    def __init__(self, weight):
        self.w = weight

    def xTwo(self, xOne):
        xTwo = -(self.w[1]/self.w[2])*xOne-(self.w[0]/self.w[2])
        return xTwo

#Function to plot a decision boundary given weight vector using calculation of xOne and xTwo
def plotDecisionBoundary(weight, fill=True, graphTitle=''):
    versicolorLengths, versicolorWidths, virginicaLengths, virginicaWidths = getData()
    xOnes = np.linspace(0, 7.5, 75)
    xTwos = []
    decisionBoundary = DecisionBoundary(weight)
    for x in xOnes:
        xTwos.append(decisionBoundary.xTwo(x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(versicolorLengths, versicolorWidths, color='red', label='Versicolor')
    ax.scatter(virginicaLengths, virginicaWidths, color='blue', label='Virginica')
    plt.plot(xOnes, xTwos, color='black')
    if fill:
        plt.fill_between(xOnes, xTwos, 2.6, color='blue', alpha=0.1)
        plt.fill_between(xOnes, xTwos, color='red', alpha=0.1)
    plt.title(f"Iris Data: {graphTitle}" if (graphTitle != '') else "Iris Data")

    plt.ylabel("Petal Width in cm")
    plt.xlabel("Petal Length in cm")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()
    plt.show()

#Plot 2 decision boundaries
def plotTwoDecisionBoundaries(w1, w2):
    versicolorLengths, versicolorWidths, virginicaLengths, virginicaWidths = getData()
    xOnes = np.linspace(0, 7.5, 75)
    xTwosWeightOne = []
    xTwosWeightTwo = []
    for x in xOnes:
        xTwosWeightOne.append(DecisionBoundary(w1).xTwo(x))
        xTwosWeightTwo.append(DecisionBoundary(w2).xTwo(x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(versicolorLengths, versicolorWidths, color='blue', label='Versicolor')
    ax.scatter(virginicaLengths, virginicaWidths, color='red', label='Virginica')
    plt.plot(xOnes, xTwosWeightOne, color='green')
    plt.plot(xOnes, xTwosWeightTwo, color='orange')
    plt.fill_between(xOnes, xTwosWeightTwo, color='blue', alpha=0.2)
    plt.fill_between(xOnes, xTwosWeightTwo, np.max(xTwosWeightTwo), color='red', alpha=0.1)
    plt.fill_between(xOnes, xTwosWeightOne, color='blue', alpha=0.2)
    plt.fill_between(xOnes, xTwosWeightOne, np.max(xTwosWeightOne), color='red', alpha=0.1)
    plt.title("Iris Data")
    plt.ylabel("Petal Width in cm")
    plt.xlabel("Petal Length in cm")
    plt.xlim(0, 7.5)
    plt.ylim(0.6, 3.0)
    plt.legend()
    plt.show()

#use sigmoid non-linearity to classify petal length and petal width
class Classifier:
    def __init__(self, weight):
        self.w = weight

    def classify(self, xOne, xTwo):
        y = (self.w[0] + self.w[1]*xOne + self.w[2]*xTwo)
        sigmoid = 1 / (1 + np.exp(-y))
        return sigmoid

    def getWeight(self):
        return self.w

    def setWeight(self, w):
        self.w = w    

#3d plot of petal length, width and sigmoid value
def surfacePlot(weight):
    xOne = np.linspace(0, 7, 50)  # Length of petal
    xTwo = np.linspace(0, 2.6, 25)  # Width of petal

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    xOne, xTwo = np.meshgrid(xOne, xTwo)
    sigmoid = Classifier(weight).classify(xOne, xTwo)
    ax.plot_surface(xOne, xTwo, sigmoid, cmap=matplotlib.cm.Greens)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 2.5)
    ax.set_zlim(0, 1.1)
    ax.set_xlabel("Petal Length in cm")
    ax.set_ylabel("Petal Width in cm")
    ax.set_zlabel("Value of sigmoid")
    ax.view_init(elev=16, azim=-60)
    plt.show()

#classifies a data point and checks to see if it is accurate with the given data set
def classify(index, petalLengths, petalWidths, species, weight):
    classVal = Classifier(weight).classify(petalLengths[index], petalWidths[index])
    classSpecies = "versicolor" if classVal < 0.5 else "virginica"
    print('Class:', classVal)
    print("Petal Length:", petalLengths[index], ", Petal Width:", petalWidths[index], ", Class:", species[index], ", Classifier Output:", classSpecies)

#calculates mean square error of prediction and actual value
def mse(vector, weight, species):
    tot = 0
    for i in range(len(vector[0])):
        prediction = Classifier(weight).classify(vector[0, i], vector[1, i])
        classVal = 0 if species[i] == "versicolor" else 1
        tot += (classVal - prediction)**2
    return tot/len(vector[0])   

#same as mse() but with different parameters
def mseTwo(X, m, b, species):
    w = np.array([b, m[0], m[1]]).T
    mse = 0
    for i in range(2):
        xOne = X[0, i]
        xTwo = X[1, i]
        print(X)
        prediction = minimalClassify(w, xOne, xTwo)
        classVal = 0 if species[i] == "versicolor" else 1
        mse += (classVal - prediction)**2
    return mse/len(X[0, :])

#same as classify() but different parameters
def minimalClassify(w, xOne, xTwo):
    y = (w[0] + w[1]*xOne + w[2]*xTwo)
    sigmoid = 1 / (1 + np.exp(y))
    return sigmoid

#returns a vector of the gradient of each index of the input data vector
def summedGradient(vectorizedX, w, species):
    gradient = np.zeros(3) #gradient is in form <#,#,#>
    for i in range(len(vectorizedX[0, :])):
        vector = vectorizedX[:, i]  # column data vector
        y = 0.0 if species[i] == "versicolor" else 1.0
        sigmoid = 1 / (1 + math.exp(-np.dot(w, vector)))
        gradient[0] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * vector[0]
        gradient[1] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * vector[1]
        gradient[2] += -2 * (y - sigmoid) * (sigmoid * (1 - sigmoid)) * vector[2]
    return gradient

"""
Uses gradient descent to find optimal decision boundary
Plots the Initial, Intermediate, & Converged decision boundaries as well MSE over iterations at these points
"""
def optimalDecisionBoundary(classifier, stepSize, vectorX, species):
    threshold = 0.25
    gradient = 0

    iterations = 0
    weights = []
    mses = []
    while True:
        iterations += 1  # for graphing
        gradient = summedGradient(vectorX, classifier.getWeight(), species)
        weight = classifier.getWeight()-stepSize*gradient
        mses.append(mse(vectorX[1:3, :], weight, species))  # for graphing
        weights.append(weight)  # for graphing
        classifier.setWeight(weight)
        if np.linalg.norm(gradient) < threshold:  # convergence condition
            break
    plots = {'Initial': 0, 'Intermediate': int(iterations/2), 'Converged': iterations-1}
    for plot in plots:
        plotDecisionBoundary(fill=False, graphTitle=plot, 
        weight=weights[plots[plot]])
        if plot != 'Initial':
            MSEGraph(mses[0:plots[plot]], plots[plot], graphTitle=plot)

    return classifier.getWeight()

#Generates a graph that shows MSE over Iterations
def MSEGraph(mse, iterations, graphTitle = ''):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(range(1, iterations+1), mse, color='green')
    plt.title(f"MSE vs Iterations {graphTitle}" if (graphTitle != '') else "MSE vs Iterations")
    plt.ylabel("MSE")
    plt.xlabel("Iterations")
    plt.show()

#Generates a random weight vector
def randomWeightGenerator():
    weights = np.zeros(3)
    weights[0] = random.uniform(-10, -1)
    weights[2] = (-1 * random.uniform(-10, -1))/random.uniform(-0.1, 3.5)
    comp = random.uniform(abs(weights[0])/30, abs(weights[0])/4)
    weights[1] = comp if weights[2] > 0 else (-1*comp)
    return weights

def answer_2a():
    sns.relplot(data=df, x='petal_length', y='petal_width', hue='species', aspect=1.51)
    plt.show()

def answer_2c():
    plotDecisionBoundary(w)

def answer_2d():
    surfacePlot(w)

def answer_2e():
    print("Data points of virginica petals")
    classify(54, X1, Y1, Z1, w)
    classify(72, X1, Y1, Z1, w)
    classify(86, X1, Y1, Z1, w)
    print("Data points of versicolor petals")
    classify(0, X1, Y1, Z1, w)
    classify(8, X1, Y1, Z1, w)
    classify(22, X1, Y1, Z1, w)
    print("data points on / around the decision boundaries")
    classify(56, X1, Y1, Z1, w)
    classify(27, X1, Y1, Z1, w)
    classify(33, X1, Y1, Z1, w)

def answer_3b():
    MSE = mse(A1, w, Z1)
    print('MSE of First Decision Boundary:', MSE)
    w2 = np.array([-6, 1.0, 2.1]).T
    plotTwoDecisionBoundaries(w, w2)
    MSE = mse(A1, w2, Z1)
    print('MSE of Second Decision Boundary:', MSE)

def answer_3e():
    weights = np.copy(w)
    print("weights", weights)
    plotDecisionBoundary(graphTitle="First", weight=weights)
    gradientSummed = summedGradient(vectorizedX, weights, Z1)
    gradientWeight = weights - stepSize*gradientSummed
    print('gradient summed', gradientWeight)
    print('new weights', gradientWeight)
    plotDecisionBoundary(graphTitle="Second", weight=gradientWeight)

def answer_4c():
    weights = randomWeightGenerator()
    classifier = Classifier(weight=weights)
    print('Starting Weight', classifier.getWeight())
    convergedWeight = optimalDecisionBoundary(classifier, stepSize, vectorizedX, Z1)
    print('Final Weight', convergedWeight)



answer_2a()
answer_2c()
answer_2d()
answer_2e()
answer_3b()
answer_3e()
answer_4c()