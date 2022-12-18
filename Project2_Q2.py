import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


w = np.array([-7.1, 0.8, 1.8]).T

iris=pd.read_csv("irisdata.csv")
df =pd.DataFrame(iris)
df1 = df[df['species'].str.contains("versicolor")]
df2 = df[df['species'].str.contains("virginica")]
frames = [df1, df2]
df = pd.concat(frames)

def getData():
    # method that returns lengths/widths of each class into own lists
    return

def xTwo(self, xOne):
    xTwo = -(self.w[1]/self.w[2])*xOne-(self.w[0]/self.w[2])
    return xTwo

def classify(xOne, xTwo):
        y = (w[0] + w[1]*xOne + w[2]*xTwo)
        sigmoid = 1 / (1 + np.exp(-y))
        return sigmoid

def plotDecisionBoundary():
    versicolorLengths, versicolorWidths, virginicaLengths, virginicaWidths = getData()
    xOnes = np.linspace(0, 7.5, 75)
    xTwos = []

    for x in xOnes:
        xTwos.append(xTwo(x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(versicolorLengths, versicolorWidths, color='red', label='Versicolor')
    ax.scatter(virginicaLengths, virginicaWidths, color='blue', label='Virginica')
    plt.plot(xOnes, xTwos, color='black')
    plt.fill_between(xOnes, xTwos, 2.6, color='green', alpha=0.1)
    plt.fill_between(xOnes, xTwos, color='orange', alpha=0.1)

    plt.ylabel("Petal Width in cm")
    plt.xlabel("Petal Length in cm")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()
    plt.show()

def surfacePlot():
    length = np.linspace(0, 8, 50)
    width = np.linspace(0, 2.5, 25)  

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    length, width = np.meshgrid(length, width)
    sigmoid = classify(length, width)
    ax.plot_surface(length, width, sigmoid, cmap=matplotlib.cm.Greens)
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 2.6)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel("Petal Length in cm")
    ax.set_ylabel("Petal Width in cm")
    ax.set_zlabel("Value of sigmoid")
    ax.view_init(elev=15, azim=-60)

    plt.show()


sns.relplot(data=df, x='petal_length', y='petal_width', hue='species', aspect=1.81)
plt.show()


#plotDecisionBoundary()


surfacePlot()