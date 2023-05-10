import matplotlib.pyplot as plt

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

def plot2Features(feature1, feature2, normalizationFunction, feature1Name = "GDP", feature2Name = "Freedom"):
    plt.plot(feature1, feature2, 'ro', label='raw data')
    nF1 = normalizationFunction(feature1)
    nF2 = normalizationFunction(feature2)
    plt.plot(nF1, nF2,'b^', label="Normalized")
    plt.legend()
    plt.xlabel(feature1Name)
    plt.ylabel(feature2Name)
    plt.show()

# plotDataHistogram([1.61646318435669, 1.48238301277161, 1.480633020401, 1.56497955322266, 1.44357192516327], 'Capita GDP')
# plotDataHistogram([7.53700017929077, 7.52199983596802, 7.50400018692017, 7.49399995803833, 7.4689998626709], "Happiness score")