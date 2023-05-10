from math import sqrt

import numpy as np

from graphics.graphic_utils import plotDataHistogram
from normalization.normal import getMinMaxParameters, getStatisticalParameters, minMaxScalingParam, \
    statisticalScalingParam, minMaxScaling
from regression.batch_gradient_descent import BGDRegression
from runModes.multiVar import evalSingleTargetRegression
from utils.utils import loadData


def uniVar(path, scaling = 0, normalizationFunction = None):

    print("\nUNIVARIABLE COD PROPRIU / SCALING : " + str(scaling) + "\n")
    inputsGDP, outputs = loadData(path,
                        "Economy..GDP.per.Capita.", "Happiness.Score")
    inputsFreedom, outputs = loadData(path, "Freedom",
                            "Happiness.Score")


    if scaling == 1:
        plotDataHistogram(inputsGDP, "GDP")
        plotDataHistogram(normalizationFunction(inputsGDP), "GDP (SCALED)")

    np.random.seed(5)
    indexes = [i for i in range(len(inputsGDP))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputsGDP)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = {}
    trainInputs['GDP'] = [inputsGDP[i] for i in trainSample]
    trainInputs['Freedom'] = [inputsFreedom[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = {}
    validationInputs['GDP'] = [inputsGDP[i] for i in validationSample]
    validationInputs['Freedom'] = [inputsFreedom[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    if scaling == 1:
        minimGDP, maximGDP = getMinMaxParameters(trainInputs['GDP'])
        minimO, maximO = getMinMaxParameters(trainOutputs)

        meanGDP, devGDP = getStatisticalParameters(trainInputs['GDP'])
        meanO, devO = getStatisticalParameters(trainOutputs)

        trainInputs['GDP'] = normalizationFunction(trainInputs['GDP'])
        trainOutputs = normalizationFunction(trainOutputs)

        if normalizationFunction == minMaxScaling:
            validationInputs['GDP'] = minMaxScalingParam(validationInputs['GDP'], minimGDP, maximGDP)
            validationOutputs = minMaxScalingParam(validationOutputs, minimO, maximO)
        else:
            validationInputs['GDP'] = statisticalScalingParam(validationInputs['GDP'], meanGDP, devGDP)
            validationOutputs = statisticalScalingParam(validationOutputs, meanO, devO)

    # form the training inputs
    features = []
    for i in range(len(trainInputs['GDP'])):
        features.append([trainInputs['GDP'][i]])


    reg = BGDRegression()
    print("WEIGHTS : ", reg.fit(features, trainOutputs, 0.01, 1000, 10))

    # form the validation inputs
    features = []
    for i in range(len(validationInputs['GDP'])):
        features.append([validationInputs['GDP'][i]])
    v = reg.predict(features)
    e1GDP, e2GDP = evalSingleTargetRegression(validationOutputs, v)
    print("MAE : ", e1GDP)
    print("RMSE : ", e2GDP)