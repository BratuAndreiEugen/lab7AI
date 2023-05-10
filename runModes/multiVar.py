from math import sqrt

import numpy as np

from graphics.graphic_utils import plot2Features
from normalization.normal import minMaxScaling, getMinMaxParameters, minMaxScalingParam, getStatisticalParameters, \
    statisticalScalingParam
from regression.batch_gradient_descent import BGDRegression
from utils.utils import loadData

def evalSingleTargetRegression(actual, predicted):
    print("Given validation data : ", actual)
    print("Predicted data : ", predicted)
    errorL1 = sum(abs(r - c) for r, c in zip(actual, predicted)) / len(actual)
    errorL2 = sqrt(sum((r - c) ** 2 for r, c in zip(actual, predicted)) / len(actual))
    return errorL1, errorL2


def multiVar(path, scaling = 0, normalizationFunction = None):

    print("\nMULTIVARIABLE COD PROPRIU / SCALING : " + str(scaling) + "\n")
    inputsGDP, outputs = loadData(path,
                        "Economy..GDP.per.Capita.", "Happiness.Score")
    inputsFreedom, outputs = loadData(path, "Freedom",
                            "Happiness.Score")

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
        plot2Features(inputsGDP, inputsFreedom, normalizationFunction)

    if scaling == 1:
        minimGDP, maximGDP = getMinMaxParameters(trainInputs['GDP'])
        minimF, maximF = getMinMaxParameters(trainInputs['Freedom'])
        minimO, maximO = getMinMaxParameters(trainOutputs)

        meanGDP, devGDP = getStatisticalParameters(trainInputs['GDP'])
        meanF, devF = getStatisticalParameters(trainInputs['Freedom'])
        meanO, devO = getStatisticalParameters(trainOutputs)

        trainInputs['GDP'] = normalizationFunction(trainInputs['GDP'])
        trainOutputs = normalizationFunction(trainOutputs)
        trainInputs['Freedom'] = normalizationFunction(trainInputs['Freedom'])
        if normalizationFunction == minMaxScaling:

            validationInputs['GDP'] = minMaxScalingParam(validationInputs['GDP'], minimGDP, maximGDP)
            validationOutputs = minMaxScalingParam(validationOutputs, minimO, maximO)
            validationInputs['Freedom'] = minMaxScalingParam(validationInputs['Freedom'], minimF, maximF)
        else:

            validationInputs['GDP'] = statisticalScalingParam(validationInputs['GDP'], meanGDP, devGDP)
            validationOutputs = statisticalScalingParam(validationOutputs, meanO, devO)
            validationInputs['Freedom'] = statisticalScalingParam(validationInputs['Freedom'], meanF, devF)

    # form the training inputs
    features = []
    for i in range(len(trainInputs['GDP'])):
        features.append([trainInputs['GDP'][i], trainInputs['Freedom'][i]])


    reg = BGDRegression()
    print("WEIGHTS : ", reg.fit(features, trainOutputs, 0.01, 1000, 10))

    # form the validation inputs
    features = []
    for i in range(len(validationInputs['GDP'])):
        features.append([validationInputs['GDP'][i], validationInputs['Freedom'][i]])
    v = reg.predict(features)
    e1GDP, e2GDP = evalSingleTargetRegression(validationOutputs, v)
    print("MAE : ", e1GDP)
    print("RMSE : ", e2GDP)