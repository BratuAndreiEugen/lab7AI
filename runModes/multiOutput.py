from statistics import mean

import numpy as np

from graphics.graphic_utils import plot2Features
from normalization.normal import minMaxScaling, getMinMaxParameters, minMaxScalingParam
from regression.batch_gradient_descent import BGDRegression
from runModes.multiVar import evalSingleTargetRegression
from utils.utils import loadData


def multiOutput(path):
    print("\nMULTIVARIABLE MULTIOUTPUT COD PROPRIU / SCALING : " + "minMaxScaling" + "\n")
    inputsGDP, outputsHappiness = loadData(path,
                                  "Economy..GDP.per.Capita.", "Happiness.Score")
    inputsFreedom, outputsHealth = loadData(path, "Freedom",
                                      "Health..Life.Expectancy.")

    np.random.seed(5)
    indexes = [i for i in range(len(inputsGDP))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputsGDP)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = {}
    trainInputs['GDP'] = [inputsGDP[i] for i in trainSample]
    trainInputs['Freedom'] = [inputsFreedom[i] for i in trainSample]
    trainOutputsHappiness = [outputsHappiness[i] for i in trainSample]
    trainOutputsHealth = [outputsHealth[i] for i in trainSample]


    validationInputs = {}
    validationInputs['GDP'] = [inputsGDP[i] for i in validationSample]
    validationInputs['Freedom'] = [inputsFreedom[i] for i in validationSample]
    validationOutputsHappiness = [outputsHappiness[i] for i in validationSample]
    validationOutputsHealth = [outputsHealth[i] for i in validationSample]

    plot2Features(inputsGDP, inputsFreedom, minMaxScaling)
    plot2Features(outputsHappiness, outputsHealth, minMaxScaling, "Happiness", "Health")

    minimGDP, maximGDP = getMinMaxParameters(trainInputs['GDP'])
    minimF, maximF = getMinMaxParameters(trainInputs['Freedom'])
    minimO, maximO = getMinMaxParameters(trainOutputsHappiness)
    minimOp, maximOp = getMinMaxParameters(trainOutputsHealth)

    trainInputs['GDP'] = minMaxScaling(trainInputs['GDP'])
    trainOutputsHappiness = minMaxScaling(trainOutputsHappiness)
    trainOutputsHealth = minMaxScaling(trainOutputsHealth)
    trainInputs['Freedom'] = minMaxScaling(trainInputs['Freedom'])

    validationInputs['GDP'] = minMaxScalingParam(validationInputs['GDP'], minimGDP, maximGDP)
    validationOutputsHappiness = minMaxScalingParam(validationOutputsHappiness, minimO, maximO)
    validationOutputsHealth = minMaxScalingParam(validationOutputsHealth, minimOp, maximOp)
    validationInputs['Freedom'] = minMaxScalingParam(validationInputs['Freedom'], minimF, maximF)

    # form the training inputs
    features = []
    for i in range(len(trainInputs['GDP'])):
        features.append([trainInputs['GDP'][i], trainInputs['Freedom'][i]])

    reg = BGDRegression()
    print("\nHAPPINESS\n")
    print("WEIGHTS (HAPPINESS) : ", reg.fit(features, trainOutputsHappiness, 0.01, 1000, 10))

    # form the validation inputs
    features = []
    for i in range(len(validationInputs['GDP'])):
        features.append([validationInputs['GDP'][i], validationInputs['Freedom'][i]])
    v = reg.predict(features)
    e1GDP1, e2GDP1 = evalSingleTargetRegression(validationOutputsHappiness, v)
    print("MAE : ", e1GDP1)
    print("RMSE : ", e2GDP1)

    reg = BGDRegression()
    print("\nHEALTH\n")
    print("WEIGHTS (HEALTH) : ", reg.fit(features, trainOutputsHealth, 0.01, 1000, 10))

    # form the validation inputs
    features = []
    for i in range(len(validationInputs['GDP'])):
        features.append([validationInputs['GDP'][i], validationInputs['Freedom'][i]])
    v = reg.predict(features)
    e1GDP2, e2GDP2 = evalSingleTargetRegression(validationOutputsHealth, v)
    print("MAE : ", e1GDP2)
    print("RMSE : ", e2GDP2)

    print("ERRORS FOR 2 OUTPUTS (MEAN)")
    print("MAE : ", mean([e1GDP1, e1GDP2]))
    print("RMSE : ", mean([e2GDP1, e2GDP2]))
