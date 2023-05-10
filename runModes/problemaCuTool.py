from math import sqrt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from regression.batch_gradient_descent import BGDRegression
from runModes.multiVar import evalSingleTargetRegression
from utils.utils import loadData

def multiVarTool(path):
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

    # form the training inputs
    # standardize the features
    scaler = StandardScaler()
    features = []
    for i in range(len(trainInputs['GDP'])):
        features.append([trainInputs['GDP'][i], trainInputs['Freedom'][i]])
    features_Normalized = scaler.fit_transform(features)
    nt = [[o] for o in trainOutputs]
    op_Normalized = scaler.fit_transform(nt)



    reg = SGDRegressor(loss='squared_error', penalty=None, eta0=0.01, max_iter=1000, tol=1e-3, random_state=42)
    reg.fit(features_Normalized, op_Normalized)
    weights = (reg.coef_, reg.intercept_)
    print("WEIGHTS : ", weights)

    # form the validation inputs
    features = []
    for i in range(len(validationInputs['GDP'])):
        features.append([validationInputs['GDP'][i], validationInputs['Freedom'][i]])
    validations = scaler.fit_transform(features)
    v = reg.predict(validations)
    vo = scaler.fit_transform([[o] for o in validationOutputs])
    e1GDP, e2GDP = evalSingleTargetRegression([o[0] for o in vo], v)
    print("MAE : ", e1GDP)
    print("RMSE : ", e2GDP)