
import numpy as np

class BGDRegression:
    def __init__(self):
        self.__w = None
        self.__inter = 0
        pass

    def fit(self, features, output, learning_rate, iterations, batch_size):
        num_features = len(features[0])
        weights = np.zeros(num_features)
        intercept = 0.0
        print("feat : ", features)
        print("op : ", output)
        # set number of batches
        batches = len(features) // batch_size

        for i in range(iterations):
            # for each batch,
            for j in range(batches):
                batch_features = features[j * batch_size: (j + 1) * batch_size]
                batch_output = output[j * batch_size: (j + 1) * batch_size]
                pred = np.dot(batch_features, weights) + intercept

                # compute the gradients of the loss for weights and intercept
                batch_features = np.array(batch_features)
                error = pred - batch_output
                dw = (1 / batch_size) * np.dot(batch_features.T, error)
                di = (1 / batch_size) * np.sum(error)

                # intercept and weights update
                weights = weights - learning_rate * dw
                intercept = intercept - learning_rate * di


        self.__w = weights
        self.__inter = intercept
        return weights, intercept

    def predict(self, inputs):
        return np.dot(inputs, self.__w) + self.__inter
