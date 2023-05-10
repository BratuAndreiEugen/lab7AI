import csv

from graphics.graphic_utils import plotDataHistogram


def loadData(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = []
    for i in range(0, len(data)):
        if data[i][selectedVariable] == '':
            inputs.append(0.0)
        else:
            inputs.append(float(data[i][selectedVariable]))

    selectedOutput = dataNames.index(outputVariabName)
    outputs = []
    for i in range(0, len(data)):
        if data[i][selectedVariable] == '':
            outputs.append(0.0)
        else:
            outputs.append(float(data[i][selectedOutput]))


    return inputs, outputs

# a, b = loadData("C:\Proiecte SSD\Python\lab6AI\data\\v1_world-happiness-report-2017.csv", "Economy..GDP.per.Capita.", "Happiness.Score")
# print(a)
# print(b)
# plotDataHistogram(a, "GDP")
# plotDataHistogram(b, "Happiness")