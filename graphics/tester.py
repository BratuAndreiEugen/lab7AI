from utils.utils import loadData

path = "C:\Proiecte SSD\Python\lab6AI\data\\v1_world-happiness-report-2017.csv"

gdp1, out1 = loadData(path,"Economy..GDP.per.Capita.", "Happiness.Score")
freedom1, out1 = loadData(path, "Freedom", "Happiness.Score")

path = "C:\Proiecte SSD\Python\lab6AI\data\\v2_world-happiness-report-2017.csv"

gdp2, out2 = loadData(path,"Economy..GDP.per.Capita.", "Happiness.Score")
freedom2, out2 = loadData(path, "Freedom", "Happiness.Score")

print("GDP")
for i in range(0, len(out2)):
    if gdp1[i] != gdp2[i]:
        print(i, gdp1[i], gdp2[i])

print("FREEDOM")
for i in range(0, len(out2)):
    if freedom2[i] != freedom1[i]:
        print(i, freedom1[i], freedom2[i], gdp2[i])

print("OUT")
for i in range(0, len(out2)):
    if out1[i] != out2[i]:
        print(i, out1[i], out2[i])
