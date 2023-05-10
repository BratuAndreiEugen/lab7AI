from normalization.normal import minMaxScaling, statisticalNormalisation
from runModes.multiOutput import multiOutput
from runModes.multiVar import multiVar
from runModes.problemaCuTool import multiVarTool
from runModes.uniVar import uniVar

#multiVarTool("C:\Proiecte SSD\Python\lab7AI\\2017.csv")

#No Normalization
#multiVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv")
#uniVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv")

#Min Max Scaling
#multiVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv", 1, minMaxScaling)
#uniVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv", 1, minMaxScaling)

#Statistical Normalisation
#multiVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv", 1, statisticalNormalisation)
#uniVar("C:\Proiecte SSD\Python\lab7AI\\2017.csv", 1, statisticalNormalisation)

# multi output independent
# to do: individual single output
multiOutput("C:\Proiecte SSD\Python\lab7AI\\2017.csv")