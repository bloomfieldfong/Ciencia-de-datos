import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##file location
file = "C:/Users/bloom/Desktop/Ciencia-de-datos/Proyecto/proyecto_training_data.npy"


##Importamos la data
data = np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
data = pd.DataFrame(data)

##Separacion de la data en 80% y 20%    
training_data = data.loc[0:1168]
test_data = data.loc[1167:1460]


##Headers de los datos
##SalesPrice - OverallQual - 1stFlrSF (First Floor square feet) - TotRmsAbvGrd - YearBuilt - LotFrontage

correlation_table = data.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_table, annot=True)
plt.show()
