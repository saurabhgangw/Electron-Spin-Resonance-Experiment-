import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#loading csv into pandas dataframe
df = pd.read_csv("data.csv") 

#defining axis
x =  df(["Time"]).tonumpy() 
y_1 = df(["Channel A"]).tonumpy() 
y_2 = df(["Channel A"]).tonumpy()

#plotting 

fig, ax = plt.subplots()
ax.plot(x, y_1)
ax.plot(x, y_2)
plt.show()