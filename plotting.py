import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv") # reading the dataset into a pandas dataframe 
x = 'Time'
y_1 = 'Channel A'
y_2 = 'Channel B'
fig, ax = plt.subplots()
#df.plot(x='Time', y='Channel A', kind='line')
#df.plot(x='Time', y='Channel B', kind='line')

ax.plot(x, y_1)
ax.plot(x, y_2)
plt.show()