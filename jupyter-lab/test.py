'''
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import networkx as nx

nx.

%matplotlib inline

data = {
  'x': np.random.rand(5),
  'y': np.random.rand(5),
  'color': np.random.rand(5),
  'size': np.random.rand(5),
}
df = pd.DataFrame(data)
df.head()

style.use('seaborn-whitegrid')

plt.scatter('x', 'y', c='color', s='size', data=df, cmap=plt.cm.Blues)
plt.xlabel('x')
plt.ylabel('y')
'''
