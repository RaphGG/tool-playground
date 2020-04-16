from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import networkx as nx

g = nx.read_multiline_adjlist('graph-list.txt')
'''
g = nx.petersen_graph()
nx.write_multiline_adjlist(g, 'graph-list-3.txt')
g.add_edge(1, 2)
g.add_edge(1, 3)
'''
#g[1][2]['color'] = "red"
nx.draw(g, with_Labels=True, font_weight='bold')
plt.show()
