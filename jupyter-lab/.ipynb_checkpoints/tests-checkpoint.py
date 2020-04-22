import plotly.graph_objects as go
#fig = go.FigureWidget(data=go.Bar(y=[2, 3, 1]))
f = go.FigureWidget()
f.show()


'''
#PLOTLY NETWORKX EXAMPLE
import plotly.graph_objects as go
import networkx as nx


G = nx.random_geometric_graph(200, 0.125)

edge_x = []
edge_y = []

for edge in G.edges():
  x0, y0 = G.nodes[edge[0]]['pos']
  x1, y1 = G.nodes[edge[1]]['pos']
  edge_x.append(x0)
  edge_x.append(x1)
  edge_x.append(None)
  edge_y.append(y0)
  edge_y.append(y1)
  edge_y.append(None)

edge_trace = go.Scatter(
  x=edge_x, y=edge_y,
  line=dict(width=0.5, color="#888"),
  hoverinfo='none', mode='lines'
)

node_x = []
node_y = []

for node in G.nodes():
  x, y = G.nodes[node]['pos']
  node_x.append(x)
  node_y.append(y)

node_trace = go.Scatter(
  x=node_x, y=node_y,
  mode='markers',
  hoverinfo='text',
  marker=dict(
    showscale=True,
    # colorscale options
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    colorscale='YlGnBu',
    reversescale=True,
    color=[],
    size=10,
    colorbar=dict(
      thickness=15,
      title='Node Connections',
      xanchor='left',
      titleside='right'
    ),
    line_width=2
  )
)

node_adjacencies = []

node_text = []
for node, adjacencies in enumerate(G.adjacency()):
  node_adjacencies.append(len(adjacencies[1]))
  node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
  layout=go.Layout(
    title='<br>Network graph made with Python',
    titlefont_size=16,
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    annotations=[ dict(
      text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
      showarrow=False,
      xref="paper", yref="paper",
      x=0.005, y=-0.002 ) ],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
fig.show()
'''
'''
# NETWORKX + PANDAS EXAMPLE
# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
 
# Build a dataframe with 4 connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
df
 
# Build your graph

G=nx.from_pandas_dataframe(df, 'from', 'to')
 
# Plot it
nx.draw(G, with_labels=True)
plt.show()
'''

'''
# TESTING STUFF
heap = [1, 8, 5, 181, 11, 9]

size = heap.__len__()

#print(size)
retval = heap[0]
size -= 1
heap[0] = heap[size]
#print(heap)
heap.pop()
#print(heap)
#self.percolateDown(0)

from collections import defaultdict

adjlist = defaultdict(list)

v1 = 1
v2 = 2
v3 = 3

edge1 = {'vertex': v2, 'weight': 56}
edge2 = {'vertex': v1, 'weight': 56}
edge3 = {'vertex': v3, 'weight': 22}
edge4 = {'vertex': v1, 'weight': 22}
edge5 = {'vertex': v3, 'weight': 31}
edge6 = {'vertex': v2, 'weight': 31}
adjlist[v1].append(edge1)
adjlist[v1].append(edge3)
adjlist[v2].append(edge2)
adjlist[v2].append(edge5)
adjlist[v3].append(edge4)
adjlist[v3].append(edge6)

print(adjlist[v1][0]['vertex'])
'''