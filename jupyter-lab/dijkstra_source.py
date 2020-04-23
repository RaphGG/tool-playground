from collections import defaultdict
from IPython.display import display
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sys

class Heap():
  '''
  Object Class to represent a Min-Heap Data Structure.
  Supports two basic functions: 

  deleteMin() - Best Case: O(1) / Worse Case: O(log n)
  insert() - Best Case: O(1) / Worst Case: O(log n)

  This simple Min Heap is to be used in conjunction Dijkstra's Algorithm as a priority queue
  to support O((|E| + |V|)log |V|) behavior.
  '''
  def __init__(self):
    self.array = []

  def isEmpty(self):
    return True if len(self.array) <= 0 else False

  def swap(self, i, j):
    temp = self.array[i]
    self.array[i] = self.array[j]
    self.array[j] = temp

  def getParent(self, child):
    return 0 if child == 0 else ( child - 1 ) // 2

  def percolateUp(self, child):
    parent = self.getParent(child)

    while self.array[child] < self.array[parent]:
      self.swap(child, parent)
      child = parent
      parent = self.getParent(child)

  def insert(self, data):
    self.array.append(data)
    self.percolateUp(len(self.array)-1)

  def smallestChild(self, parent):
    length = len(self.array)
    leftchild = (2 * parent + 1) if (2 * parent + 1) < length else parent
    rightchild = (2 * parent + 2) if (2 * parent + 2) < length else parent

    retbool = self.array[leftchild] < self.array[rightchild]
    return leftchild if retbool else rightchild

  def percolateDown(self, parent):
    child = self.smallestChild(parent)

    while (self.array[parent] > self.array[child]):
      self.swap(parent, child)
      parent = child
      child = self.smallestChild(parent)

  def deleteMin(self):
    if self.isEmpty():
      print("This heap is empty, no root to delete.")
      return

    elif len(self.array) - 1 == 0:
      return self.array.pop()
    
    retval = self.array[0]
    self.array[0] = self.array[len(self.array) - 1]
    self.array.pop()
    self.percolateDown(0)
    
    return retval

  def heapify(self):
    start = self.getParent(len(self.array) - 1)

    for index in range(start, -1, -1):
      self.percolateDown(index)

class Vertex():
  '''
  Object Class to represent Vertices in a Graph.
  These vertices contain 3 primary fields:

  ID: int - An number that identifies the vertex
  dist: int - The distance to this vertex relative to a source vertex at
              any given step within Dijkstra's Algorithm.

  path: list - The shortest cost path to reach this vertex relative to a
               source vertex at any given step within
               Dijkstra's Algorithm.
  '''
  def __init__(self, ID, dist=0, path=[]):
    self.ID = ID
    self.dist = dist
    self.path = path

  def __lt__(self, rhs):
    return self.dist < rhs.dist
  
  def __gt__(self, rhs):
    return self.dist > rhs.dist

  def __str__(self):
    pathString = ""
    for i, v in enumerate(self.path):
      if i == len(self.path) - 1:
        pathString += f"{v}"
      else:
        pathString += f"{v} -> "

    return f"Shortest path to vertex {self.ID} is {pathString} with a total cost of {self.dist}"

class Graph():
  '''
  Object Class to represent Weighted Non-Directional Graphs.
  The graphs are stored and represented by an adjacency list format.

  These graphs contain 2 primary fields:
  order: int - The number of vertices in this graph
  adjlist: defaultdict(list) - The adjacency list
  '''
  def __init__(self, order=0):
    self.order = order
    self.adjlist = defaultdict(list)
    for vertex in range(order):
      self.adjlist[vertex] = []

  def addEdge(self, v1, v2, weight):
    vertices = self.adjlist.keys()
    if not v1 in vertices:
      self.order += 1
    if not v2 in vertices:
      self.order += 1
    self.adjlist[v1].append((v2, weight))
    self.adjlist[v2].append((v1, weight))

  def addNode(self, v):
    if not v in self.adjlist.keys():
      self.adjlist[v] = []
      self.order += 1

  @staticmethod
  def readadjlist(filename):
    g = Graph()
    with open(filename) as file:
      for line in file:
        tokens = line.split()
        if len(tokens) > 1:
          g.addEdge(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        elif len(tokens) == 1:
          g.addNode(int(tokens[0]))
    return g

  def dijkstra(self, start):
    dist = [float('inf')] * self.order
    visited = [False] * self.order
    numVisited = 0

    dist[start] = 0

    heap = Heap()
    heap.insert(Vertex(start, 0, [start]))

    while not heap.isEmpty() and numVisited < self.order:
      vertex = heap.deleteMin()
      if visited[vertex.ID]:
        continue

      print(vertex)
      #print("Vertex ID:", vertex.ID)
      #print("Vertex Dist:", vertex.dist)
      #print("dist array:", dist)
      #print("visited array:", visited)
      visited[vertex.ID] = True
      numVisited += 1

      for edge in self.adjlist[vertex.ID]:
        if edge[1] + vertex.dist < dist[edge[0]]:
          dist[edge[0]] = edge[1] + vertex.dist
          newPath = vertex.path.copy()
          newPath.append(edge[0])
          newVertex = Vertex(edge[0], dist[edge[0]], newPath)
          heap.insert(newVertex)

  def printDijkstra(self, start):
    plotGraph = self.toNX()
    try:
      pos = nx.planar_layout(plotGraph)
    except nx.NetworkXException:
      pos = nx.spring_layout(plotGraph)

    edge_labels = nx.get_edge_attributes(plotGraph, 'weight')
    options = {'node_size':700}
    node_colors = ['green'] * len(plotGraph.nodes())
    edge_colors = ['black'] * len(plotGraph.edges())

    dist = [float('inf')] * self.order
    visited = [False] * self.order
    numVisited = 0

    dist[start] = 0

    heap = Heap()
    heap.insert(Vertex(start, 0, [start]))
    vertices = []

    print(f"Begin Dijkstra's Algorithm over the following graph with {self.order} vertices at source vertex: {start}")
    nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
    nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, **options)
    plt.show()
    print("\nInitial table of vertices and their distances from the source: ")
    df = pd.DataFrame({'vertex':range(self.order), 'distance/cost':dist})
    display(df.style.hide_index())
    while not heap.isEmpty() and numVisited < self.order:
      vertex = heap.deleteMin()
      if visited[vertex.ID]:
        continue

      vertices.append(vertex)
      print('{:-^50}'.format('-'))
      print(f"\nVisiting vertex: {vertex.ID}")
      visited[vertex.ID] = True
      numVisited += 1
      updateCheck = False
      node_index = list(plotGraph.nodes()).index(vertex.ID)
      node_colors[node_index] = 'yellow'

      if len(vertex.path) > 1:
        for i in range(len(vertex.path[:-1])):
          index = list(plotGraph.edges()).index((vertex.path[i], vertex.path[i+1]))
          edge_colors[index] = 'blue'

      for edge in self.adjlist[vertex.ID]:
        if edge[1] + vertex.dist < dist[edge[0]]:
          updateCheck = True
          index = list(plotGraph.edges()).index((vertex.ID, edge[0]))
          edge_colors[index] = 'red'
          nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
          nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, **options)
          plt.show()
          print("Smaller distance found, table updated!")
          edge_colors[index] = 'black'

          dist[edge[0]] = edge[1] + vertex.dist
          newPath = vertex.path.copy()
          newPath.append(edge[0])
          newVertex = Vertex(edge[0], dist[edge[0]], newPath)
          heap.insert(newVertex)

          df.update({'distance/cost':dist})
          display(df.style.apply(colorCell, row=edge[0], axis=None).hide_index())
          print("\n")
      if not updateCheck:
        print("No smaller distances found.")
      node_colors[node_index] = 'green'

    print('{:-^50}'.format('-'))
    print(f"\nShortest paths to each vertex from vertex: {start}")
    edge_colors = ['black'] * len(plotGraph.edges())
    for vertex in vertices:
      for i in range(len(vertex.path[:-1])):
        index = list(plotGraph.edges()).index((vertex.path[i], vertex.path[i+1]))
        edge_colors[index] = 'blue'
      index = list(plotGraph.nodes()).index(vertex.ID)
      node_colors[index] = 'yellow'
      nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
      nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, **options)
      plt.show()
      node_colors[index] = 'green'
      edge_colors = ['black'] * len(plotGraph.edges())
      print(vertex)
      print('{:-^50}'.format('-'))
      print("\n")

  def toNX(self):
    g = nx.Graph()
    for node, edgelist in self.adjlist.items():
      g.add_node(node)
      for edge in edgelist:
        g.add_edge(node, edge[0], weight=edge[1])

    return g

class GraphDi():
  '''
  Object Class to represent Weighted Directional Graphs.
  The graphs are stored and represented by an adjacency list format.

  These graphs contain 2 primary fields:
  order: int - The number of vertices in this graph
  adjlist: defaultdict(list) - The adjacency list
  '''
  def __init__(self, order=0):
    self.order = order
    self.adjlist = defaultdict(list)
    for vertex in range(order):
      self.adjlist[vertex] = []

  def addEdge(self, v1, v2, weight):
    vertices = self.adjlist.keys()
    if not v1 in vertices:
      self.order += 1
    if not v2 in vertices:
      self.order += 1
      self.adjlist[v2] = []

    self.adjlist[v1].append((v2, weight))

  def addNode(self, v):
    if not v in self.adjlist.keys():
      self.order += 1
      self.adjlist[v] = []

  @staticmethod
  def readadjlist(filename):
    g = GraphDi()
    with open(filename) as file:
      for line in file:
        tokens = line.split()
        if len(tokens) > 1:
          g.addEdge(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        elif len(tokens) == 1:
          g.addNode(int(tokens[0]))
    return g

  def dijkstra(self, start):
    dist = [float('inf')] * self.order
    visited = [False] * self.order
    numVisited = 0

    dist[start] = 0

    heap = Heap()
    heap.insert(Vertex(start, 0, [start]))

    while not heap.isEmpty() and numVisited < self.order:
      vertex = heap.deleteMin()
      if visited[vertex.ID]:
        continue

      print(vertex)
      #print("Vertex ID:", vertex.ID)
      #print("Vertex Dist:", vertex.dist)
      #print("dist array:", dist)
      #print("visited array:", visited)
      visited[vertex.ID] = True
      numVisited += 1

      for edge in self.adjlist[vertex.ID]:
        if edge[1] + vertex.dist < dist[edge[0]]:
          dist[edge[0]] = edge[1] + vertex.dist
          newPath = vertex.path.copy()
          newPath.append(edge[0])
          newVertex = Vertex(edge[0], dist[edge[0]], newPath)
          heap.insert(newVertex)

  def printDijkstra(self, start):
    plotGraph = self.toNX()
    try:
      pos = nx.planar_layout(plotGraph)
    except nx.NetworkXException:
      pos = nx.spring_layout(plotGraph)

    edge_labels = nx.get_edge_attributes(plotGraph, 'weight')
    options = {'node_size':700}
    node_colors = ['green'] * len(plotGraph.nodes())
    edge_colors = ['black'] * len(plotGraph.edges())

    dist = [float('inf')] * self.order
    visited = [False] * self.order
    numVisited = 0

    dist[start] = 0

    heap = Heap()
    heap.insert(Vertex(start, 0, [start]))
    vertices = []

    print(f"Begin Dijkstra's Algorithm over the following graph with {self.order} vertices at source vertex: {start}")
    nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
    nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, **options)
    plt.show()
    print("\nInitial table of vertices and their distances from the source: ")
    df = pd.DataFrame({'vertex':range(self.order), 'distance/cost':dist})
    display(df.style.hide_index())
    while not heap.isEmpty() and numVisited < self.order:
      vertex = heap.deleteMin()
      if visited[vertex.ID]:
        continue

      vertices.append(vertex)
      print('{:-^50}'.format('-'))
      print(f"\nVisiting vertex: {vertex.ID}")
      visited[vertex.ID] = True
      numVisited += 1
      updateCheck = False
      node_index = list(plotGraph.nodes()).index(vertex.ID)
      node_colors[node_index] = 'yellow'

      if len(vertex.path) > 1:
        for i in range(len(vertex.path[:-1])):
          index = list(plotGraph.edges()).index((vertex.path[i], vertex.path[i+1]))
          edge_colors[index] = 'blue'

      for edge in self.adjlist[vertex.ID]:
        if edge[1] + vertex.dist < dist[edge[0]]:
          updateCheck = True
          index = list(plotGraph.edges()).index((vertex.ID, edge[0]))
          edge_colors[index] = 'red'
          nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
          nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, **options)
          plt.show()
          print("Smaller distance found, table updated!")
          edge_colors[index] = 'black'

          dist[edge[0]] = edge[1] + vertex.dist
          newPath = vertex.path.copy()
          newPath.append(edge[0])
          newVertex = Vertex(edge[0], dist[edge[0]], newPath)
          heap.insert(newVertex)

          df.update({'distance/cost':dist})
          display(df.style.apply(colorCell, row=edge[0], axis=None).hide_index())
          print("\n")
      if not updateCheck:
        print("No smaller distances found.")
      node_colors[node_index] = 'green'

    print('{:-^50}'.format('-'))
    print(f"\nShortest paths to each vertex from vertex: {start}")
    edge_colors = ['black'] * len(plotGraph.edges())
    for vertex in vertices:
      for i in range(len(vertex.path[:-1])):
        index = list(plotGraph.edges()).index((vertex.path[i], vertex.path[i+1]))
        edge_colors[index] = 'blue'
      index = list(plotGraph.nodes()).index(vertex.ID)
      node_colors[index] = 'yellow'
      nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
      nx.draw(plotGraph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, **options)
      plt.show()
      node_colors[index] = 'green'
      edge_colors = ['black'] * len(plotGraph.edges())
      print(vertex)
      print('{:-^50}'.format('-'))
      print("\n")

  def toNX(self):
    g = nx.DiGraph()
    print(self.adjlist.items)
    for node, edgelist in self.adjlist.items():
      g.add_node(node)
      for edge in edgelist:
        g.add_edge(node, edge[0], weight=edge[1])

    return g

def colorCell(x, row=0):
  df = pd.DataFrame('', x.index, x.columns)
  df.iloc[row, 1] = 'color: red'
  return df

'''
testGraph = Graph()
testGraph.addEdge(0, 1, 1)
testGraph.addEdge(0, 2, 3)
testGraph.addEdge(1, 2, 1)
testGraph.addEdge(1, 3, 5)
testGraph.addNode(4)
print(testGraph.order)

testGraphDi = GraphDi()
testGraphDi.addEdge(0, 1, 1)
testGraphDi.addEdge(0, 2, 3)
testGraphDi.addEdge(1, 2, 1)
testGraphDi.addEdge(1, 3, 5)
testGraphDi.addNode(1)
print(testGraphDi.order)
'''
'''
exampleGraph = GraphDi(4)
exampleGraph.addEdge(0, 1, 1)
exampleGraph.addEdge(0, 2, 3)
exampleGraph.addEdge(1, 2, 1)
exampleGraph.addEdge(1, 3, 5)
exampleGraph.addEdge(2, 3, 3)
exampleGraph.addEdge(3, 0, 6)
exampleGraph.addEdge(3, 1, 5)
exampleGraph.addNode(4)
g = exampleGraph.toNX()
print(g.nodes)
print(exampleGraph.adjlist)
'''
'''
vertex = Vertex(0, 0, [0])
t = (1, 0)
vertex.path.append(t[0])

print(vertex.path)


g = Graph(3)
g.addEdgeDirectional(0, 1, 2)
g.addEdgeDirectional(1, 2, 4)
g.addEdgeDirectional(0, 2, 7)

g.dijkstra(0)
'''
'''
print(g.adjlist)
for edge in g.adjlist[0]:
  print(edge)
  print(edge[0])
  print(edge[1])

e1 = Vertex(1, 20)
e2 = Vertex(2, 19)

heap = Heap()

heap.insert(e1)
heap.insert(e2)
#
heap = Heap()
print("Is this heap empty?", heap.isEmpty())
heap.deleteMin()

heap.insert(9)
heap.insert(11)
heap.insert(181)
heap.insert(5)
heap.insert(8)
heap.insert(1)

print(heap.array)
print(heap.size)

for i in range(6):
  print(heap.deleteMin())

print(heap.size)
print("Is this heap empty?", heap.isEmpty())
'''

