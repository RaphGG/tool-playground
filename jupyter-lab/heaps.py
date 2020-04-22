from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sys


class Heap():
  '''
  Object Class to represent a Min-Heap Data Structure.
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
  def __init__(self, order):
    self.order = order
    self.adjlist = defaultdict(list)

  def addEdge(self, v1, v2, weight):
    #vertex = {'index': v2, 'weight': weight}
    self.adjlist[v1].append((v2, weight))
    self.adjlist[v2].append((v1, weight))

  def addEdgeDi(self, v1, v2, weight):
    self.adjlist[v1].append((v2, weight))

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
  
  def toNX(self):
    g = nx.Graph()
    for node, edgelist in self.adjlist.items():
      for edge in edgelist:
        g.add_edge(node, edge[0], weight=edge[1])

    return g
  
  def toNXDi(self):
    g = nx.DiGraph()
    for node, edgelist in self.adjlist.items():
      for edge in edgelist:
        g.add_edge(node, edge[0], weight=edge[1])

    return g

class GraphDi():
  def __init__(self, order):
    self.order = order
    self.adjlist = defaultdict(list)

  def addEdge(self, v1, v2, weight):
    self.adjlist[v1].append((v2, weight))
  
  def addNode(self, v):
    self.adjlist[v] = []

  def dijkstra(self, start):
    vertices = []
    plotGraph = self.toNX()
    try:
      pos = nx.planar_layout(plotGraph)
    except nx.NetworkXException:
      pos = nx.spring_layout(plotGraph)

    edge_labels = nx.get_edge_attributes(plotGraph, 'weight')
    options = {'node_size':700,
    'cmap':plt.cm.Blues,
    'node_color':plotGraph.nodes()}

    edge_colors = ['black'] * len(plotGraph.edges())

    dist = [float('inf')] * self.order
    visited = [False] * self.order
    numVisited = 0

    dist[start] = 0

    heap = Heap()
    heap.insert(Vertex(start, 0, [start]))

    while not heap.isEmpty() and numVisited < self.order:
      vertex = heap.deleteMin()
      vertices.append(vertex)
      if visited[vertex.ID]:
        continue

      #print("Vertex ID:", vertex.ID)
      #print("Vertex Dist:", vertex.dist)
      #print("dist array:", dist)
      #print("visited array:", visited)
      visited[vertex.ID] = True
      numVisited += 1

      for edge in self.adjlist[vertex.ID]:
        if edge[1] + vertex.dist < dist[edge[0]]:
          index = list(plotGraph.edges()).index((vertex.ID, edge[0]))
          edge_colors[index] = 'red'
          nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
          nx.draw(plotGraph, pos, with_labels=True, edge_color=edge_colors, **options)
          edge_colors[index] = 'black'
          dist[edge[0]] = edge[1] + vertex.dist
          newPath = vertex.path.copy()
          newPath.append(edge[0])
          newVertex = Vertex(edge[0], dist[edge[0]], newPath)
          heap.insert(newVertex)

    for vertex in vertices:
      for i in range(len(vertex.path[:-1])):
        index = list(plotGraph.edges()).index((vertex.path[i], vertex.path[i+1]))
        edge_colors[index] = 'blue'

      nx.draw_networkx_edge_labels(plotGraph, pos, edge_labels=edge_labels)
      nx.draw(plotGraph, pos, with_labels=True, edge_color=edge_colors, **options)
      print(vertex)

  def toNX(self):
    g = nx.DiGraph()
    print(self.adjlist.items)
    for node, edgelist in self.adjlist.items():
      g.add_node(node)
      for edge in edgelist:
        g.add_edge(node, edge[0], weight=edge[1])

    return g

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

