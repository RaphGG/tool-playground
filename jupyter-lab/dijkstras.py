# Greedy Algorithms - Dijkstra's Algorithm
# A python program for use with both directed and undirected weighted graphs
# with an adjacency list representations.
# Utilizes custom minheap data structures to hold vertex & edge data and
# achieve an run time complexity of: 
# with a space complexity of: 

from collections import defaultdict
import sys

class Heap():
  def __init__(self):
    self.array = []
    self.size = 0
    self.pos = []

  def newNode(self, vertex, weight):
    node = [vertex, weight]
    return node

  def swapNodes(self, idx1, idx2):
    temp = self.array[idx1]
    self.array[idx1] = self.array[idx2]
    self.array[idx2] = temp


  def heapify(self, idx):
    sIdx = idx
    left = 2*idx + 1
    right = 2*idx + 2

    if left < self.size and self.array[left][1] < self.array[sIdx][1]:
      sIdx = left

    if right < self.size and self.array[right][1] < self.array[sIdx][1]:
      sIdx = right

    if sIdx != idx:
      self.pos[ self.array[sIdx][0] ] = idx
      self.pos[ self.array[idx][0] ] = sIdx

      self.swapNodes(sIdx, idx)
      self.heapify(sIdx)

  def minNode(self):
    if self.isEmpty() == True:
      return

    root = self.array[0]

    lastNode = self.array[self.size - 1]
    self.array[0] = lastNode
    self.pos[lastNode[0]] = 0
    self.pos[root[0]] = self.size - 1

    self.size -= 1
    self.heapify(0)
    return root

  def isEmpty(self):
    return True if self.size == 0 else False
  
  def decreaseKey(self, vertex, weight):
    idx = self.pos[vertex]

    self.array[idx][1] = vertex

    while idx > 0 and self.array[idx][1] < self.array[(idx - 1) / 2][1]:
      self.pos[self.array[idx][0]] = (idx-1) / 2
      self.pos[self.array[(idx - 1) / 2][0]] = idx
      self.swapNodes(idx, (idx-1)/2)

      idx = (idx-1) / 2

  def contains(self, vertex):
    if self.pos[vertex] < self.size:
      return True
    return False

def printArr(weight, n):
  print("Vertex Distance from source")
  for i in range(n):
    print("%f\t\t%f" % (i, weight[i]))

class Graph():

  def __init__(self, order):
    self.order = order
    self.graph = defaultdict(list)

  def addEdge(self, v1, v2, weight):

    node = [v2, weight]
    self.graph[v1].insert(0, node)

    node = [v1, weight]
    self.graph[v2].insert(0, node)
    #print(self.graph)

  def dijkstra(self, start):

    numVertices = self.order
    dist = []


    minHeap = Heap()

    for vertex in range(numVertices):
      dist.append(float('inf'))
      minHeap.array.append( minHeap.newNode(vertex, dist[vertex]) )
      minHeap.pos.append(vertex)

    minHeap.pos[start] = start
    dist[start] = 0
    minHeap.decreaseKey(start, dist[start])

    minHeap.size = numVertices

    while minHeap.isEmpty == False:
      newNode = minHeap.minNode()
      newVertex = newNode[0]

      for pCrawl in self.graph[newVertex]:
        v = pCrawl[0]

        if minHeap.contains(v) and dist[newVertex] != float('inf') and \
          pCrawl[1] + dist[newVertex] < dist[v]:
            dist[v] = pCrawl[1] + dist[newVertex]
            minHeap.decreaseKey(v, dist[v])

    printArr(dist, numVertices)

graph = Graph(5)
graph.addEdge(0, 1, 4)
graph.addEdge(0, 4, 1)
graph.addEdge(1, 2, 16)
graph.addEdge(2, 3, 12)
graph.addEdge(3, 4, 2)

graph.dijkstra(0)