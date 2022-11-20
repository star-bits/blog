# Graph

## Traversal

```python
graph = [
    [],
    [2, 3, 8], # 1
    [1, 7],    # 2
    [1, 4, 5], # 3
    [3, 5],    # 4
    [3, 4],    # 5
    [7],       # 6
    [2, 6, 8], # 7
    [1, 7]     # 8
]
```

### DFS: stack-recursion

```python
def dfs(graph, node, visited): 
    visited[node] = True
    print(node, end=' ')
    
    for i in graph[node]:
        if not visited[i]:
            dfs(graph, i, visited)

visited = [False] * len(graph)

dfs(graph, 1, visited)
```

```
1 2 7 6 8 3 4 5 
```

### BFS: queue

```python
from collections import deque

def bfs(graph, node, visited):
    q = deque([node])
    visited[node] = True
    
    while q:
        node = q.popleft()
        print(node, end=' ')
        
        for i in graph[node]:
            if not visited[i]:
                q.append(i)
                visited[i] = True
                
visited = [False] * len(graph)

bfs(graph, 1, visited)
```

```
1 2 3 8 7 4 5 6 
```

## Shortest path

### Dijkstra: 한 노드에서 다른 모든 노드까지의 최단거리

한 번 선택된 노드는 최단거리가 감소하지 않음

```python
nodes = 6
edges = 11
start = 1
graph = [
    [], 
    [(2, 2), (3, 5), (4, 1)], # 1 (node, cost)
    [(3, 3), (4, 2)],         # 2 
    [(2, 3), (6, 5)],         # 3
    [(3, 3), (5, 1)],         # 4
    [(3, 1), (6, 2)],         # 5
    []                        # 6
]
```

$O(N^2)$

```python
def dijkstra(start):
    distance[start] = 0
    visited[start] = True
    
    for i in graph[start]:
        distance[i[0]] = i[1]
    
    for _ in range(nodes-1): # 시작 노드를 제외한 나머지 nodes-1개의 노드에 대해 반복
        node = shortest_path_node() # 방문하지 않은 노드 중 distance가 가장 작은 노드 리턴
        visited[node] = True
        
        for i in graph[node]:
            cost = distance[node] + i[1] # 현재 노드를 거쳐서 갈 때의 cost가
            if cost<distance[i[0]]: # 기존 cost보다 작은 경우
                distance[i[0]] = cost

def shortest_path_node():
    min_ = int(1e9)
    node = 0
    
    for i in range(1, nodes+1):
        if distance[i]<min_ and not visited[i]:
            min_ = distance[i]
            node = i
    
    return node

visited = [False] * (nodes+1)
distance = [int(1e9)] * (nodes+1)

dijkstra(start)

for i in range(1, nodes+1):
    print(f'node {i}: cost {distance[i]}')
```

```
node 1: cost 0
node 2: cost 2
node 3: cost 3
node 4: cost 1
node 5: cost 2
node 6: cost 4
```

$O(E \log N)$

```python
import heapq # priority queue; min heap by default; (weight, value)

# heapq를 사용하면서 shortest_path_node() 부분이 필요없어짐
def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0
    
    while q:
        dist, node = heapq.heappop(q)
        
        if distance[node]<dist:
            continue
        
        for i in graph[node]:
            cost = dist + i[1] # 현재 노드를 거쳐서 갈 때의 cost가
            if cost<distance[i[0]]: # 기존 cost보다 작은 경우
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

visited = [False] * (nodes+1)
distance = [int(1e9)] * (nodes+1)

dijkstra(start)

for i in range(1, nodes+1):
    print(f'node {i}: cost {distance[i]}')
```

```
node 1: cost 0
node 2: cost 2
node 3: cost 3
node 4: cost 1
node 5: cost 2
node 6: cost 4
```

### Bellman-Ford: 음의 가중치를 가진 간선이 존재할 때

다익스트라와의 차이점은 매 반복마다 모든 간선을 확인한다는 것 (다익스트라는 방문하지 않은 노드 중 최단거리가 가장 가까운 노드만을 방문)

$O(N \cdot E)$

```python
nodes = int(input())
edges = int(input())
start = 1
graph = []
distance = [int(1e9)] * (nodes+1)

for _ in range(edges):
    a, b, c = map(int, input().split())
    graph.append((a, b, c))
    
def bellman_ford(start):
    distance[start] = 0
    for i in range(nodes):
        for j in range(edges):
            a = graph[j][0]
            b = graph[j][1]
            c = graph[j][2]
            
            if distance[a]!=int(1e9) and distance[b] > distance[a]+c:
                distance[b] = distance[a]+c
                
                if i==nodes-1: # n-1번 이후 반복에도 값이 갱신되면 음수 순환 존재
                    return True
                
    return False

if bellman_ford(start):
    print("negative cycle found")
else:
    for i in range(2, nodes+1):
        print(distance[i], end=" ")    
```

```
5
8
1 2 -4
1 3 5
1 4 2
1 5 3
2 4 -1
3 4 -7
4 5 6
5 4 -4
-4 5 -5 1 
```

### Floyd-Warshall: 모든 노드에서 다른 모든 노드까지의 최단거리

$O(N^3)$: N개의 노드에 대해 N*N 2차원 리스트 갱신

```python
nodes = int(input())
edges = int(input())
# N개의 노드에 대해 N*N 2차원 리스트 생성
# row는 출발인 경우, col은 도착인 경우
graph = [[int(1e9)] * (nodes+1) for _ in range(nodes+1)] 
# 자기 자신에 대한 cost는 0
for a in range(1, nodes+1):
    for b in range(1, nodes+1):
        if a==b:
            graph[a][b] = 0
            
for _ in range(edges):
    a, b, c = map(int, input().split())
    graph[a][b] = c
    
for i in range(1, nodes+1):
    for a in range(1, nodes+1):
        for b in range(1, nodes+1):
            # AB = min((A->B), (A->현재 확인하고 있는 노드->B))
            graph[a][b] = min(graph[a][b], graph[a][i]+graph[i][b])

for a in range(1, nodes+1):
    for b in range(1, nodes+1):
        print(graph[a][b], end=" ")
    print()
```

```
4
7
1 2 4
1 4 6
2 1 3
2 3 7
3 1 5
3 4 4
4 3 2
0 4 8 6 
3 0 7 9 
5 9 0 4 
7 11 2 0 
```

## Minimum Spanning Tree

최소 연결 부분 그래프 (노드의 수가 n 일때 n-1개의 간선을 갖는 그래프) 중 간선 가중치의 합이 최소인 그래프

```python
nodes = 7
edges = 9
start = 1
graph = [
    [1, 2, 29],
    [1, 5, 75],
    [2, 3, 75],
    [2, 6, 34],
    [3, 4, 7],
    [4, 6, 23],
    [4, 7, 13],
    [5, 6, 53],
    [6, 7, 25]
]
```

### Prim: MST for dense graph

인접 노드 중 최소 가중치로 연결된 노드 선택

$O(N^2)$

```python
import heapq
import collections

undirected_graph = collections.defaultdict(list)
for i in graph:
    a, b, c = i
    undirected_graph[a].append([c, a, b])
    undirected_graph[b].append([c, b, a])
        
def prim(undirected_graph, start):
    visited[start] = True
    q = undirected_graph[start] 
    heapq.heapify(q) # 인접 노드와 연결하는 간선들
    
    r = 0
    mst = []
    while q:
        # print(q)
        c, a, b = heapq.heappop(q)
        
        if not visited[b]:
            visited[b] = True
            mst.append((a, b))
            r += c
            
            for edge in undirected_graph[b]:
                _, _, node = edge # 다음 노드
                if not visited[node]:
                    heapq.heappush(q, edge)
                    
    return r, mst

visited = [False] * (nodes+1)

print(prim(undirected_graph, start))
```

```
(159, [(1, 2), (2, 6), (6, 4), (4, 3), (4, 7), (6, 5)])
```

### Kruskal: MST for sparse graph

모든 가중치 정렬 후 낮은 순으로 추가하되 사이클을 형성하는 간선은 제외

$O(E \log_2 E)$: 정렬에 걸리는 시간복잡도

```python
def find(parent, x): # 루트 노드 찾기
    if parent[x]!=x: # 루트 노드가 아니라면
        parent[x] = find(parent, parent[x]) # 경로 압축
        # return find(parent, parent[x]) # 경로 압축 미적용
    return parent[x] # 경로 압축
    # return x # 경로 압축 미적용

def union(parent, a, b): # 두 집합 합치기
    a = find(parent, a) # 루트 노드 찾기
    b = find(parent, b) # 루트 노드 찾기
    if a<b:
        parent[b] = a # a를 b의 부모 노드로 설정
    else:
        parent[a] = b # b를 a의 부모 노드로 설정

graph = sorted(graph, key=lambda x: x[2])

parent = [0] * (nodes+1)
for i in range(1, nodes+1):
    parent[i] = i
    
r = 0
mst = []
for i in graph:
    a, b, c = i
    if find(parent, a) != find(parent, b): # 루트 노드가 다르면 (= 다른 집합이면)
        # 루트 노드가 같다면 사이클이 발생한 것
        union(parent, a, b) # 두 집합 합치기
        mst.append((a, b))
        r += c

print(parent)
print(r, mst)
```

```
[0, 1, 1, 1, 3, 1, 1, 3]
159 [(3, 4), (4, 7), (4, 6), (1, 2), (2, 6), (5, 6)]
```

### Topological sort: 노드들을 출발->도착 방향에 맞게 정렬

$O(N+E)$

```python
from collections import deque

nodes, edges = map(int, input().split())
indegree = [0] * (nodes+1) # 진입차수: 특정 노드로 들어오는 간선의 갯수
graph = [[] for i in range(nodes+1)]

for _ in range(edges):
    a, b = map(int, input().split())
    graph[a].append(b)
    indegree[b]+=1
    
def topological_sort():
    r = []
    q = deque()
    
    for i in range(1, nodes+1):
        if indegree[i]==0: # 진입차수가 0인 노드를 큐에 넣음
            q.append(i)
            
    while q:
        node = q.popleft()
        r.append(node)
        
        for i in graph[node]: # 해당 노드에서 출발하는 간선들을 제거
            indegree[i]-=1
            if indegree[i]==0: # 진입차수가 0인 노드를 큐에 넣음
                q.append(i)
                
    for i in r:
        print(i, end=" ")
        
topological_sort()
```

```
7 8
1 2
1 5
2 3
2 6
3 4
4 7
5 6
6 4
1 2 5 3 6 4 7 
```
