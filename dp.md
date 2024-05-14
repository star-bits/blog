# DP

## Accumulative Computation (Dynamic Programming)

### Subsequences

```python
def lis(arr):
    n = len(arr)
    
    dp = [1 for _ in range(n)]
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j]+1 > dp[i]:
                # lis ending at j + element i > current lis recorded at i
                dp[i] = dp[j]+1
    
    max_length = max(dp)
    
    lis = []
    current_length = max_length
    for i in range(n-1, -1, -1):
        if dp[i] == current_length:
            lis.append(arr[i])
            current_length -= 1
    
    lis.reverse()
    return lis
```

```python
def lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = ""
    i, j = m, n
    while i>0 and j>0:
        if str1[i-1] == str2[j-1]:
            lcs = str1[i-1] + lcs
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return lcs
```

### Stairs 

```python
# 한 단 또는 두 단씩 올라감

def max_score(scores):
    n = len(scores)

    dp = [0 for _ in range(n)]
    dp[0] = scores[0]
    dp[1] = scores[0] + scores[1]

    for i in range(2, n):
        dp[i] = max(dp[i-1] + scores[i], dp[i-2] + scores[i])

    return dp[n-1]
```

```python
# 한 단 또는 두 단씩 올라감 하지만 세 단 연속은 금지

def max_score(scores):
    n = len(scores)

    dp = [0 for _ in range(n)]
    dp[0] = scores[0]
    dp[1] = scores[0] + scores[1]
    dp[2] = max(scores[0] + scores[2], scores[1] + scores[2])

    for i in range(3, n):
        dp[i] = max(dp[i-2] + scores[i], dp[i-3] + scores[i-1] + scores[i])

    return dp[n-1]
```

```python
# 해당 단에서 올라갈 수 있는 최대 단 수의 리스트가 주어짐

def variable_jump(jumps):
    n = len(jumps)
    
    dp = [float('inf')] * n
    dp[0] = 0

    for i in range(1, n):
        for j in range(i):
            if j+jumps[j] >= i:
                dp[i] = min(dp[i], dp[j]+1)

    return dp[n-1] if dp[n-1]!=float('inf') else -1
```

### Coins 

```python
target = 5
coin_types = [1, 2, 5]

def num_combinations(target, coin_types):
    dp = [1] + [0 for _ in range(target)]
    
    for coin_type in coin_types:
        for i in range(coin_type, target+1):
            dp[i] += dp[i-coin_type]
    
    return dp[target]
```

```python
target = 11
coins = [1, 5, 5, 6]

def target_sum(target, coins):
    dp = [True] + [False for _ in range(target)]
    
    for coin in coins:
        for i in range(target, coin-1, -1):
            dp[i] = dp[i] or dp[i-coin]
    
    return dp[target]
```

```python
target = 11
coins = [1, 5, 5, 6]

def target_sum(target, coins):
    dp = [1] + [0 for _ in range(target)]

    for coin in coins:
        for i in range(target, coin-1, -1):
            dp[i] += dp[i-coin]

    return dp[target]
```

## 백준용 파이썬 템플릿

```python
data = """"""
inputs = iter(data.split('\n'))

def input():
    return next(inputs)

# import sys
# input = sys.stdin.readline
```
## Some notable clever approaches

```python
# Minimum Cost to Make All Characters Equal

def min_cost(s):
    return sum(min(i, len(s)-i) for i in range(1, len(s)) if s[i]!=s[i-1])
```

```python
# Reverse a Nested List

l = [1, [2, 3, [4, 5]]]

def reverse(l):
    if not isinstance(l, list):
        return l

    return [reverse(i) for i in l[::-1]]

```

```python
l = [1, [2, 3, [4, 5]]]

def reverse(l):
    out = ""
    for i in str(l)[::-1]:
        if i=='[':
            out += ']'
        elif i==']':
            out += '['
        else:
            out += i
    return eval(out)
```

```python
# 덩치 등수

for i in l:
    rank = 1
    for j in l:
        if i[0]<j[0] and i[1]<j[1]:
            rank += 1
    print(rank)
```

## key=lambda x

```python
sl = sorted(l, key=lambda x: (x[0], -x[1])) # [0]은 내림차순, [1]은 오름차순

nd = d.copy()

nd.clear()

sd = dict(sorted(d.items(), reverse=True))

sd = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

fd = {k: v for k, v in d.items() if v>3}

mv = max(d.items(), key=lambda x: x[1])[0]

lk = list(d) # list of d is a list of keys of d

mv = min(d, key=lambda x: (-d[x], x)) # min iterates over each key in d
```

## itertools

```python
# product returns an iterator that yields tuples containing one item from each input iterable

from itertools import product

candidates = [[3, 4], [-1], [10, 11, 12], [-1], [1, 2]]

print(min(product(*candidates)))
```

```python
# 1부터 n까지의 자연수 중 중복 없이 m개를 고른 수열

from itertools import combinations

for combination in combinations(range(1, n+1), m):
    print(combination)
```

```python
# combination

def dfs(n, m, start=1, seq=None):
    if seq is None:
        seq = []

    if len(seq)==m:
        print(seq)
        return
    
    for i in range(start, n+1):
        seq.append(i)
        dfs(n, m, i+1, seq)
        seq.pop()

dfs(n, m)
```

```python
# permutation

def dfs(n, m, start=1, seq=None, used=None):
    if seq is None:
        seq = []
    if used is None:
        used = [False] * (n + 1)

    if len(seq)==m:
        print(seq)
        return
    
    for i in range(start, n+1):
        if not used[i]:
            seq.append(i)
            used[i] = True
            dfs(n, m, start, seq, used)
            seq.pop()
            used[i] = False

dfs(n, m)
```

## More DFS and BFS

```python
# Max Depth

visited = [False for _ in range(n+1)]
def bfs(start):
    num = 0
    q = deque()
    q.append(start)
    visited[start] = True

    while q:
        v = q.popleft()
        for nv in graph[v]:
            if not visited[nv]:
                visited[nv] = True
                q.append(nv)
                num += 1

    return num

bfs(start)        
```

```python
# Max Depth

num = 0
visited = [False for _ in range(n+1)]
def dfs(v):
    global num
    visited[v] = True

    for nv in graph[v]:
        if not visited[nv]:
            num += 1
            dfs(nv)

dfs(start)
```

```python
# Number of Connected Components

from collections import deque

graph = [[] for _ in range(n+1)]
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

num = 0
visited = [False for _ in range(n+1)]
def bfs(v):
    q = deque()
    q.append(v)
    visited[v] = True

    while q:
        v = q.popleft()
        for nv in graph[v]:
            if not visited[nv]:
                q.append(nv)
                visited[nv] = True

for v in range(1, n+1):
    if not visited[v]:
        num += 1
        bfs(v)
```

```python
# Number of Connected Components

from collections import deque

rows, cols = len(grid), len(grid[0])

num = 0
visited = set()
def bfs(r, c):
    q = deque()
    q.append((r, c))
    visited.add((r, c))

    while q:
        r, c = q.popleft()
        nd = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in nd:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]=='1' and (nr, nc) not in visited:
                q.append((nr, nc))
                visited((nr, nc))

for r in range(rows):
    for c in range(cols):
        if grid[r][c]=='1' and (r, c) not in visited:
            num += 1
            bfs(r, c)
```

```python
# Number of Connected Components

rows, cols = len(grid), len(grid[0])

num = 0
visited = set()
def dfs(r, c):
    visited.add((r, c))
    
    nd = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in nd:
        nr, nc = r+dr, c+dc
        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]=='1' and (nr, nc) not in visited:
            dfs(nr, nc)

for r in range(rows):
    for c in range(cols):
        if grid[r][c]=='1' and (r, c) not in visited:
            num += 1
            dfs(r, c)
```

## More Graph

## Even More Graph (Shortest Path)

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    pq = [(0, start)]
    while pq:
        current_cost, current_node = heapq.heappop(pq)

        if current_cost > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            new_cost = current_cost + weight

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))

    return distances
```

```python
def bellman_ford(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Relax edges |V| - 1 times
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    # Check for negative-weight cycles
    for node in graph:
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    return distances
```

```python
def floyd_warshall(graph):
    nodes = list(graph.keys())
    distances = {node: {neighbor: float('inf') for neighbor in nodes} for node in nodes}
    for node in nodes:
        distances[node][node] = 0
        for neighbor, weight in graph[node]:
            distances[node][neighbor] = weight

    for i in nodes:
        for a in nodes:
            for b in nodes:
                # Compare the cost of the a -> i -> b route with the direct a -> b route
                if distances[a][b] > distances[a][i] + distances[i][b]:
                    distances[a][b] = distances[a][i] + distances[i][b]

    return distances
```
