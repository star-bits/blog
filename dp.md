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


## More DFS and BFS

## More Graph
