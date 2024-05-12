# DP

## 백준용 파이썬 템플릿

```python
data = """"""
inputs = iter(data.split('\n'))

def input():
    return next(inputs)

# import sys
# input = sys.stdin.readline
```

## Accumulative Computation (Dynamic Programming)

```python
# 한 걸음에 한 단 또는 두 단 또는 세 단 올라 이 계단을 모두 올라가는 경우의 수

dp = [0 for _ in range(n+1)]
dp[0] = 1 # 시작점에 도달하는 경우의 수
dp[1] = 1 # 첫번째 계단에 도달하는 경우의 수
dp[2] = 2 # 두번째 계단에 도달하는 경우의 수

for i in range (3, n+1):
    # all the ways to reach the i-th step
    dp[i] = dp[i-1] + dp[i-2] + dp[i-3]

print(dp[n])
```

```python
# 한 번에 한 계단 또는 두 계단씩 오르되 연속된 세 개의 계단을 모두 밟지 않고 오르면서 마지막 계단은 반드시 밟는 경우의 수

dp = [0 for _ in range(n+2)]
dp[0] = l[0] # l은 각 계단의 점수 리스트
dp[1] = max(l[0]+l[1], l[1])
dp[2] = max(l[0]+l[2], l[1]+l[2])

for i in range(3, n):
    dp[i] = max(dp[i-3]+l[i-1]+l[i], dp[i-2]+l[i])

print(dp[n-1])
```

```python
# Longest Decreasing Subsequence

dp = [1 for _ in range(n)]
# will eventually represent the length of the longest decreasing subsequence ending at index i

for i in range(1, n):
    for j in range(i):
        if arr[j] > arr[i] and dp[j]+1 > dp[i]:
            # dp[j] + 1 > dp[i]: checks if
            # the longest decreasing subsequence ending at j plus the element at i
            # would be longer than the current longest subsequence recorded at i
            dp[i] = dp[j]+1

print(max(dp))
```

## Combination and Permutation


## More DFS and BFS

## More Graph
