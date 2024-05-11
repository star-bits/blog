```python
def stair_combinations(n):
    dp = [0 for _ in range(n+1)]
    dp [0] = 1 # one way to reach the base step
    dp [1] = 1 # one way to reach the first step
    dp [2] = 2 # two ways to reach the second step

    for i in range (3, n+1):
        # all the ways to reach the i-th step
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]

    return dp[n]

# 한 걸음에 한 단 또는 두 단 또는 세 단 올라 이 계단을 모두 올라가는 경우의 수
print(stair_combinations(6))
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
