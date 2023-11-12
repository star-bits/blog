# DP

## DP

### Longest Common Subsequence (LCS)

```python
def longest_common_subsequence(X, Y):
    # Get the lengths of the input strings
    m = len(X)
    n = len(Y)

    # Create a 2D table to store the length of LCS for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table in a bottom-up manner
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the current characters in both strings match
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            # If the characters don't match, find the max LCS length from the adjacent cells
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # The bottom-right cell of the table contains the length of the LCS
    return dp[m][n]

# Example usage
X = "AGGTAB"
Y = "GXTXAYB"
print("Length of LCS:", longest_common_subsequence(X, Y))
```
```
Length of LCS: 4
```

### Longest Increasing Subsequence (LIS)
```python
def longest_increasing_subsequence(arr):
    # Get the length of the input array
    n = len(arr)

    # Create a 1D table to store the length of LIS for subproblems
    # Initialize all values in the table to 1, as each element is a valid subsequence of length 1
    dp = [1] * n

    # Fill the table in a bottom-up manner
    for i in range(1, n):
        for j in range(0, i):
            # If the current element is greater than the previous element and the LIS at the current
            # position is smaller than the LIS at the previous position + 1, update the LIS
            if arr[i] > arr[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1

    # The length of the LIS is the maximum value in the table
    return max(dp)

# Example usage
arr = [10, 22, 9, 33, 21, 50, 41, 60]
print("Length of LIS:", longest_increasing_subsequence(arr))
```
```
Length of LIS: 5
```

### 0/1 Knapsack problem
```python
def knapsack(values, weights, W):
    # Get the number of items
    n = len(values)

    # Create a 2D table to store the maximum value of the knapsack for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # Fill the table in a bottom-up manner
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            # If the current item's weight is less than or equal to the current weight capacity
            if weights[i - 1] <= w:
                # Determine if taking the current item provides a better value than not taking it
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            # If the current item's weight is more than the current weight capacity, don't take it
            else:
                dp[i][w] = dp[i - 1][w]

    # The bottom-right cell of the table contains the maximum value of the knapsack
    return dp[n][W]

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
print("Maximum value of the knapsack:", knapsack(values, weights, W))
```
```
Maximum value of the knapsack: 220
```

### Matrix Chain Multiplication (MCM)
```python
def matrix_chain_multiplication(dimensions):
    # Get the number of matrices
    n = len(dimensions) - 1

    # Create a 2D table to store the minimum number of scalar multiplications for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * n for _ in range(n)]

    # Fill the table in a bottom-up manner
    for length in range(2, n + 1):  # Length of the chain
        for i in range(n - length + 1):  # Starting index of the chain
            j = i + length - 1  # Ending index of the chain
            dp[i][j] = float('inf')
            for k in range(i, j):  # Possible partition positions
                # Calculate the cost of multiplication for the current partition
                cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                # Update the table with the minimum cost
                dp[i][j] = min(dp[i][j], cost)

    # The top-right cell of the table contains the minimum number of scalar multiplications
    return dp[0][n - 1]

# Example usage
dimensions = [30, 35, 15, 5, 10, 20, 25]
print("Minimum number of scalar multiplications:", matrix_chain_multiplication(dimensions))
```
```
Minimum number of scalar multiplications: 15125
```

### Coin Change problem
```python
def coin_change(coins, amount):
    # Create a 1D table to store the minimum number of coins for subproblems
    # Initialize all values in the table to be greater than the maximum possible number of coins
    dp = [float('inf')] * (amount + 1)

    # Base case: The minimum number of coins needed to make change for 0 is 0
    dp[0] = 0

    # Fill the table in a bottom-up manner
    for coin in coins:
        for i in range(coin, amount + 1):
            # If the current coin can be used to make change for the current amount,
            # update the table with the minimum number of coins
            dp[i] = min(dp[i], dp[i - coin] + 1)

    # If the last cell of the table contains a value greater than the maximum possible number of coins,
    # it means that it's not possible to make change for the given amount using the available coins
    return dp[amount] if dp[amount] != float('inf') else -1

# Example usage
coins = [1, 4, 5]
amount = 8
print("Minimum number of coins:", coin_change(coins, amount))
```
```
Minimum number of coins: 2
```

### Edit Distance (Levenshtein distance)
```python
def edit_distance(str1, str2):
    # Get the lengths of the input strings
    m = len(str1)
    n = len(str2)

    # Create a 2D table to store the minimum number of edits for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table in a bottom-up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If the first string is empty, the edit distance is the length of the second string
            if i == 0:
                dp[i][j] = j
            # If the second string is empty, the edit distance is the length of the first string
            elif j == 0:
                dp[i][j] = i
            # If the current characters in both strings match, no edits are required
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            # If the characters don't match, consider the three possible operations (insert, delete, replace)
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    # Insert
                                  dp[i - 1][j],     # Remove
                                  dp[i - 1][j - 1]) # Replace

    # The bottom-right cell of the table contains the minimum edit distance
    return dp[m][n]

# Example usage
str1 = "sunday"
str2 = "saturday"
print("Minimum edit distance:", edit_distance(str1, str2))
```
```
Minimum edit distance: 3
```

### Rod Cutting problem
```python
def rod_cutting(prices, n):
    # Create a 1D table to store the maximum revenue for subproblems
    # Initialize all values in the table to 0
    dp = [0] * (n + 1)

    # Fill the table in a bottom-up manner
    for i in range(1, n + 1):
        max_value = float('-inf')
        for j in range(1, i + 1):
            # Find the maximum revenue by checking all possible cuts for the current rod length
            max_value = max(max_value, prices[j - 1] + dp[i - j])
        # Update the table with the maximum revenue for the current rod length
        dp[i] = max_value

    # The last cell of the table contains the maximum revenue for the given rod length
    return dp[n]

# Example usage
prices = [1, 5, 8, 9, 10, 17, 17, 20]
n = len(prices)
print("Maximum revenue:", rod_cutting(prices, n))
```
```
Maximum revenue: 22
```

### Maximum Sum Subarray (Kadane's Algorithm)
```python
def max_sum_subarray(arr):
    # Initialize the maximum sum and current sum to the first element of the array
    max_sum = curr_sum = arr[0]

    # Iterate through the array, starting from the second element
    for i in range(1, len(arr)):
        # Update the current sum to be the maximum of the current element or the sum of the current element and the previous sum
        curr_sum = max(arr[i], curr_sum + arr[i])

        # Update the maximum sum if the current sum is greater
        max_sum = max(max_sum, curr_sum)

    # Return the maximum sum of the subarray
    return max_sum

# Example usage
arr = [-2, -3, 4, -1, -2, 1, 5, -3]
print("Maximum sum of the subarray:", max_sum_subarray(arr))
```

### Egg Dropping problem
```python
def egg_drop(eggs, floors):
    # Create a 2D table to store the minimum number of trials for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * (floors + 1) for _ in range(eggs + 1)]

    # Base cases: if there is only one egg or one floor
    for i in range(1, eggs + 1):
        dp[i][1] = 1
        dp[i][0] = 0

    # Base cases: if there is only one egg, the number of trials equals the number of floors
    for j in range(1, floors + 1):
        dp[1][j] = j

    # Fill the table in a bottom-up manner
    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            dp[i][j] = float('inf')
            for x in range(1, j + 1):
                # Calculate the maximum number of trials in the worst case (when the egg breaks or doesn't break)
                res = 1 + max(dp[i - 1][x - 1], dp[i][j - x])

                # Update the table with the minimum number of trials in the worst case
                dp[i][j] = min(dp[i][j], res)

    # The bottom-right cell of the table contains the minimum number of trials in the worst case
    return dp[eggs][floors]

# Example usage
eggs = 2
floors = 10
print("Minimum number of trials in the worst case:", egg_drop(eggs, floors))
```
```
Minimum number of trials in the worst case: 4
```

### Palindromic Subsequence
```python
def longest_palindromic_subsequence(s):
    # Get the length of the input string
    n = len(s)

    # Create a 2D table to store the length of the longest palindromic subsequence for subproblems
    # Initialize all values in the table to 0
    dp = [[0] * n for _ in range(n)]

    # Base case: the length of the longest palindromic subsequence for a single character is 1
    for i in range(n):
        dp[i][i] = 1

    # Fill the table in a bottom-up manner
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # If the current characters match, add 2 to the length of the longest palindromic subsequence
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            # If the characters don't match, take the maximum of the lengths of the longest palindromic subsequences
            # without the current characters
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    # The top-right cell of the table contains the length of the longest palindromic subsequence
    return dp[0][n - 1]

# Example usage
s = "BBABCBCAB"
print("Length of the longest palindromic subsequence:", longest_palindromic_subsequence(s))
```
```
Length of the longest palindromic subsequence: 7
```

## Other

### KMP
```python
def kmp_search(pattern, text):
    def create_lps(pattern):
        lps = [0] * len(pattern)  # Initialize the LPS array with all values set to 0
        length = 0  # Length of the previous longest prefix-suffix
        i = 1

        # Calculate the LPS array values
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return lps

    # Preprocess the pattern to create the LPS (longest proper prefix-suffix) array
    lps = create_lps(pattern)

    i = 0  # Index for text
    j = 0  # Index for pattern

    # Search for the pattern in the text
    while i < len(text):
        # If the characters match, move both indices forward
        if text[i] == pattern[j]:
            i += 1
            j += 1

        # If the entire pattern is found, return the starting index
        if j == len(pattern):
            return i - j

        # If there is a mismatch after some characters have matched
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:  # Move the pattern index j to the value in the LPS array
                j = lps[j - 1]
            else:  # Move the text index i forward if j is 0
                i += 1

    # Pattern not found in the text
    return -1

# Example usage
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
index = kmp_search(pattern, text)
print("Pattern found at index:", index)
```
```
Pattern found at index: 10
```

### N-Queen
```python
def solve_n_queens(n):
    """
    Solve the N-Queens problem for a board of size n x n.
    
    Parameters:
    - n (int): The size of the board.
    
    Returns:
    - list: A list of lists representing the board. Each inner list contains the column index
            where a queen is placed in that row.
    """
    def is_valid(board, row, col):
        """Check if placing a queen at (row, col) is valid."""
        for i in range(row):
            if board[i] == col or \
               abs(board[i] - col) == abs(i - row):
                return False
        return True

    def backtrack(board, row):
        """Use backtracking to place queens."""
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_valid(board, row, col):
                board[row] = col
                backtrack(board, row + 1)

    solutions = []
    board = [-1] * n  # Initialize board with -1, meaning no queen is placed
    backtrack(board, 0)
    return solutions

# Solve the 8-Queens problem
n = 8
solutions = solve_n_queens(n)

# Print the solutions
for i, solution in enumerate(solutions):
    print(f"Solution {i + 1}: {solution}")
```
```
Solution 1: [0, 4, 7, 5, 2, 6, 1, 3]
Solution 2: [0, 5, 7, 2, 6, 3, 1, 4]
Solution 3: [0, 6, 3, 5, 7, 1, 4, 2]
Solution 4: [0, 6, 4, 7, 1, 3, 5, 2]
Solution 5: [1, 3, 5, 7, 2, 0, 6, 4]
Solution 6: [1, 4, 6, 0, 2, 7, 5, 3]
Solution 7: [1, 4, 6, 3, 0, 7, 5, 2]
Solution 8: [1, 5, 0, 6, 3, 7, 2, 4]
Solution 9: [1, 5, 7, 2, 0, 3, 6, 4]
Solution 10: [1, 6, 2, 5, 7, 4, 0, 3]
Solution 11: [1, 6, 4, 7, 0, 3, 5, 2]
Solution 12: [1, 7, 5, 0, 2, 4, 6, 3]
Solution 13: [2, 0, 6, 4, 7, 1, 3, 5]
Solution 14: [2, 4, 1, 7, 0, 6, 3, 5]
Solution 15: [2, 4, 1, 7, 5, 3, 6, 0]
Solution 16: [2, 4, 6, 0, 3, 1, 7, 5]
Solution 17: [2, 4, 7, 3, 0, 6, 1, 5]
Solution 18: [2, 5, 1, 4, 7, 0, 6, 3]
Solution 19: [2, 5, 1, 6, 0, 3, 7, 4]
Solution 20: [2, 5, 1, 6, 4, 0, 7, 3]
Solution 21: [2, 5, 3, 0, 7, 4, 6, 1]
Solution 22: [2, 5, 3, 1, 7, 4, 6, 0]
Solution 23: [2, 5, 7, 0, 3, 6, 4, 1]
Solution 24: [2, 5, 7, 0, 4, 6, 1, 3]
Solution 25: [2, 5, 7, 1, 3, 0, 6, 4]
Solution 26: [2, 6, 1, 7, 4, 0, 3, 5]
Solution 27: [2, 6, 1, 7, 5, 3, 0, 4]
Solution 28: [2, 7, 3, 6, 0, 5, 1, 4]
Solution 29: [3, 0, 4, 7, 1, 6, 2, 5]
Solution 30: [3, 0, 4, 7, 5, 2, 6, 1]
Solution 31: [3, 1, 4, 7, 5, 0, 2, 6]
Solution 32: [3, 1, 6, 2, 5, 7, 0, 4]
Solution 33: [3, 1, 6, 2, 5, 7, 4, 0]
Solution 34: [3, 1, 6, 4, 0, 7, 5, 2]
Solution 35: [3, 1, 7, 4, 6, 0, 2, 5]
Solution 36: [3, 1, 7, 5, 0, 2, 4, 6]
Solution 37: [3, 5, 0, 4, 1, 7, 2, 6]
Solution 38: [3, 5, 7, 1, 6, 0, 2, 4]
Solution 39: [3, 5, 7, 2, 0, 6, 4, 1]
Solution 40: [3, 6, 0, 7, 4, 1, 5, 2]
Solution 41: [3, 6, 2, 7, 1, 4, 0, 5]
Solution 42: [3, 6, 4, 1, 5, 0, 2, 7]
Solution 43: [3, 6, 4, 2, 0, 5, 7, 1]
Solution 44: [3, 7, 0, 2, 5, 1, 6, 4]
Solution 45: [3, 7, 0, 4, 6, 1, 5, 2]
Solution 46: [3, 7, 4, 2, 0, 6, 1, 5]
Solution 47: [4, 0, 3, 5, 7, 1, 6, 2]
Solution 48: [4, 0, 7, 3, 1, 6, 2, 5]
Solution 49: [4, 0, 7, 5, 2, 6, 1, 3]
Solution 50: [4, 1, 3, 5, 7, 2, 0, 6]
Solution 51: [4, 1, 3, 6, 2, 7, 5, 0]
Solution 52: [4, 1, 5, 0, 6, 3, 7, 2]
Solution 53: [4, 1, 7, 0, 3, 6, 2, 5]
Solution 54: [4, 2, 0, 5, 7, 1, 3, 6]
Solution 55: [4, 2, 0, 6, 1, 7, 5, 3]
Solution 56: [4, 2, 7, 3, 6, 0, 5, 1]
Solution 57: [4, 6, 0, 2, 7, 5, 3, 1]
Solution 58: [4, 6, 0, 3, 1, 7, 5, 2]
Solution 59: [4, 6, 1, 3, 7, 0, 2, 5]
Solution 60: [4, 6, 1, 5, 2, 0, 3, 7]
Solution 61: [4, 6, 1, 5, 2, 0, 7, 3]
Solution 62: [4, 6, 3, 0, 2, 7, 5, 1]
Solution 63: [4, 7, 3, 0, 2, 5, 1, 6]
Solution 64: [4, 7, 3, 0, 6, 1, 5, 2]
Solution 65: [5, 0, 4, 1, 7, 2, 6, 3]
Solution 66: [5, 1, 6, 0, 2, 4, 7, 3]
Solution 67: [5, 1, 6, 0, 3, 7, 4, 2]
Solution 68: [5, 2, 0, 6, 4, 7, 1, 3]
Solution 69: [5, 2, 0, 7, 3, 1, 6, 4]
Solution 70: [5, 2, 0, 7, 4, 1, 3, 6]
Solution 71: [5, 2, 4, 6, 0, 3, 1, 7]
Solution 72: [5, 2, 4, 7, 0, 3, 1, 6]
Solution 73: [5, 2, 6, 1, 3, 7, 0, 4]
Solution 74: [5, 2, 6, 1, 7, 4, 0, 3]
Solution 75: [5, 2, 6, 3, 0, 7, 1, 4]
Solution 76: [5, 3, 0, 4, 7, 1, 6, 2]
Solution 77: [5, 3, 1, 7, 4, 6, 0, 2]
Solution 78: [5, 3, 6, 0, 2, 4, 1, 7]
Solution 79: [5, 3, 6, 0, 7, 1, 4, 2]
Solution 80: [5, 7, 1, 3, 0, 6, 4, 2]
Solution 81: [6, 0, 2, 7, 5, 3, 1, 4]
Solution 82: [6, 1, 3, 0, 7, 4, 2, 5]
Solution 83: [6, 1, 5, 2, 0, 3, 7, 4]
Solution 84: [6, 2, 0, 5, 7, 4, 1, 3]
Solution 85: [6, 2, 7, 1, 4, 0, 5, 3]
Solution 86: [6, 3, 1, 4, 7, 0, 2, 5]
Solution 87: [6, 3, 1, 7, 5, 0, 2, 4]
Solution 88: [6, 4, 2, 0, 5, 7, 1, 3]
Solution 89: [7, 1, 3, 0, 6, 4, 2, 5]
Solution 90: [7, 1, 4, 2, 0, 6, 3, 5]
Solution 91: [7, 2, 0, 5, 1, 4, 6, 3]
Solution 92: [7, 3, 0, 2, 5, 1, 6, 4]
```
