# m by n matrix

The core idea is that the first number (m or axis 0) represents the most-layered dimension.

## Math

In $3 \times 2$ matrix, 3 represents the vertical axis, and 2 represents the horizontal axis. 

$$\begin{bmatrix}
a & b \\
c & d \\
e & f
\end{bmatrix}$$

It can be conceptualized as a stack of three 2-element 1D arrays.

## Python

```python
m, n = 3, 2
mat = [[0] * n for _ in range(m)]

print(len(mat), len(mat[0]))
print(mat)
```
```
3 2
[[0, 0], [0, 0], [0, 0]]
```

## NumPy

```python
import numpy as np

mat = np.random.randint(0, 10, (3, 2))

print(mat.shape) # (axis 0, axis 1)
print(mat)
```
```
(3, 2)
[[0 1]
 [4 7]
 [8 3]]
```

## PyTorch

```python
import torch

mat = torch.randint(0, 10, (4, 3, 2))

print(mat.shape) # torch.Size([axis 0, axis 1, axis 2])
print(mat)
```
```
torch.Size([4, 3, 2])
tensor([[[7, 9],
         [2, 9],
         [9, 3]],

        [[5, 5],
         [8, 7],
         [5, 6]],

        [[6, 1],
         [7, 3],
         [4, 3]],

        [[6, 0],
         [5, 1],
         [8, 4]]])
```
