# Sorting algorithms

## Bubble sort: 인접 element끼리 swap

$O(n^2)$

```python
l = [i for i in range(9, -1, -1)]
print(l)

def bubblesort(l):
    for i in range(len(l)-1): # sort completed in the penultimate iteration.
        for j in range(len(l)-i-1): # -i because i elements in the right are already sorted.
            if l[j] > l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
                
bubblesort(l)
print(l)
```

## Selection sort: 최솟값 찾아 맨 앞으로 보냄

$O(n^2)$

```python
l = [i for i in range(9, -1, -1)]
print(l)

def selectionsort(l):
    for i in range(len(l)-1):
        min_idx = i
        for j in range(i+1, len(l)):
            if l[j] < l[min_idx]:
                min_idx = j
        l[i], l[min_idx] = l[min_idx], l[i]
        
selectionsort(l)
print(l)
```

최댓값 찾아 맨 뒤로 보냄

```python
l = [i for i in range(9, -1, -1)]
print(l)

def selectionsort(l):
    for i in range(len(l)-1, -1, -1):
        max_idx = i
        for j in range(i):
            if l[j] > l[max_idx]:
                max_idx = j
        l[i], l[max_idx] = l[max_idx], l[i]
        
selectionsort(l)
print(l)
```

## Insertion sort: 오른쪽으로 한칸씩 진출하면서 알맞은 자리에 꽂아넣음

Worst: $O(n^2)$, Average: $O(n^2)$, Best: $O(n)$

```python
l = [i for i in range(9, -1, -1)]
print(l)

def insertionsort(l):
    for i in range(1, len(l)):
        for j in range(i, 0, -1):
            if l[j-1]>l[j]:
                l[j-1], l[j] = l[j], l[j-1]
                
insertionsort(l)
print(l)
```

## Merge sort: merging two sorted lists, over and over

$O(n \log n)$

```python
l = [i for i in range(9, -1, -1)]
print(l)

def mergesort(l):
    if len(l)==1:
        return l
    
    mid = len(l)//2
    left = mergesort(l[:mid])
    right = mergesort(l[mid:])
    return merge(left, right)
    
def merge(left, right):
    output = []
    i = j = 0 # pointers for `left` and `right`
    
    while i<len(left) and j<len(right):
        if left[i] < right[j]:
            output.append(left[i])
            i+=1
        else:
            output.append(right[j])
            j+=1     
    output.extend(left[i:])
    output.extend(right[j:])
    
    return output

l = mergesort(l)
print(l)
```

## Quick sort: pivot 골라서 pivot보다 작으면 왼쪽, 크면 오른쪽

Worst: $O(n^2)$, Average: $O(n \log n)$, Best: $O(n \log n)$

```python
l = [2, 7, 1, 3, 6, 5, 4]
print(l)

def quicksort(l, first, last): # index of the first and last element
    if first < last:
        pivot = partition(l, first, last)
        quicksort(l, first, pivot-1)
        quicksort(l, pivot+1, last)
        
def partition(l, first, last):
    pivot = l[last]
    j = first
    
    # pivot보다 l[i]가 크면 j를 그 element에 고정하고 swap 없이 다음 i로 넘어감.
    # pivot보다 l[i]가 작으면 l[i]와 l[j] swap하고 j+=1.
    # 마지막에 j와 pivot도 swap.
    # j는 항상 pivot보다 큰 첫번째 element의 index.
    for i in range(first, last):
        if l[i] < pivot:
            l[j], l[i] = l[i], l[j]
            j = j+1
            
    l[j], l[last] = l[last], l[j]
    return j

quicksort(l, 0, len(l)-1)
print(l)
```

```python
e.g. 2713654
i=0            ->1->2            ->3            ->4->5
j=0->2713654->1      ->2173654->2   ->2137654->3      ->2134657
```

## Heap sort

$O(n \log n)$

```python

```

## Counting sort: `count[l[i]]-1`은 `l[i]`이 위치할 수 있는 최대 인덱스

$O(n+k)$

```python
l = [4, 2, 2, 8, 3, 3, 1]
print(l)

def countingsort(l):
    output = [0] * len(l)
    count = [0] * (max(l)+1)
    
    # count
    for i in l:
        count[i]+=1
    
    # cumulative count
    for i in range(max(l)):
        count[i+1]+=count[i]
    # print(count)
    
    # i: reversed l
    for i in reversed(range(len(l))):
        output[count[l[i]]-1] = l[i] # output 리스트에 l[i]이 들어갈 수 있는 최대 인덱스는 count[l[i]]-1 
        count[l[i]]-=1 # 방금 l[i]을 올바른 인덱스에 넣었으므로 다음 l[i]은 그것보다 하나 작은 인덱스에 넣어야 됨
    
    return output

l = countingsort(l)
print(l)
```

## Radix sort: decimal place 올려가면서 `l`->`buckets`, `buckets`->`l`

$O(d*(n+k))$

```python
l = [152, 73, 69, 41, 28, 1247, 2, 33, 674, 388]
print(l)

def radixsort(l):
    # 해당 digit의 숫자(0-9)에 따라 담아놓을 queue들
    buckets = [[] for _ in range(10)]
    
    max_ = max(l)
    decimal = 1 # decimal place to examine
    
    while max_ >= decimal:
        while l: # l에서 빼서 corresponding한 bucket들로 이동
            i = l.pop(0)
            buckets[(i//decimal)%10].append(i)
            
        for bucket in buckets: # bucket들에서 순서대로 빼서 l로 이동
            while bucket:
                l.append(bucket.pop(0))
                
        decimal*=10
        
radixsort(l)
print(l)
```

