# Notes from the [fastai book](https://github.com/fastai/fastbook)

## Some random stuff
- Criticism that deep learning is 'blackbox' couldn't be further from the truth.
- Non-images can be represented as images, and those images can be used in image classification. Problems deal with non-images such as sound, time series, and mouse movement were converted into images and deep learning showed SOTA or near-SOTA performances.
- Start blogging. It's like a resume, only better.

## L1 and L2 norm
- L1 norm (Mean Absolute Value): absolute value of differences -> mean of it
- L2 norm (RMSE): square of differences -> mean of it -> square root of it
- Difference between L1 norm and MSE: latter penalizes bigger mistakes more heavily and is more lenient with small mistakes.

## PyTorch Autograd
- In autograd, if any input tensor of an operation has `requires_grad=True`, the computation will be tracked.
```python
x = torch.tensor([3., 4.]).requires_grad_()
x
```
```
tensor([3., 4.], requires_grad=True)
```
```python
def f(x): return (x**2).sum()
y = f(x)
y
```
```
tensor(25., grad_fn=<SumBackward0>)
```
```python
y.backward()
x.grad
```
```
tensor([6., 8.])
```
- 덧셈노드의 역전파는 상류의 값을 그대로 하류로 흘려보냄. 따라서 2를 곱해주기만 하면 됨.
- 그런데 2를 곱해줘야 한다는 정보는 어디에 저장? `grad_fn=<PowBackward0>`에?
- 여러 operation을 한꺼번에 한 경우 마지막 operation의 `grad_fn`만 보이는듯.
- power operation 이후 addition operation을 하니 `grad_fn`으로 `<SumBackward0>`만 찍힘.
```python
x = torch.tensor([3., 4.]).requires_grad_()
x
```
```
tensor([3., 4.], requires_grad=True)
```
```python
def f(x): return torch.mul(*(x**2))
y = f(x)
y
```
```
tensor(144., grad_fn=<MulBackward0>)
```
```python
y.backward()
x.grad
```
```
tensor([96., 72.])
```
- 곱셈노드의 역전파는 순전파 때의 값을 서로 바꿔 곱해 하류로 흘려보냄. 
- `6*16=96`, `8*9=72`
- `9, 16`의 정보는 `grad_fn=<MulBackward0>`에 저장해두나봄. 

## Loss
- Loss is a whatever function we've decided to use to optimize the parameters of our model. 
- Accuracy is not useful as loss function. It's about either being right or wrong. Derivative of accuracy is nil everywhere and infinity at the threshold. Loss must be a function that has a meaningful derivative. 

## Why use deeper models?
- Performance. Deeper model means less parameters, and less parameters mean that model trains more quickly and needs less memory.

## Softmax
```python
def softmax(x): return exp(x) / exp(x).sum()
```
- Taking exponential ensures all our numbers are positive. 
- Dividing by the sum ensures all our numbers add up to 1.

## Early stopping is unlikely to give the best result
- Because by the time training is stopped by early stopping, learning rate hasn't reach the small values.
- Instead, retrain from scratch, and this time select a total number of epochs based on where the previous best result were.

## Sigmoid+BCE, Softmax+CE, MSE
- Sigmoid+BCE: nn.BCEWithLogitsLoss
  - $y = \frac{1}{1+\exp(-x)}$, $L = -(t \cdot \log (y) + (1-t) \cdot \log (1-y))$
  - $\frac{\partial y}{\partial x} = y(1-y)$, $\frac{\partial L}{\partial y} = - \frac{y}{y} + \frac{1-t}{1-y}$
  - $\frac{\partial L}{\partial y} = y-t$
- Softmax+CE: nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss(Negative Log Likelihood Loss)
- MSE: nn.MSELoss
  - $L = \frac{1}{2}(t_1 - x_1)^2 + \frac{1}{2}(t_2 - x_2)^2$
  - $\frac{\partial L}{\partial x_1} = 2 \cdot \frac{1}{2} (t_1 - x_1) \cdot (-1) + 0 = x_1-t_1$
- Both have `output - target` as backprop.

## Mixup and label-smoothing
- Model output always goes through sigmoid or softmax -- is never exactly 0 or 1. 
- Mixup and label-smoothing solve it, since their targets are somewhere between 0 and 1. 
- You won't generally see significant improvements from mixup and label-smoothing until later epochs.

## Weight decay (L2 regularization)
- Adds the sum of all weights squared to the loss function.
- The idea behind it is that it encourages the weights to be small, because larger coefficients leads to sharper canyons in the loss function.
```python
loss = loss + weight_decay_rate*(parameters**2).sum()
```
- Above is effectively the same as below:
```python
parameters.grad += weight_decay_rate * (2*parameters)
```

## Bagging and random forest
- Bagging: averaging models
- Feeding random subset of data to multiple models and averaging them -- bagging -- shows better performance.
- Also using random subset of features (or columns, as they say in traditional machine learning community) along with bagging is called random forest.
- OOB(Out-of-Bag) error: metric of model calculated by using unused data from bagging -- which I guess is kinda like cross-validation error, but for random forest.
- Random forests are resilient to overfitting, and do not require much of hyperparameter tuning.
- One shortcoming of trees in general is that they cannot predict values outside the range of training data.

## Boosting and gradient boosting
- Boosting: adding models
- Gradient Boosting Machines utilize multiple underfitting models on top of each other, each feeding from residual data of previous underfitting model.
- GBMs overfit easily.
