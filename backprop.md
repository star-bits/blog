# [Gradient](https://github.com/star-bits/blog/blob/main/backprop.ipynb)

목표: $\frac{\partial L}{\partial w}$ 구하기

## Numerical gradient

$w$ 값을 $\pm h$한 상태로 $loss(x, t)$를 계산함 (= $\frac{\partial L}{\partial w}$ = `dw`)

$w, x \rightarrow z$

$z \rightarrow y$

$y, t \rightarrow loss$

### Simple numerical gradient with independent variable w

$w$ 값을 $\pm h$한 상태로 $f(w)$를 계산함 (= $\frac{\partial f}{\partial w}$)

$w \rightarrow f$

```python
import numpy as np


def numerical_gradient(f, w):
    h = 1e-4
    grad = np.zeros_like(w)
    
    for i in range(w.size):
        tmp = w[i]
        w[i] = tmp+h
        f1 = f(w)
        
        w[i] = tmp-h
        f2 = f(w)
        
        grad[i] = (f1-f2)/(2*h)
        w[i] = tmp
        
    return grad
```

$f(w_0, w_1) = w_0^2 + w_1^2$ 

```python
def f(w):
    return w[0]**2 + w[1]**2

numerical_gradient(f, np.array([3.0, 4.0]))
```

```
array([6., 8.])
```

```python
def gradient_descent(f, w, lr=0.1, step_num=100):    
    for i in range(step_num):
        grad = numerical_gradient(f, w)
        w -= lr * grad
        
    return w

init_w = np.array([-3.0, 4.0])
gradient_descent(f, init_w)
```

```
array([-6.11110793e-10,  8.14814391e-10])
```

### Numerical gradient on neural network

```python
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
    
net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
```

```
[[ 0.20916827 -0.07467028 -0.85182626]
 [ 2.4889391   0.03834412  0.79656462]]
2.348940785732206
[[ 0.49656805  0.04614931 -0.54271735]
 [ 0.74485207  0.06922396 -0.81407603]]
```

```python
def f(W):
    return net.loss(x, t)
```

$W$ 값을 $\pm h$한 상태로 $loss(x, t)$를 계산함 (= $\frac{\partial L}{\partial W}$ = `dW`)

$W, x \rightarrow z$

$z \rightarrow y$

$y, t \rightarrow loss$

## Backprop

### Addition

$z = x + y$

$\frac{\partial z}{\partial x} = 1$, $\frac{\partial z}{\partial y} = 1$

Forward propagation:

$x, y \rightarrow z$

Backward propagation:

$\frac{\partial L}{\partial z} \cdot 1, \frac{\partial L}{\partial z} \cdot 1 \leftarrow \frac{\partial L}{\partial z}$

덧셈노드의 역전파는 상류의 값을 그대로 하류로 흘려보냄

### Multiplication

$z = xy$

$\frac{\partial z}{\partial x} = y$, $\frac{\partial z}{\partial y} = x$

Forward propagation:

$x, y \rightarrow z$

Backward propagation:

$\frac{\partial L}{\partial z} \cdot y, \frac{\partial L}{\partial z} \cdot x \leftarrow \frac{\partial L}{\partial z}$

곱셈노드의 역전파는 순전파 때의 값을 서로 바꿔 곱해 하류로 흘려보냄

### ReLU

$$
y = \left\{
    \begin{array}\\
        x & (x>0)\\
        0 & (x\leq0)
    \end{array}
\right.
$$

$$
\frac{\partial y}{\partial x} = \left\{
    \begin{array}\\
        1 & (x>0)\\
        0 & (x\leq0)
    \end{array}
\right.
$$

Forward propagation:

$$
\left\{
    \begin{array}\\
        x \rightarrow y=x & (x>0)\\
        x \rightarrow y=0 & (x\leq0)
    \end{array}
\right.
$$

Backward propagation:

$$
\left\{
    \begin{array}\\
        \frac{\partial L}{\partial y} \leftarrow \frac{\partial L}{\partial y} & (x>0)\\
        0 \leftarrow \frac{\partial L}{\partial y} & (x\leq0)
    \end{array}
\right.
$$

```python
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        # 0보다 작으면 mask = True
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
```

### Sigmoid

$y = \frac{1}{1 + \exp(-x)}$

$x \rightarrow -x \rightarrow \exp(-x) \rightarrow 1+\exp(-x) \rightarrow \frac{1}{1 + \exp(-x)}$

`/`: $y = \frac{1}{x}$, $\frac{\partial y}{\partial x} = -\frac{1}{x^2} = -y^2$

`+`: 상류의 값을 그대로 하류로 흘려보냄

`exp`: $y = \exp(x)$, $\frac{\partial y}{\partial x} = \exp(x)$

`*`: 순전파 때의 값을 서로 바꿔 곱해 하류로 흘려보냄

$\frac{\partial L}{\partial y} y^2 \exp(-x) \leftarrow -\frac{\partial L}{\partial y} y^2 \exp(-x) \leftarrow -\frac{\partial L}{\partial y} y^2 \leftarrow -\frac{\partial L}{\partial y} y^2 \leftarrow \frac{\partial L}{\partial y}$

$\frac{\partial L}{\partial y} y^2 \exp(-x) = \frac{\partial L}{\partial y}\frac{1}{(1+\exp(-x))^2}\exp(-x) = \frac{\partial L}{\partial y}\frac{1}{1+\exp(-x)}\frac{\exp(-x)}{1 + \exp(-x)} = \frac{\partial L}{\partial y}y(1-y)$

Forward propagation:

$x \rightarrow y$

Backward propagation:

$\frac{\partial L}{\partial y} y (1-y) \leftarrow \frac{\partial L}{\partial y}$

```python
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx
```

### Affine

$X \cdot W + B = Y$

$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$, $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$, $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y}$

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
```

```python
dx = np.dot(dout, self.W.T)
self.dW = np.dot(self.x.T, dout)
self.db = np.sum(dout, axis=0)
```

$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$, $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$, $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y}$

### Softmax-with-Loss

Softmax의 손실 함수로 Cross Entropy Error를 사용하면 역전파가 $(y_1-t, y_2-t, y_3-t)$로 말끔히 떨어짐

$L = -(t \log(y) + (1-t) \log(1-y))$

$\frac{\partial L}{\partial y} = - \frac{t}{y} + \frac{1-t}{1-y}$

$y = \frac{1}{1+\exp(-x)}$

$\frac{\partial y}{\partial x} = y(1-y)$

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \left( - \frac{t}{y} + \frac{1-t}{1-y} \right) \cdot y(1-y) = y-t$

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = np.exp(x) / np.sum(np.exp(x))
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
```

### Complete neural network

```python
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
```

```
W1:4.091033022270988e-10
b1:2.37278236850109e-09
W2:5.324391306385561e-09
b2:1.395024796846389e-07
```

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

```
0.0876 0.0881
0.90345 0.9079
0.92415 0.9273
0.9356 0.9371
0.9450666666666667 0.9438
0.9525 0.9511
0.9568833333333333 0.955
0.9604666666666667 0.9562
0.9644666666666667 0.9616
0.9675833333333334 0.9618
0.97025 0.9663
0.9724166666666667 0.9662
0.9732166666666666 0.9665
0.9735666666666667 0.9653
0.97645 0.9696
0.9777666666666667 0.9692
0.97865 0.9705
```
