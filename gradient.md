# [Gradient](https://github.com/star-bits/blog/blob/main/gradient.ipynb)

목표: $\frac{\partial L}{\partial w}$ 구하기

## Numerical gradient

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

$W$ 값을 $\pm h$한 상태로 $loss(x, t)$를 계산함 (= $\frac{\partial L}{\partial W}$ = `dW`) ⭐

$W, x \rightarrow z$

$z \rightarrow y$

$y, t \rightarrow loss$

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


def f(W): # ⭐
    return net.loss(x, t) # ⭐


dW = numerical_gradient(f, net.W)
print(dW)
```

```
[[-3.01115348  0.4640963  -1.85174813]
 [ 0.19180367  0.7046224  -1.30501329]]
3.3107699086628117
[[ 0.04199902  0.53610661 -0.57810563]
 [ 0.06299853  0.80415992 -0.86715845]]
```

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
y = \begin{cases}
        x & (x>0)\\
        0 & (x\leq0)
    \end{cases}
$$

$$
\frac{\partial y}{\partial x} = \begin{cases}
        1 & (x>0)\\
        0 & (x\leq0)
    \end{cases}
$$

Forward propagation:

$$
\begin{cases}
    x \rightarrow y=x & (x>0)\\
    x \rightarrow y=0 & (x\leq0)
\end{cases}
$$

Backward propagation:

$$
\begin{cases}
    \frac{\partial L}{\partial y} \leftarrow \frac{\partial L}{\partial y} & (x>0)\\
    0 \leftarrow \frac{\partial L}{\partial y} & (x\leq0)
\end{cases}
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

$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$, $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$, $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y}$ ⭐

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
        dx = np.dot(dout, self.W.T) # ⭐
        self.dW = np.dot(self.x.T, dout) # ⭐
        self.db = np.sum(dout, axis=0) # ⭐
        
        return dx
```

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
        print(f'{train_acc:.4f} {test_acc:.4f}')
```

```
0.1131 0.1176
0.9052 0.9060
0.9234 0.9252
0.9359 0.9382
0.9458 0.9465
0.9503 0.9481
0.9547 0.9528
0.9601 0.9571
0.9642 0.9590
0.9655 0.9612
0.9683 0.9634
0.9698 0.9630
0.9729 0.9660
0.9732 0.9668
0.9755 0.9668
0.9758 0.9686
0.9767 0.9689
```
