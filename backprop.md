# Backprop

## Forward Propagation
```
Input:             x1                    x2

                 w1, w2,               w3, w4
Layer1:    z1 = w1*x1 + w2*x2    z2 = w3*x1 + w4*x2
            h1 = sigmoid(z1)      h2 = sigmoid(z2)
            
                 w5, w6,               w7, w8  
Layer2:    z3 = w5*h1 + w6*h2    z4 = w7*h1 + w8*h2
            o1 = sigmoid(z3)      o2 = sigmoid(z4)
```

Using MSE as loss function,

$E_{o1} = \frac{1}{2}(target_1 - output_1)^2$

$E_{o2} = \frac{1}{2}(target_2 - output_2)^2$

$E_{total} = E_{o1} + E_{o2}$

## Backward Propagation Step 1

Weights to update: `w5`, `w6`, `w7`, `w8`

To update `w5`, we need to get $\frac{\partial E_{total}}{\partial w_5}$

The rate of small change in total error per small change in weight is the whole idea behind gradient descent.

With chain rule, $\frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial o_1} \cdot \frac{\partial o_1}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_5}$

### First part:

Because $E_{total} = \frac{1}{2}(target_1 - o_1)^2 + \frac{1}{2}(target_2 - o_2)^2$

$\frac{\partial E_{total}}{\partial o_1} = 2 \cdot \frac{1}{2} (target_1 - o_1)^{2-1} \cdot (-1) + 0 = -(target_1 - o_1)$

### Second part:

Because $o_1 = sigmoid(z_3)$ 

and $\frac{d}{dx} sigmoid(x) = sigmoid(x)(1-sigmoid(x))$

$\frac{\partial o_1}{\partial z_3} = o_1 (1-o_1)$

As $sigmoid(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}$

$\frac{d}{dx}sigmoid(x) = \frac{e^x \cdot (1+e^x) - e^x \cdot e^x}{(1+e^x)^2} = \frac{e^x}{(1+e^x)^2} = \frac{e^x}{1+e^x} \cdot \frac{1+e^x-e^x}{1+e^x} = \frac{e^x}{1+e^x} \cdot (1-\frac{e^x}{1+e^x}) = sigmoid(x)(1-sigmoid(x))$

### Third part:

Since $z_3 = h_1 \cdot w_5$

$\frac{\partial z_3}{\partial w_5} = h_1$ (곱셈노드의 역전파는 순전파 때의 값을 서로 바꿔 곱해 하류로 흘려보냄)

### Weight update:

$\frac{\partial E_{total}}{\partial w_5} = -(target_1 - o_1) \cdot o_1 (1-o_1) \cdot h_1$

$w_5^+ = w_5 - lr \cdot \frac{\partial E_{total}}{\partial w_5}$

## Backward Propagation Step 2

Weights to update: `w1`, `w2`, `w3`, `w4`

To update `w1`, we need to get $\frac{\partial E_{total}}{\partial w_1}$

With chain rule, $\frac{\partial E_{total}}{\partial w_1} = \frac{\partial E_{total}}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$

### First part:

$\frac{E_{total}}{\partial h_1} = \frac{\partial E_{o1}}{\partial h_1} + \frac{\partial E_{o2}}{\partial h_1}$ ⭐

```
h1 -> z3 -> o1 -> Eo1
h1 -> z4 -> o2 -> Eo2
```

$\frac{\partial E_{o1}}{\partial h_1} = \frac{\partial E_{o1}}{\partial o_1} \cdot \frac{\partial o_1}{\partial z_3} \cdot \frac{\partial z_3}{\partial h_1} = -(target_1 - o_1) \cdot o_1 (1-o_1) \cdot w_5$

$\frac{\partial E_{o2}}{\partial h_1} = \frac{\partial E_{o2}}{\partial o_2} \cdot \frac{\partial o_2}{\partial z_4} \cdot \frac{\partial z_4}{\partial h_1} = -(target_2 - o_2) \cdot o_2 (1-o_2) \cdot w_7$

### Second part:

$h_1 = sigmoid(z_1)$

$\frac{\partial h_1}{\partial z_1} = h_1 (1-h_1)$

### Third part:

$z_1 = w_1 x_1$

$\frac{\partial z_1}{\partial w_1} = x_1$

### Weight update:

$\frac{\partial E_{total}}{\partial w_1} = (\frac{\partial E_{o1}}{\partial h_1} + \frac{\partial E_{o2}}{\partial h_1}) \cdot h_1 (1-h_1) \cdot x_1$

$w_1^+ = w_1 - lr \cdot \frac{\partial E_{total}}{\partial w_1}$

