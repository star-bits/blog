# Machine Learning Specialization (Dec 2023)

## C1: Linear regression (Regression) and Logistic regression (Classification)

### Overview
- Supervised learning
  - Regression
    - Univariate linear regression
    - Multi-variable linear regression
      - Polynomial regression
  - Classification
    - Binary classification
    - Multi-class classification
- Unsupervised learning
  - Clustering
  - Anomaly detection
  - Dimensionality reduction
 
### Univariate linear regression
- $f_{w,b}(x^{(i)}) = wx^{(i)} + b$
- $J(w,b) = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
- $w = w - \alpha \frac{\partial J(w,b)}{\partial w}$
  - $\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$
- $b = b - \alpha \frac{\partial J(w,b)}{\partial b}$
  - $\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$
     
### Classification (Logistic regression)
- Why not use linear regression for classification?
  - The outlier problem. An outlier can shift the decision boundary by a lot.
- Why not use squared error cost function for logistic regression?
  - The non-linear nature of the model results in a "wiggly", non-convex cost function with many local minima. (Sigmoid cost function gives convex cost function. The proof of it is out of scope.)
   
### Binary classification
- $f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$
- $L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \quad \text{if } y^{(i)}=1$
  - $\text{as } f(x) \to 0, \text{loss} \to \infty$
  - $\text{as } f(x) \to 1, \text{loss} \to 0$
  - $y=-\log(x)$ is $y=\log(x)$ reflected across the x-axis.
- $L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \quad \text{if } y^{(i)}=0$
  - $\text{as } f(x) \to 0, \text{loss} \to 0$
  - $\text{as } f(x) \to 1, \text{loss} \to \infty$
  - $y=-\log(1-x)$ is $y=-\log(x)$ refected across the y-axis and then translated along the x-axis by +1.
- $L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -\left(y^{(i)}\right) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$
- $J(\mathbf{w},b) = \frac{1}{m} \sum\limits_{i=1}^{m} \left[ L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] = -\frac{1}{m} \sum\limits_{i=1}^{m} \left[ \left(y^{(i)}\right) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) + \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right]$
- $w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j}$
  - $\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}$
- $b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}$
  - $\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})$

## C2-1: Neural Networks and TensorFlow

### Activation function
- Why is ReLU better than sigmoid?
  - When you have a function that is flat on a lot of places, the cost function graph too will have a lot of flat places, making gradient descent to be slower.

### Softmax regression (as opposed to Logistic regression)
- $a_i = \frac{e^{z_i}}{\sum\limits_{k=1}^{N}{e^{z_k} }}$
- $L(\mathbf{a},y) = -\log(a_1), \quad \text{if } y=1$
- $L(\mathbf{a},y) = -\log(a_N), \quad \text{if } y=N$

### Neural Network with TensorFlow
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
```
```python
model = Sequential(
    [
        Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'), # 'relu', 'linear'
        Dense(1, activation='sigmoid', name = 'layer2')
    ]
)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(), # CategoricalCrossentropy(), SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
```
```python
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 layer1 (Dense)              (None, 3)                 9         
                                                                 
 layer2 (Dense)              (None, 1)                 4         
                                                                 
=================================================================
Total params: 13 (52.00 Byte)
Trainable params: 13 (52.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
```python
model.fit(X_train, Y_train, epochs=10)
```
```python
W1, b1 = model.get_layer("layer1").get_weights()
```
```python
model.get_layer("layer1").set_weights([W1, b1])
```

## C2-2: Decision Trees and XGBoost

### Decision tree
- Which feature to split upon in each node?
  - Maximum purity = Highest information gain = Biggest amount of reduction in entropy = More order
- $H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$
- $\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right)$
- Regression tree: reduction in variance instead of reduction in entropy
- Random forest
- Boosted trees

## C3: 
