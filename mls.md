# Machine Learning Specialization (Dec 2023)

## C1: Linear regression (regression) and Logistic regression (classification)

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
- $b = b - \alpha \frac{\partial J(w,b)}{\partial b}$
- $\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$
- $\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$
     

### Classification (Logistic regression)
- Why not use linear regression for classification? The outlier problem. An outlier can shift the decision boundary by a lot.
- Why not use squared error cost function for logistic regression? The non-linear nature of the model results in a "wiggly", non-convex cost function with many local minima. (Sigmoid cost function gives convex cost function. The proof of it is out of scope.)
   
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



\begin{equation}
  L(\mathbf{a},y)=\begin{cases}
    -log(a_1), & \text{if $y=1$}.\\
        &\vdots\\
     -log(a_N), & \text{if $y=N$}
  \end{cases} \tag{3}
\end{equation}




## C2-2: Decision Trees and XGBoost

## C3: 
