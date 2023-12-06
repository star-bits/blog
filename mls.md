# Machine Learning Specialization (Dec 2023)

## C1: Regression (Linear regression) and Classification (Logistic regression)

### Overview
- Supervised learning
  - Regression (Linear regression)
    - Univariate linear regression
      - $f_{w,b}(x^{(i)}) = wx^{(i)} + b$
      - $J(w,b) = \frac{1}{m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
      - $w = w - \alpha \frac{\partial J(w,b)}{\partial w}$
      - $b = b - \alpha \frac{\partial J(w,b)}{\partial b}$
      - $\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$
      - $\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})$
    - Multiple variable linear regression
    - Polynomial regression
  - Classification (Logistic regression)
    - Why not use linear regression for classification? The outlier problem. An outlier can shift the decision boundary by a lot.
    - Why not use squared error cost function for logistic regression? The non-linear nature of the model results in a "wiggly", non-convex cost function with many local minima. (Sigmoid cost function gives convex cost function. The proof of it is out of scope.)
    - Binary classification
      - $f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$
      - $L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \quad \text{if } y^{(i)}=1$
        - $\text{as } f(x) \to 0, \text{loss} \to \infty$
        - $\text{as } f(x) \to 1, \text{loss} \to 0$
        - $y=-\log(x)$ is $y=\log(x)$ reflected across the x-axis.
      - $L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \quad \text{if } y^{(i)}=0$
        - $\text{as } f(x) \to 0, \text{loss} \to 0$
        - $\text{as } f(x) \to 1, \text{loss} \to \infty$
        - $y=-\log(1-x)$ is $y=-\log(x)$ refected across the y-axis and then translated along the x-axis by +1.

- Unsupervised learning
  - Clustering
  - Anomaly detection
  - Dimensionality reduction


## C2: 

## C3: 
