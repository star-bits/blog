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

## C2-1: Neural Networks

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
model.fit(X_train, y_train, epochs=10)
```
```python
W1, b1 = model.get_layer("layer1").get_weights()
model.get_layer("layer1").set_weights([W1, b1])
```

## C2-2: Decision Trees

### Decision trees
- Which feature to split upon in each node?
  - Maximum purity = More order = Biggest amount of reduction in entropy = Highest information gain
- $H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$
- $\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right)$
- Regression trees: Reduction in variance instead of reduction in entropy
- Random forest: Sampling with replacement
- Boosted trees: Insead of picking all examples with equal probability, make it more likely to pick misclassified examples from previously trained trees.

### DecisionTreeClassifier from scikit-learn
- Low `min_samples_split` (the minimum number of samples required to split an node) leads to overfitting
- High `max_depth` leads to overfitting
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = RANDOM_STATE).fit(X_train, y_train)

print(accuracy_score(decision_tree_model.predict(X_train), y_train)) # 0.8583
print(accuracy_score(decision_tree_model.predict(X_val), y_val)) # 0.8641
```

### RandomForestClassifier from scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators = 100,
                                             min_samples_split = 10,
                                             max_depth = 16,
                                             random_state = RANDOM_STATE).fit(X_train, y_train)

print(accuracy_score(random_forest_model.predict(X_train), y_train)) # 0.9292
print(accuracy_score(random_forest_model.predict(X_val), y_val)) # 0.8967
```

### Gradient Boosting model XGBClassifier from XGBoost
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1, verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit, y_train_fit, eval_set = [(X_train_eval, y_train_eval)], early_stopping_rounds = 10)
# model was allowed up to 500 estimators, but the algorithm only fit 26 estimators

print(accuracy_score(xgb_model.predict(X_train), y_train)) # 0.9251
print(accuracy_score(xgb_model.predict(X_val), y_val)) # 0.8641
```

## C3-1: Unsupervised learning 

### K-means clustering
- Random initialization -> Iterative process of (Compute distances to datapoints -> Move centroids)
- Cost function: MSE
- Number of random initializations: 50-1000 times
- Number of clusters K: Huristic

### Anomaly detection
- Gaussian distribution and pre-assigned threshold

### Principal Component Analysis
- Reduces the number of features to 2-3
- Mainly used for data visualization
- Looks for the axis with maximum variance

```python
from sklearn.decomposition import PCA

X = np.array([[ 99,  -1],
              [ 98,  -1],
              [ 97,  -2],
              [101,   1],
              [102,   1],
              [103,   2]])

pca = PCA(n_components=1)
pca.fit(X)
pca.explained_variance_ratio_ # returns a list of the amount of variance explained by each principal component
```
```
array([0.99244289])
```
The coordinates on the first "principal component" (first axis) retain 99.24% of the information (explained variance).

```python
pca = PCA(n_components=2)
pca.fit(df) # df is a dataframe with features as columns and items as rows
X_pca = pca.transform(df)
df_pca = pd.DataFrame(X_pca, columns = ['principal_component_1', 'principal_component_2'])

sum(pca.explained_variance_ratio_) # 0.14572843555106277

plt.scatter(df_pca['principal_component_1'], df_pca['principal_component_2'], color = "#C00000")
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA decomposition')
plt.show()
```
```python
pca = PCA(n_components=3)
pca.fit(df)
X_pca = pca.transform(df)
df_pca = pd.DataFrame(X_pca, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])

fig = px.scatter_3d(df_pca, x = 'principal_component_1',
                            y = 'principal_component_2',
                            z = 'principal_component_3').update_traces(marker = dict(color = "#C00000"))
fig.show()
```

## C3-2: Recommender systems

### Collaborative filtering
- For each user, parameter vector $w^{user}$: embodies a movie taste of the user
- For each movie, feature vector $x^{movie}$: embodies some description of the movie
- The dot product of the two vectors plus the bias term produces an estimate of the rating the user might give to the movie.
- Cost function:
  - $J({\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)},\mathbf{w}^{(0)},b^{(0)},...,\mathbf{w}^{(n_u-1)},b^{(n_u-1)}}) =$
  - $\left[ \frac{1}{2}\sum\limits_{(i,j):r(i,j)=1}(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2 \right]$
  - $+\left[ \frac{\lambda}{2} \sum\limits_{j=0}^{n_u-1} \sum\limits_{k=0}^{n-1} (\mathbf{w}^{(j)}_k)^2 \right]$: regularization term 1
  - $+\left[ \frac{\lambda}{2} \sum\limits_{i=0}^{n_m-1} \sum\limits_{k=0}^{n-1} (\mathbf{x}_k^{(i)})^2 \right]$: regularization term 2

```python
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = R * (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J
```
```python
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func(X, W, b, Y_norm, R, lambda_)

    grads = tape.gradient( cost_value, [X,W,b] )
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
```

### Content-based filtering
- For each user, user vector $x_{user}^{(j)}$
- For each movie, movie vector $x_{movie}^{(i)}$
- User network: $X_{user} \to V_{user}$
- Movie network: $X_{movie} \to V_{movie}$
- $g(v_{user}^{(j)} \cdot v_{movie}^{(i)})$ to predict

```python
num_outputs = 32

user_NN = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),
])
item_NN = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),
])

input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

output = tf.keras.layers.Dot(axes=1)([vu, vm])

model = tf.keras.Model([input_user, input_item], output)

cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
```

## C3-3: Reinforcement learning


$Q_{i+1}(s,a) = R + \gamma \max_{a'}Q_i(s',a')$
