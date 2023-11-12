# Notes from [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) and [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)

## Overfitting: how to prevent
- 데이터 양을 늘이기
- 모델의 복잡도 줄이기
- Regularization 적용하기 (loss를 최소화하기 위해 weight들이 작아짐)
  - L1 norm: weight들의 절대값 합계를 비용함수에 추가 
    - 어떤 feature들이 모델에 영향을 주고 있는지 판단하는데 유용 (weight들이 0이 되니까)
  - L2 norm: weight들의 제곱값 합계를 비용함수에 추가 
    - 과적합을 막는데 유용 (also called weight decay)
- Dropout 적용하기

## Vanishing gradient
- 시그모이드 함수의 출력값이 0과 1에 가까워질수록 x축에 평행하게 됨.
- 따라서 이 구간에서의 미분값은 0에 가까워짐.
- 역전파 과정에서 0에 가까운 값이 누적해서 곱해지면서 입력층에 가까워질수록 미분값이 잘 전달되지 않게 됨.
- 은닉층에서는 시그모이드 함수를 사용하지 말아야.

## Batch Normalization
- Internal Covariate Shift
  - 학습 과정에서 layer별로 입력 데이터의 분포가 다른 현상
  - Batch normalization은 이 internal covariate shift를 줄임. (Internal covariate shift를 줄이기 때문에 batch normalization의 효과가 좋은 것인지는 불분명.)
  - Covariate shift는 internal covariate shift와 다른 현상이며, 훈련 데이터의 분포와 테스트 데이터의 분포가 다른 것을 가리킴.
- 활성화 함수를 통과하기 전에 수행됨. (e.g. Conv->BN->ReLU)
- 큰 learning rate를 사용할 수 있어 학습 속도가 개선됨.
- Vanishing gradient 문제를 개선시킴.
- 과적합을 방지하는 효과가 있음.
- 가중치 초기화에 덜 민감해짐.
- 모델이 더 복잡해지기 때문에 inference 시 실행 시간이 느려짐.

## [NNLM](https://arxiv.org/pdf/1301.3781.pdf)
$\text{Input layer} \rightarrow (W_{v \times m}) \rightarrow \text{Projection layer} \rightarrow (W_{n \times m \times h}) \rightarrow \text{Hidden layer} \rightarrow (W_{h \times v}) \rightarrow \text{Output layer}$

- Input layer
  - n one-hot vectors of size v
- $W_{v \times m}$
  - v to m transformation
- Projection layer
  - n embedding vectors of size m
- $W_{n \times m \times h}$
  - n by m to h transformation
- Hidden layer
  - Nonlinear layer
  - A vector of size h
- $W_{h \times v}$
  - h to v transformation
- Output layer
  - A vector of size v

단점: n-gram 모델과 마찬가지로 한 번에 n개의 단어만을 참고할 수 있음.

Computational complexity: $n \times m + n \times m \times h + h \times v$ (첫번째 연산의 경우 vector of size m을 n 번 lookup table에서 불러옴. 두번째 연산의 경우 n 개의 vector of size m을 vector of size h로 transform하는 연산임. 세번째 연산의 경우 vector of size h 하나를 vector of size v로 transform하는 연산임.)

## Word2Vec(CBOW)

$\text{Input layer} \rightarrow (W_{v \times m}) \rightarrow \text{Projection layer} \rightarrow (W_{m \times v}^{'}) \rightarrow \text{Output layer}$

- Input layer
  - n one-hot vectors of size v
- $W_{v \times m}$
  - v to m transformation
- Projection layer
  - n embedding vectors of size m
- $W_{m \times v}^{'}$
  - This is not a transpose of $W_{v \times m}$. 서로 다른 행렬임.
  - n embedding vectors are averaged into one vector of size m.
  - m to v transformation
- Output layer
  - A vector of size v
  - 이후 loss function으로 cross-entropy를 사용함. 

$W_{v \times m}$ or $W_{m \times v}^{'}$ or 둘의 평균을 임베딩 벡터로 사용함.

Computational complexity: $n \times m + m \times v$ (첫번째 연산의 경우 vector of size m을 n 번 lookup table에서 불러옴. 두번째 연산의 경우 vector of size m 하나를 vector of size v로 transform하는 연산임.)

## NNLM vs. Word2Vec(CBOW)
- NNLM의 목적은 다음 단어 예측 (이전 단어들을 이용하여 학습)
- Word2Vec의 목적은 워드 임베딩 생성 (이전과 이후 단어들을 이용하여 학습)

## CBOW, Skip-gram, SGNS
- Continuous Bag-of-Words: 주변 단어를 통해 중심 단어를 예측함. (전체 단어 집합을 두고 이진 분류를 수행함.)
- Continuous Skip-gram: 중심 단어를 통해 주변 단어를 예측함. (전체 단어 집합을 두고 이진 분류를 수행함.)
- Continuous Skip-gram with Negative Sampling: 전체 단어 집합이 아니라 일부 단어 집합만을 샘플링해 negative로, 주변 단어들을 positive로 두고 이진 분류를 수행함.

## Naive Bayes Binary Classification

- $P(A|B)$: B가 일어나고 나서 A가 일어날 확률
- $P(B|A)$: A가 일어나고 나서 B가 일어날 확률

$P(B|A)$를 쉽게 구할 수 있는 상황이라면, 다음 식을 통해 $P(A|B)$를 구할 수 있음: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

입력 텍스트가 주어졌을 때 정상 메일인지 스팸 메일인지 구분하는 확률:
- $P(\text{정상 메일}|\text{입력 텍스트}) = \text{입력 텍스트가 정상 메일일 확률} = \frac{P(\text{입력 텍스트}|\text{정상 메일})P(\text{정상 메일})}{P(\text{입력 텍스트})}$
- $P(\text{스팸 메일}|\text{입력 텍스트}) = \text{입력 텍스트가 스팸 메일일 확률} = \frac{P(\text{입력 텍스트}|\text{스팸 메일})P(\text{스팸 메일})}{P(\text{입력 텍스트})}$

공통 분모 $P(\text{입력 텍스트})$ 를 제거하면,
- $P(\text{정상 메일}|\text{입력 텍스트}) = P(\text{입력 텍스트}|\text{정상 메일})P(\text{정상 메일})$
- $P(\text{스팸 메일}|\text{입력 텍스트}) = P(\text{입력 텍스트}|\text{스팸 메일})P(\text{스팸 메일})$

메일의 본문에 있는 단어가 3개이고 모든 단어가 독립적이라고 가정하면,
- $P(\text{정상 메일}|\text{입력 텍스트}) = P(w_1|\text{정상 메일})P(w_2|\text{정상 메일})P(w_3|\text{정상 메일})P(\text{정상 메일})$
  - $P(w_1|\text{정상 메일}) = \frac{\text{정상 메일 훈련 데이터에서 }w_1\text{이 등장한 횟수}}{\text{정상 메일 훈련 데이터에 등장한 모든 단어의 등장 횟수의 총합}}$
  - $P(\text{정상 메일}) = \frac{\text{정상 메일 훈련 데이터의 갯수}}{\text{전체 메일 훈련 데이터의 갯수}}$ 
- $P(\text{스팸 메일}|\text{입력 텍스트}) = P(w_1|\text{스팸 메일})P(w_2|\text{스팸 메일})P(w_3|\text{스팸 메일})P(\text{스팸 메일})$

주의: 정상 메일 훈련 데이터 중 $w_1$이 0번 등장했다면 $P(\text{정상 메일}|\text{입력 텍스트})$은 무조건 0이 됨.

## 1D CNN Classification

- The input is a matrix of size n by m, where n represents the number of tokens in a sentence, and m represents the dimensionality of each token's embedding vector.
- Kernels are k by m matrices that slide over the input matrix, performing convolution at each step.
- Convolution with each kernel produces a 1-dimensional vector of length n-k+1.
- Max pooling is applied to each vector to select the maximum value in that vector.
- The resulting maximum values are concatenated to form a single layer.
- A fully connected layer is applied to the concatenated layer for classification.
