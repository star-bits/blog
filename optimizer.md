# Optimizer

- GD: 전체 train set에 대해 업데이트
- SGD: mini-batch에 대해 업데이트
- SGD with momentum: 과거의 gradient가 현재의 업데이트에 영향
- AdaGrad: 각 dimension에서 업데이트가 동일 비율로 일어나게 조절
- RMSprop: AdaGrad의 속도조절기가 시간이 지날수록 decay하게
- Adam: momentum + RMSprop

## SGD

$w_{t+1} = w_t - \alpha \nabla L(w)$

## SGD with momentum

$v_{t+1} = \beta_1 v_t + (1-\beta_1) \nabla L(w)$\
$w_{t+1} = w_t - \alpha v_{t+1}$

Now past gradients impact current updates. Momentum $v$ builds up over time.

## AdaGrad

$s_{t+1} = s_t + \nabla L(w)^2$\
$w_{t+1} = w_t - \alpha \frac{\nabla L(w)}{\sqrt{s_{t+1}} + 1e^{-5}}$

You keep the running sum of squared gradients, $s_{t+1}$, during update. Consider a loss function in a multi dimension space. In axes with small gradient values, the running sum of squared gradients become even smaller -- so let's divide the whole thing with it. The division will give us a bigger number.

Small steps become big steps, and big steps become small steps. We force the algorithm to make updates in any direction with the same properties.

Problem: learning becomes very slow.

## RMSprop

$s_{t+1} = \beta_2 s_t + (1-\beta_2) \nabla L(w)^2$\
$w_{t+1} = w_t - \alpha \frac{\nabla L(w)}{\sqrt{s_{t+1}} + 1e^{-5}}$

RMSprop still keeps the running sum of squared gradients. But instead of letting that sum grow continuously, we let it decay.

## Adam

$v_0 = 0, s_0 = 0$\
$v_{t+1} = \beta_1 v_t + (1-\beta_1) \nabla L(w) \quad (momentum)$\
$s_{t+1} = \beta_2 s_t + (1-\beta_2) \nabla L(w)^2 \quad (RMSprop)$\
$w_{t+1} = w_t - \alpha \frac{v_{t+1}}{\sqrt{s_{t+1}} + 1e^{-5}}$

$v_0 = 0, s_0 = 0$\
$v_{t+1} = \beta_1 v_t + (1-\beta_1) \nabla L(w) \quad (momentum)$\
$s_{t+1} = \beta_2 s_t + (1-\beta_2) \nabla L(w)^2 \quad (RMSprop)$\
$v_{t+1} = \frac{v_{t+1}}{1-\beta_1}$\
$s_{t+1} = \frac{s_{t+1}}{1-\beta_2}$\
$w_{t+1} = w_t - \alpha \frac{v_{t+1}}{\sqrt{s_{t+1}} + 1e^{-5}}$

$(\beta_1 = 0.9, \beta_2 = 0.999, \alpha = 0.001 \text{--} 0.0001)$
