# Latent Space Reasoning
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/pdf/2412.06769)

## Reasoning in Latent Space
- language state가 아닌 latent state에서 reasoning을 하겠다는 아이디어.
- 인간이 생각을 할 때 뇌의 language network는 대부분 비활성화되어 있다. 생각은 언어로(만) 하는게 아닐 것.
- last hidden state를 reasoning state로 생각/이용 하자.
- 이 last hidden state를 word token으로 decoding하는 것이 아니라 다시 input embedding으로 모델에 넣어줌.
- 이렇게 하면 이 reasoning state는 multiple alternative reasoning steps를 encode 할 수 있다고 함. 마치 BFS를 하는 효과 (라고 함).

## 그래서 어떻게 만들었다는 걸까
![scrsht 2024-12-29 at 20 19 18](https://github.com/user-attachments/assets/338e2de1-70eb-4aff-a030-cb59ca4fe951)
- 우리가 원하는 건 n개의 `[Thought]` 토큰을 통해 생각을 하고, 마지막에 답을 내는 것.
- training data에 있는 자연어 CoT step들을 stage마다 thought token으로 바꿔감.
- 이 과정을 만들어 내기 위해서 각 stage마다 수 epoch씩 train을 함.
- 사용한 데이터셋: GSM8K, ProntoQA, ProsQA (직접 만든 ProntoQA의 변형)

## 생각
- latent space에서 reasoning을 하겠다는 모티베이션은 좋은데, 단순히 training data의 word token으로 이루어진 CoT를 thought token으로 바꿔준 것 이상의 의미를 찾기 어려움.
- 생각이 벡터여야 한다는 측면에서 기존 word token space 상에서의 CoT보다는 발전된 방식이지만 training 방식이 reasoning의 breakthrough가 되기엔 너무 지저분함.
- 정말로 생각은 latent space상에서 하고, 벡터가 생각의 흐름에 따라 변형된 뒤 그 생각을 마지막에 word token으로 decoding 하는 방식이 필요해 보임.
- 마치 stable diffusion처럼 latent space상에서 생각을 여러 방향으로 steer하고, 그 적절한 steering direction을 RL로 찾아낼 수 있지 않을까.
- 일단 인위적인 사고 흐름을 몇 개 생각해 볼 수 있음: `branching ('on second thought...')`, `backtracking ('wait, but...')`, `continuing ('therefore...')`. `encapsulating ('in summary...')`
