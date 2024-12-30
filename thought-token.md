# Latent Thought-Token CoT
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/pdf/2412.06769)

## Reasoning in Latent Space
- discrete한 language space가 아닌 continuous한 latent space에서 reasoning을 하겠다는 아이디어.
- 인간이 생각을 할 때 뇌의 language network는 대부분 비활성화되어 있다고 함. 직관적으로도, 생각은 언어로만 하는게 아닐 것. (인간의 언어는 reasoning보다 communication에 최적화되어 있다고 함. 하긴 그렇겠지.)
- language space에서는 computation이 reasoning이 아니라 textual coherence를 위해 사용된다.
- last hidden state를 reasoning state로 생각/이용 하자.
- 이 last hidden state를 word token으로 (기존에 하듯이) decoding하는 것이 아니라 다시 input embedding으로 모델에 넣어줌.
- "[I]nstead of mapping between hidden states and language tokens using the language model head and embeddng layer, COCONUT directly feeds the last hidden state as the input embedding for the next token."
- 이렇게 하면 이 reasoning state는 multiple alternative reasoning steps를 encode 할 수 있다고 함. 마치 BFS를 하는 효과라고.

## 이전의 유사 방법론들
- prompting LLMs to generate succinct reasoning chains (Madaan and Yazdanbakhsh, 2022).
- performing additional reasoning before generating some critical tokens (Zelikman et al., 2024).
- training LLMs to generate reasoning chains with supervised finetuning (Yue et al., 2023; Yu et al., 2023).
- training LLMs to generate reasoning chains with reinforcement learning (Wang et al., 2024; Havrilla et al., 2024; Shao et al., 2024; Yu et al., 2024a).
- classifying the tokens in CoT into symbols, patterns, and text and guide the LLM to generate concise CoT based on analysis of their roles (Madaan and Yazdanbakhsh, 2022).
- by employing GoT, the effective depth of the transformer increases because the generated outputs are looped back into the input (Feng et al., 2023).
- autoregressive generation nature of CoT makes it challenging to mimic human reasoning on more complex problems which typically require planning and search (LeCun, 2022; Hao et al., 2023).
- pretraining the model by randomly inserting a learnable <pause> token to the training corpus (Goyal et al., 2023).
- iCoT (Deng et al., 2024).

## 그래서 어떻게 만들었다는 걸까
![scrsht 2024-12-29 at 20 19 18](https://github.com/user-attachments/assets/338e2de1-70eb-4aff-a030-cb59ca4fe951)
- 우리가 원하는 건 n개의 `[Thought]` 토큰을 통해 생각을 하고, 마지막에 답을 내는 것.
- training data에 있는 자연어 CoT step들을 stage마다 thought token으로 바꿔감.
- 이 과정을 만들어 내기 위해서 각 stage마다 수 epoch씩 train을 함.
- 이러한 multi-stage training strategy는 inspired by Deng et al. (2024).
- 사용한 데이터셋: GSM8K (Cobbe et al., 2021) for math reasoning, ProntoQA (Saparov and He, 2022) and ProsQA (직접 만든 ProntoQA의 변형) for logical reasoning.
- word token으로 하는 CoT보다 적은 토큰으로 reasoning task를 수행할 수 있다.

## 생각
- latent space에서 reasoning을 하겠다는 모티베이션은 좋은데, 단순히 training data의 word token으로 이루어진 CoT를 thought token으로 바꿔준 것 이상의 의미를 찾기 어려움.
- 생각이 벡터여야 한다는 측면에서 기존 word token space 상에서의 CoT보다는 발전된 방식이지만 training 방식이 reasoning의 breakthrough가 되기엔 너무 지저분함.
- 생각을 latent space 상에서 하기 때문에 여러 언어에 잘 대응할 수 있다고 하던데, 그냥 출력된 언어를 다른 언어로 번역하되 그 과정에서 SONAR space를 이용한 것과 다른 의미가 있나?
- 정말로 생각은 latent space상에서 하고, 벡터가 생각의 흐름에 따라 변형된 뒤 그 생각을 마지막에 word token으로 decoding 하는 방식이 필요해 보임.
- 마치 stable diffusion처럼 latent space상에서 생각을 여러 방향으로 steer하고, 그 적절한 steering direction을 RL로 찾아낼 수 있지 않을까.
- 일단 steering에 필요한 인위적인 사고 흐름을 몇 개 생각해 볼 수 있음: `branching ('on second thought...')`, `backtracking ('wait, but...')`, `continuing ('therefore...')`. `encapsulating ('in summary...')`
