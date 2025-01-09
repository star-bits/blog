# Latent Sentence (Concept) Model
- [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/pdf/2412.08821)

## Next Token Prediction 대신 Next 'Concept' Prediction을 하자
- "[We] assume that a concept corresponds to a sentence." 데이터를 문장 단위로 자름. average sentence length is assumed to be 10-20 tokens. 너무 길면 여러 개로 나눔. sentence segmentation 방법은 SpaCy segmentor(rule-based nlp toolkit)와 Sagment any Text (SaT. predict sentence boundaries at the token level.) 두 가지가 있음.
- 그리고 각 문장을 SONAR space(pretrained sentence embedding space임)으로 변환함.
- 이후 1) decoder-only transformer로 next 'concept' prediction을 하거나, 2) diffusion-based generation을 하거나, 3) FAISS clustering을 통해 SONAR space를 quantize해서 

![fig1](https://github.com/user-attachments/assets/9a6706dc-e5f0-4c66-998f-c441687d72c3)

## 1) 디코더로 next sentence prediction
- 역사와 전통의 minimize MSE loss하는 방법
- 근데 directly minimizing the MSE loss in the embedding space does not yield good results.

## 2) Diffusion
- diffusion 방법론은 aiming to learn conditional probability distributions over continuous data.
- 무슨 말이냐 - many different images may satisfy the same input prompt, hence the model has to learn a probability distribution over continuous pixel data. 
- sentences in the SONAR space, despite being represented as continuous vectors, remain discrete combinatorial objects. this makes diffusion modeling struggle on the text modality.
- contrastive nature of cross-entropy loss based on softmax outputs which is used for next token prediction plays a critical role for many downstream task where higher accuracy is requred. on the opposite, continuous diffusion modeling does not allow to integrate such a contrastive objective.

## 3) space quantization
- SONAR representations을 discretize 한다고 함.
- residual vector quantization of the SONAR space. quantized large concept model based on these discrete units.
- vector quantization maps continuous input embeddings to the nearest entry in a learnt codebook.
- RVQ iteratively quantize residual errors from previous quantizations using additional codebook for each iteration.
- the text modality remains discrete, and despite dealing with continuous representations in the SONAR space, all possible text sentences are a cloud of points rather than a real continous distribution in the SONAR space.

## related works
- Marfurt and Hendersen (2021) and Cornille et al. (2024) used predicted next sentence embeddings in a fully generative setting, for summarization and generic language modelling, respectively. considered sentence-level connections only as an addition to the token-level connections across sentences, not as their replacement.
- PLANNER architecture (Zhang et al., 2023) consists of a variational autoencoder for paragraphs and a diffusion model trained to predict latent autoencoder representations conditional on the textual context or on the class label.
- Lovalace et al. (2024) augmented a decoder-only language model with an encoded sematic proposal of the continuation text, with an easily guidable diffusion model predicting the embedding of the next proposal.
- TEncDM model (Shabablin et al., 2024) performs diffusion in the space of contextual token embeddings which are then decoded non-autoregressively.
- Semformer (Yin et al., 2024) proposed training transformers language models to plan several steps ahead by including special planning tokens, the representations of which are trained to be informative about the future toekns.
- Ye et al. (2024) applied discrete diffusion to language models as an alternative to autoregressive generation, more suitable for tasks that require multi-step planning
- Ubukata et al. (2024) give an overview of applications of diffusion for planning tasks.

## 생각
- token도 이미 큰 baggage/overhead인데 거기다 sentence까지 더하자고?
- 채팅, 코드, 수식 등의 다양한 포맷의 데이터에서는 sentence를 어떻게 자르나? sentence segmentation이 arbitrary 하기 때문에 너무 지저분해 질 것 같음.
- one "concept" per sentence도 너무 불안정함. 하나의 coherent한 thought을 문장으로 구분할 수 있을까. 
- 게다가 separately trained됨. 
- continuous end-to-end stream of vector representation으로 깔끔하게 latent reasoning을 했으면 좋겠다.
- next sentence prediction이 아니라 생각 전체를 다 하고 이후 일필휘지로 갈기는 게 더 직관적. 생각을 마치고.
- 그 생각의 흐름은 diffusion으로 하는거. diffusion 활용은 굉장히 직관적으로 합리적이고 매력적.
- thought의 latent diffusion 상에서 생각이 얼마나 변하는지 제어한다면 연속적인 생각들 간의 information density를 uniform하게 조정할 수 있지 않을까.
- latent diffusion 상에서 변화하는 생각의 방향을 decoupled cross attention으로 넣어줄 수 있나? 근데 그러면 각 생각의 방향을 symbolic하게 인위적으로 구분해서 넣어줘야 할 것. 깔끔하진 않네.
- diffusion models seems to be able to extrapolate, whereas llms seems to only capable of interpolation.
