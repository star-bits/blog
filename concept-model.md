# Latent Sentence (Concept) Model
- [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/pdf/2412.08821)

## Next Token Prediction 대신 Next 'Concept' Prediction을 하자
- "[We] assume that a concept corresponds to a sentence."
- 데이터를 문장 단위로 자름.
- 그리고 각 문장을 SONAR space(pretrained sentence embedding space임)으로 변환함.
- 이후 1) decoder-only transformer로 next 'concept' prediction을 하거나, 2) diffusion-based generation을 하거나, 3) FAISS clustering을 통해 SONAR space를 quantize해서 

- average sentence length is assumed to be 10-20 tokens

sentence segmentation
SpaCy segmentor. rule-based nlp toolkit
Sagment any Text. SaT. predict sentence boundaries at the token level.
코드, 수식?
coherent thought cap.

decoder. minimize MSE loss

computer vision. aiming to learn conditional probability distributions over continuous data.
many different images may satisfy the same input prompt, hence the model has to learn a probability distribution over continuous pixel data. 
sentence embedding generation.
directly minimizing the MSE loss in the embedding space does not yield good results.

sentences in the SONAR space, despite being represented as continuous vectors, remain discrete combinatorail objects. this makes diffusion modeling struggle on the text modality.
contrastive nature of cross-entropy loss based on softmax outputs which is used for next token prediction plays a critical role for many downstream task where higher accuracy is requred. on the opposite, continuous diffusion modeling does not allow to integrate such a contrastive objective.

quantizing said data to ultimately model with discrete units.
residual quantizers for the SONAR space. quantized large concept model based on these discrete units.
residual vector quantization. to discretize SONAR representations.
vector quantization maps continuous input embeddings to the nearest entry in a learnt codebook.
RVQ iteratively quantize residual errors from previous quantizations using additional codebook for each iteration.


# related works
Marfurt and Hendersen (2021) and Cornille et al. (2024) used predicted next sentence embeddings in a fully generative setting, for summarization and generic language modelling, respectively. considered sentence-level connections only as an addition to the token-level connections across sentences, not as their replacement.


PLANNER architecture (Zhang et al., 2023) consists of a variational autoencoder for paragraphs and a diffusion model trained to predict latent autoencoder representations conditional on the textual context or on the class label.
Lovalace et al. (2024) augmented a decoder-only language model with an encoded sematic proposal of the continuation text, with an easily guidable diffusion model predicting the embedding of the next proposal.
TEncDM model (Shabablin et al., 2024) performs diffusion in the space of contextual token embeddings which are then decoded non-autoregressively.

planning capabilities.
Semformer (Yin et al., 2024) proposed training transformers language models to plan several steps ahead by including special planning tokens, the representations of which are trained to be informative about the future toekns.
Ye et al. (2024) applied discrete diffusion to language models as an alternative to autoregressive generation, more suitable for tasks that require multi-step planning
Ubukata et al. (2024) give an overview of applications of diffusion for planning tasks.






## 생각
- token도 이미 큰 baggage/overhead인데 거기다 sentence까지 더하자고? + separately trained
- 채팅, 코드, 수식 등의 다양한 포맷의 데이터에서는 sentence를 어떻게 자르나? sentence segmentation이 arbitrary 하기 때문에 (not to mention one 'concept' per sentence assumption, which I find.. uncool) 너무 지저분해 질 것 같음.
- sentence로 생각을 confine하기 싫음.
- continuous end-to-end stream of vector representation으로 깔끔하게 latent reasoning을 했으면 좋겠다.
- next sentence prediction이 아니라 생각 전체를 다 하고 이후 일필휘지로 갈기는 게 더 직관적. 생각을 마치고.
- 그 생각의 흐름은 diffusion으로 하는거. 굉장히 직관적으로 합리적이고 매력적.

- the text modality remains discrete, and despite dealing with continuous representations in the SONAR space, all possible text sentences are a cloud of points rather than a real continous distribution in the SONAR space.
