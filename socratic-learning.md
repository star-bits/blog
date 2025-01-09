# RL through Socratic Learning
- [Boundless Socratic Learning with Language Games](https://arxiv.org/pdf/2411.16905) (아이디어만 던지는 논문임. 이걸 position paper라고 하나봄.)

## Socratic Learning이란 무엇인가
- pure recursive self-improvement (self-improvement: agent's own outputs influence its future learning.)
- resursive self-improvement where agent's inputs and outputs are compatible (i.e., live in the smae (here, language) space), and outputs become future inputs
- potentially unbounded improvement in a closed system -> RL
- RL agents' behavior changes the data distribution it learns on, which in turn affects its behavior policy.
- feedback, broad enough coverage of experience/data, scale (resource) (1,2: feasibiity, 3: practice)

## feedback
- true purpose resides in the external observer. feedback can only come from a proxy. 
- fundamental challenge. system-internal feedback is to be aligned with the observer. feedback must sufficiently aligned with the observer's evaluation metric.
- most common pitfall is a poorly designed reward function that becomes exploitable over time.
- self-correction of RL. what can self-correct is behavior given feedback, not the feedback itself.
- well-defined, grounded metrics in language space are often limited to narrow tasks, while more general-purpose mechanisms like ai-feedback are exploitable, especially if the input distribution is permitted to shift.
- next-token prediction loss is grounded, but insufficiently aligned with downstream usage, and unable to extrapolate beyond the training data
- human preferences are aligned by definition, but prevent learning in a closed system. (you need a human in the closed system) caching such preferences into a learned reward model (RLHF) makes it self-containedk but exploitable and potentially misaligned in the long-run, as well as weak on out-of-distribution data

## coverage
- by definition, a self-improving agent determines the distribution of data it learns from (자기가 훈련할 데이터를 만들어내니까)
- issues like collapse, drift, exploitation, or overfitting arises.
- preventing drift, collapse, or just narrowing of the generative distribution in a recursive process may be challenging
- needs to preserve sufficient coverage of the data distribution.
- coverage condition implies that the socratic learning system must keep generating language data, while preserving or expanding diversity over time

## RL works at scale
- scaling up compute sufficiently, even relatively straightforward RL algorithms can solve problems previously thought out of reach
- betting on scaling up computation (as opposed to building in human knowledge) has consistently paid off in the history of AI

## 어떻게 구현할 것인가 - Language Game
- wittgenstein's notion of language games
- language game. interaction of one or more agents that have language as inputs and outputs. scalar scoring function.
- scalable mechanism for unbounded interactive data generation and self-play
- logical consequence of the converage and feedback conditions; there is no form of interactive generation with tractiabel feedback that is not a language game
- it is not the words that capture meaning, but only the interactive nature of language can do so.
- narrow game -> reliable score function
- no individual lanuage game is perfectly aligned. filter the many games according to whether they make an overall net-positive condtribution
- meta-critic that can judge which games should be played
- structural leniency that give the language games framework a vast potential to scale.

## 생각
- 각각의 LLM에게 어떤 특성을 부여해 줘야 할까. 아이디어를 던지는 explorer. 아이디어를 비판하는 critic.
- 현실/정답에 ground 시킬 어떤 toolkit.. python
- 다른 특별한 방법 (모델의 구조를 바꾸는 - latent vector를 활용하거나, latent diffusion을 적용하거나)이 필요없나? 
- 그냥 적절한 메트릭으로 모델끼리 대화를 하게 하면서 데이터를 생성하고, 그 데이터를 다시 훈련에 활용하는 방법만으로 A-Superhuman-I가 가능할까?
- 인간의 reasoning 방법과 적은 데이터로도 reasoning을 하는 과정을 생각해보면 다른 무언가가 필요해보이는데..
- self-train. 숨에서 자기 머릿속을 들여다보듯이.
- 우주도 가만히 두면 알아서 지성체들이 등장하더라. 여기서 오는 완결성
- synthetic data만으로 extrapolation이 가능한가.
