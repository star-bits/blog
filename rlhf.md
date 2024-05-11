# Notes from [Andrej Karpathy's talk on the State of GPT](https://youtu.be/bZQun8Y4L2A)

### GPT Assistant training pipeline

Stage | Pretraining | Supervised Finetuning | Reward Modeling | Reinforcement Learning
--- | --- | --- | --- | ---
Dataset | **Raw internet**, trillions of words | **Demonstrations (ideal Assistant responses)**, 10-100k prompt-response pairs, written by contractors | **Comparisons**, 100k-1M comparisons, written by contractors | **Prompts**, 10-100k prompts, written by contractors
Algorithm | **Language modeling** (predict the next token) | **Language modeling** (predict the next token) | **Binary classification** (predict rewards consistent with preferences) | **Reinforcement Learning** (generate tokens that maximize the reward)
Model | **Base model** | **SFT model** | **RM model** | **RL model**
Notes | 1000s of GPUs, months of training (ex: GPT, LLaMA, PALM) | 1-100 GPUs, days of training (ex: Vicuna-13B) | 1-100 GPUs, days of training | 1-100 GPUs, days of training (ex: ChatGPT)

- 99% of compute time is in pretraining stage.
- Tokenization. 10-100k tokens. 1 token ~= 0.75 words. BPE.
- Example models:
  - GPT-3: 175B. 50,257 vocab size. 2048 context length. Trained on 300B tokens.
  - LLaMA: 65B. 32,000 vocab size. 2048 context length. Trained on 1-1.4T tokens.
- Just by predicting the next token, it is forced to understand the stucture of text and all the different concepts therein.
- Prompting over finetuning. Fake documents. "Tricked" into performing a task by completing the document. Few-shot prompt.
- text-davinci-003 is a GPT-3 *base* model. Not an Assistant model. Base model does not answer questions. It only wants to complete internet documents. Often responds to questions with more questions. It can be tricked into performing tasks with prompt engineering. 
- LLMs don't want to succeed. They want to imitate training sets with a spectrum of performance qualities. You want to succeed, and you should ask for it. (ex: "You are a leading expert on this topic", "Pretend you have IQ 200")

### Supervised Finetuning
- Prompt-response. Human contractors. Still language modeling. Nothing changes algorithmically. Just swapping out a training set. After training we get a SFT model. This model is deployable. 

### RLHF pipeline
- Reward Modeling + Reinforcement Learning

### Reward Modeling
- Comparison dataset. SFT model creates multiple completions for the same prompt. Human contractors rank these completions. 
- Binary classification on all the possible pairs between these completions. 
- Dataset: prompt + completion + reward readout token (score of the completion)
- RM model will predict the reward. How good that completion is for that prompt. Makes a guess about the quality of each completion. Ground truth is coming from human contractors. Only the reward token is used, the rest are ignored. 
- This is not a deployable model. 
- RM model is for the following RL stage. 

### Reinforcement Learning
- With RM model, we can score the quality of any arbitrary completion for any given prompt. 
- Depends on the score of completion given by RM model, the SFT model gets reinforced. 
- Higher probability in the future for the tokens that were sampled at completion that scored high by RM model. 
- **Why RLHF model is better? It is easier to discriminate than to generate. Judging which output is good is much easier task.** 

### Human text generation vs. LLM text generation
- Humans have internal dialoge when generating text. LLMs don't reflect in a loop. They don't do sanity check. They don't correct their mistakes on their way (by default). 
- GPT will look at every single token and spend the same amount of compute at every one of them. 
- You can't expect the Transformer to do too much reasoning per token. You have to spread out the reasoning across more tokens. 
- They can get unlucky in their sampling. They cannot recover from the bad sampling of tokens. They even know when they screwed up. They will continue the sequence even if they know that the sequence is not going to work out. 
