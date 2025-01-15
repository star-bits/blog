# Exchanging Pretraining and Test-Time Compute
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314)

## 여기서 사용할 test-time compute에는 어떤 것들이?

## 같은 FLOPs에서 test-time compute와 pretraining cost의 tradeoff 관계
- efficacy of a given approach heavily correlates with the difficulty of the problem
- adaptive, compute-optimal scaling of test-time computation
- 1. verifier, 2. proposal distribution (via revisions)
- 1. searching against dense, process-based verifier reward models. 2. updating the model's distribution over a response adaptively
- effectiveness varies depending on the difficulty of the prompt. -> compute-optimal scaling strategy.
- test-time computation을 실행해서 output을 더 좋게 할 건지 pretraining을 해서
- benefits of scaling up test-time compute. 그 범위가 어디일까
- modifying proposal distribution (iteratively asking the model to revise its response). revise incorrect answers - improving the proposal distribution. requence of N revisions. 
- MATH [13] 벤치마크과 PaLM-2 모델을 사용.
- best-of-N sampling: select with a learned verifier or a reward model. or a process-based dense verifier. verify the correctness of individual steps. Process-based Reward Model. PRM. 


easier problem -> can readily produce reasonable responses. sequence of N revisions. 
difficult problem -> requre searching, high-level approaches. sampling N independent responses. re-sampling independently in parallel. tree-search against a PRM. 

revision and search
1. input level. augmenting given prompt. additional set of tokens. modify the proposal distribution. RL-insprired ft methods. STaR or ReST. do not utilize any additional input tokens. but ft the model to induce an improved proposal distribution. Or, self-critique. improve its own propsosal distribution. critique and revise in iterative fashion.
2. output level. sampling multiple candidates. post-hoc verifiers/scorers. verifier to select the best answer from the proposal distribution. can be further improved by PRM. 
PRM800k dataset. GPT-4 generated. PaLM 2. distribution shift. approach of Wang et al. [45]. Appendix D.  
Best-of-N -> easier, Beam Search -> harder, Lookahead Search -> too much flops.
ㅜ

depends on the difficulty. the notion of question difficulty. efficacy of test-time computation. 



이렇게 하면 (compute-optimal strategy) best-of-N baseline보다 4배 적은 computation.
여기까지가 improved test-time scaling strategy.




we want to improve perf by increasing total flops by a factor of M.
M(X+Y) = total.
x = 6 N D_pretrain [14]
Y = 2 N D_infeerence [29]
N: model parameters, D_pretrain: the number of tokens used for pretraining
D_inference: the number of tokens generated at inference time.

R: 0.16, 0.79, 22.
difficult or larger D_infernce -> budget toward pretraining -> star is above the line
easy -> utilizing test-time compute is better

![fig9](https://github.com/user-attachments/assets/56f1d2f4-6405-4695-8aae-135ef6059366)

to what extent can test-time computation effectively substitute for additional pretraining.
FLOPs-matched comparison.
어떤 경우에선 작은 모델을 pretrain하고 test-time compute를 늘이는 게 더 나은 아웃풋을 내기도.
어려운 문제들에서는 별 효과 없더라.
