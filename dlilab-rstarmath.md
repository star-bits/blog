## rStar-Math의 핵심 아이디어
- code-augmented CoT (verified reasoning traces)
  - code execution verification
- two models: policy model & reward model (Process PREFERENCE Model)
- self-evolution

## Process Preference Model
- contrastive loss, preference pairs, pairwise ranking loss.
- based on q-value (averate reward)
- conventional method: directly use Q-values as reward labels, which are inherently noisy and imprecise.
- PRM, ORM
- fig 5. boost in perf using ppm.
- llm에서 헤드만 갈아끼워서 [-1, 1] range 예측
- rlhf의 교훈. judging is easier

## Self-evolution rounds
- round 1: ds-coder-v2-instruct. 236B. bootstrapping. terminal-guided. (rollout: 8, candidates: 5, depth: 16)
- round 2: SLM. (rollout: 16, candidates: 6, depth: 16). O(rollout * candidate# * depth)
- round 3: PPM-augmented MCTS.
- round 4: unsolved after 16 rollouts -> 64 -> 128 (candidates: 16)
- sft data. top2 trajectories with the highest average q-value

## MCTS
- child nodes per a node. expansion
- rollout. simulation til terminal (or depth limit). reward 1 or -1. backpropagation up to root.
- UCT. exploitation + exploration.
- PPM directly predicts a non-zero initial q value.

## Related works 
- CoT. reasoning trace
- Process Supervision. process reward
- Test-time compute

## datasets
- MATH
- AIME
- problems are augmented using gpt-4

## Findings
- emergence. intrinsic self-reflection capability.
- recognizes the mistake and self-corrects.
- MCTS-driven deep thinking.
- PPM becomes the key determinant of the upper performance limit.
