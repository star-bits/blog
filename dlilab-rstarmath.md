## rStar-Math의 핵심 아이디어
- python
- Process PREFERENCE Model
- self-evolution

## Code-augmented
- code execution verification

## Process Preference Model
- contrastive loss, preference pairs, pairwise ranking loss.
- based on q-value
- PRM, ORM
- fig 5. boost in perf using ppm.
- llm에서 헤드만 갈아끼워서 [-1, 1] range 예측
- rlhf의 교훈. judging is easier

## Self-evolution rounds
- round 1: ds-coder-v2-instruct. 236B. bootstrapping. terminal-guided. (rollout: 8, candidates: 5, depth: 16)
- round 2: SLM. (rollout: 16, candidates: 6, depth: 16). O(rollout * candidate# * depth)
- round 3: PPM-augmented MCTS.
- round 4: unsolved after 16 rollouts -> 64 -> 128 (candidates: 16)

## MCTS
- child nodes per a node. expansion
- rollout. simulation til terminal (or depth limit). reward 1 or -1. backpropagation up to root.
- UTC

## Related works - CoT
- reasoning trace

## Related works - Process Supervision
- process reward

## Related works - Test-time compute

## datasets
- MATH
- AIME
- problems are augmented using gpt-4


## 추가적으로 알아본 코드

## 추가적으로 알아본 논문


- emergence. self-reflection, self-correction
