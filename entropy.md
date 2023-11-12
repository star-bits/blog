# Entropy (information theory)
- *{expected/average}* amount of *{information content/surprisal/unpredictability/uncertainty/disorder/randomness}* in a *{system/message/random variable/probability distribution}*
- low probability = high surprisal = high information content = greater randomness = high entropy

## Formula for Information content (Surprisal)
- $I(x) = -\log P(x)$
- $P(x)$ is a probability of an event x.
- When P(x) is close to 1, the surprisal of the event is zero.
- When P(x) is close to 0, the surprisal of the event is very high.
- Log is the only function that satisfies this boundary condition.

## Formula for Entropy
- $H(X) = -\sum_{x} P(x) \log P(x)$
- Sum of information content weighted by its probability
- $-(0.5 \cdot \log_2(0.5) + 0.5 \cdot \log_2(0.5)) = 1$
- Coin flip has a maximum entropy of one bit.

## Mutual Information (Information Gain)
- Amount of information that two random variables X and Y share
- Measures how much knowing one of these variables reduces uncertainty about the other.
- $I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$
- $P(x,y)$ is the joint probability of X and Y taking values x and y.
- $P(x)$ is the marginal probability of X taking value x.
- $P(y)$ is the marginal probability of Y taking value y.

## KL Divergence
- Measures the amount of information lost when approximating true distribution P with estimated distribution Q. 
- $KL(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$
- $P(x)$ is the probability of event x in true distribution P.
- $Q(x)$ is the probability of event x in estimated distribution Q.
- low KL divergence between P and Q = Q is a good approximation of P = not much information is lost by using Q to represent P
- high KL divergence between P and Q = Q is a poor approximation of P = significant amount of information is lost by using Q to represent P
- KL divergence can be used to measure how well a language model approximates the true distribution of words in a corpus. The smaller the KL divergence, the better the language model is at generating realistic text.

## Mutual Information vs. KL Divergence
- Mutual information measures the degree of association between two random variables, while KL divergence measures the difference between two probability distributions.

## Gibbs formula for Entropy
- $S = -k_B \sum_i (p_i \ln p_i)$
- $S$ is the entropy of the system.
- $k_B$ is Boltzmann's constant.
- $p_i$ is the probability of the system being in a particular microstate $i$.
- $-p_i \ln p_i$ represents the contribution to the entropy from each microstate $i$.
- low probability (of the system being in a particular microstate) = high entropy

### Intuitive understanding of Gibbs formula for Entropy
- Heat is transferred to or within a system
- = Particles have more energy
- = Particles can occupy more positions and move in more directions
- = Increase in the number of possible arrangements or configurations of the particles
- = Increase in the number of accessible microstates available to the system
- = Decrease in the likelihood of the system being in any particular microstate
- = Higher entropy

## Second law of thermodynamics
- $\Delta S = \frac{Q}{T}$
- $\Delta S$ is the change in entropy.
- $Q$ is the amount of heat transferred to or from the system.
- $T$ is the temperature at which the heat transfer occurs.
- If heat is added to a system at a constant temperature, the entropy of the system will increase, since the added energy will increase the degree of disorder in the system.
