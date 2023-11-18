# Entropy
- The {expected or average} amount of
- {disorder, unpredictability, surprisal, information content, or randomness} in a
- {system, message, or probability distribution}.

## Entropy in thermodynamics vs. information theory
- Thermodynamics: ordered system has low entropy; chaotic, random system has high entropy.
- Information theory: unpredictable, information-dense, high-surprisal message has high entropy; predictable message has low entropy.
- Random means unpredictable.

## Entropy in statistical mechanics
- $S = k_B \ln(W)$
  - $S$: Boltzmann entropy
  - $W$: total number of microstates, configurations in which the given macrostate can be realized
  - Measures the degree of disorder at a microscopic level in a system.
- $S = - k_B \sum p_i \ln(p_i)$
  - $S$: Gibbs entropy
  - $p_i$: probability of the system being in a particular $i$-th microstate
  - Connection to Boltzmann entropy:

## Entropy in information theory
- $H = -\sum_{i} p_i \ln(p_i)$
  - $H$: Shannon entropy
  - $p_i$: probability of the $i$-th event or state
  - $- \ln(p_i)$: information content of an event $i$
    - When $p_i$ is close to 1, the surprisal of the event is zero.
    - When $p_i$ is close to 0, the surprisal of the event is very high.
  - Measures the average information content in a set of messages or probability distribution.

## Entropy in thermodynamics
- $\Delta S = \frac{\Delta Q}{T}$
  - Second law of thermodynamics: $\Delta S_{isolated} \geq 0$
  - $\Delta Q$: heat transferred
  - $T$: temperature at which the heat transfer occurs
  - Measures the degree of disorder at a macroscopic level in a system.
  - Connection to Gibbs entropy: particles with higher energy can occupy a greater number of microscopic configurations, increasing the total accessible microstates availbale to the system, thereby reducing the probability of the system being in any specific microstate; low probability of specific microstate occupancy corresponds to higher entropy.
