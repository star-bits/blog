# entropy

von Neumann advised Shannon to name his newly devised quantity "Shannon entropy" since no one knows what entropy is, so Shannon would always have an edge in any discussion. 

ok, but what *is* entropy?

quick answer: the degree of disorder

But, I must admit, that definition is good for reminding ourselves what entropy is, but not so much for learning what it is. To do that, let's start from the (Shannon's) definition of information.

## what is information?

Shannon modeled information as events which occur with certain probabilities. It means that the amount of information in an event must depend only upon its probability. The more surprised we are by an event happening, the more information this event carries. 

Shannon set three boundary conditions for *information content* $I(x)$:
1. $I$ is a function of $x$.
2. $I(x)$ is a continuous function.
3. $I(x, y) = I(x) + I(y)$

Then $I(x) = - \log(x)$.

Let's make sense of this. Picture a $y = - \log(x)$ graph, $(0 \leq x \leq 1)$. As $x \rightarrow 0$, $y \rightarrow \infty$. And as $x \rightarrow 1$, $y \rightarrow 0$. Just as when the probability of an event reaches 0, surprisal of the event is high, and when the probability of an event reaches 1, surprisal of the event is 0. This is why log is the function that satisfies the specific set of characterization.

## which leads us back to entropy:

Entropy is a measure of the *expected* information content of a random variable. In other words, it is a measure of how surprising the *average* outcome of a random variable is. 

Something surprising has high entropy, and something expected has low entropy. So it's about the level of uncertainty. And, in line with the definition of entropy from statistical mechanics, *the disgree of disorder*.

To calculate the value of entropy, we sum up all information content with a weight equals to its occurence frequency: $H(X) = - \sum p(x) \log p(x)$.
