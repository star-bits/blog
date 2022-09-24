# entropy

von Neumann advised Shannon to name his newly devised quantity "Shannon entropy" since no one knows what entropy is, so Shannon would always have an edge in any discussion. 

ok, but what *is* entropy?

quick answer: the degree of disorder

But, I must admit, that definition is good for reminding ourselves what entropy is, but not so much for learning what it is. To do that, let's start from the (Shannon's) definition of information.

## what is information?

Shannon modeled information as events which occur with certain probabilities. It means that the amount of information in an event must depend only upon its probability. The more surprised we are by an event happening, the more information this event carries. 

Shannon set three boundary conditions for $I(p_x)$:
1. $I$ is a function of $p_x$.
2. $I(p_x)$ is a continuous function.
3. $I(p_x, p_y) = I(p_x) + I(p_y)$

Then $I(p_x) = - \log(p_x)$.

Let's make sense of this. Picture a $y = - \log(x)$ graph, $(0 \leq x \leq 1)$. As $x \rightarrow 0$, $y \rightarrow \infty$. And as $x \rightarrow 1$, $y \rightarrow 0$. Just as when the probability of an event reaches 0, surprisal of the event is high, and when the probability of an event reaches 1, surprisal of the event is 0. This is why log is the function that satisfies the specific set of characterization.

## which leads us back to entropy:

Entropy is a measure of how "surprising" the average outcome of a variable is. Something surprising has high entropy, and something expected has low entropy. 

It's about the level of uncertainty as compared to certainty, unexpectedness as compared to expectedness. And, in line with the definition from statistical mechanics, the disgree of disorder, as compared to order.

Shannon's entropy $H(X)$ can be expressed as $H(X) = - \sum p(x) \log p(x)$.
