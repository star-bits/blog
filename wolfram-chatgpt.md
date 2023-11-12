# Notes from [Stephen Wolfram's blog post on ChatGPT](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)

I read Stephen Wolfram's blog post on ChatGPT, but was disappointed to find that it didn't delve into technical details as much as I had hoped. It was more about NLP in general. Nonetheless, there were still some useful insights that I could salvage. Here are some of them. The following excerpts may have been condensed or rephrased for clarity.

> In a crawl of the web there might be a few hundred billion words; in books that have been digitized there might be another hundred billion words.

Available NLP data: couple hundred billion words

> The big breakthrough in deep learning that occured around 2011 was associated with the discovery that in some sense it can be easier to find global minima when there are lots of weights involved than when there are fairly few.

A high-dimension space makes it less likely to become trapped in a local minimum.

> In the earlier days of neural nets, there was the idea that one should introduce complicated individual components into the neural net, to let it in effect "explicitly implement particular algorithmic ideas". But this has mostly turned out not to be worthwhile; instead, it's better just to deal with very simple components and let them "organize themselves" to achieve the equivalent of those algorithmic ideas. That's not to say that there are no "structuring ideas" that are relevant for neural nets. For example, having 2D arrays of neurons with local connections seems at least very useful in the early stages of processing images. And having patterns of connectivity that concentrate on "looking back in sequences" seems useful in dealing with things like human language. 

"Structuring ideas" can often be counterproductive. But of course, there are exceptions: convolution, residual, and attention--one of the most significant breakthroughs in deep learning.

> A piece of nueral net lore: one can often get away with a smaller network if there's a "squeeze" in the middle that forces everything to go through a smaller intermediate number of neurons.

Kinda like encoder-decoder system? Because it forces dimension reduction?

> "No-intermediate-layer"--or so-called "perceptron"--networks can only learn essentially linear functions--but as soon as there's even one intermediate layer it's always in principle possible to approximate any function arbitrarily well.

Universal approximation theorem

> Training rounds, or epochs, are like reminding a particular example, and is perhaps analogous to the usefulness of repetition in human memorization.

LLMs suck at negative feedback. Humans can automatically identify and mark uncharted meaning space as "wrong," while machines simply add vectors and provide "answers." Would it be possible to solve this issue by feeding all combinations of augmented "wrong" information with negative labels to the training data? 

> Most of a neural net is "idle" most of the time during training.

I haven't thought about this, but forward propagation (or backward propagation, for that matter) being computationally irreducible, I can't think of any way of implementing pipelining to neural net.

> Our current computers tend to have memory that is separate from their GPUs. But in brains it's presumably different--with every "memory element" (i.e. neuron) also being a potentially active computational element.

A chip that integrates memory and computational elements? Memory-computational units that combine a memory cell with some ALUs could serve as a neuron. Such a chip would require the architecture of the neural net, including any "structuring ideas" such as convolution, residual, or attention, to be physically carved onto it, greatly limiting its versatility. It is possible that a chip with a network consisting solely of fully-connected layers, modeled after the architecture of the human brain, could serve as a general-purpose chip. But it will come with the drawbacks of having fully-connected layers only.

> One can think of an embedding as a way to try to represent the "essence" of something by an array of numbers.

Understanding something involves abstracting its essential features. In other words, it's about performing dimension reduction on raw information. And deep learning is all about dimension reduction. 

> Transformers introduce the notion of "attention"--and the idea of "paying attention" more to some parts of the sequence than others. Maybe one day it'll make sense to just start a generic neural net and do all customization through training. But at least as of now it seems to be critical in practice to "modulize" things.

I guess you could say that the "structuring ideas" are ways to "modulize" neural nets.

> Dimension of embedding vector is 512 for the original transformer, 768 for GPT-2, and 12,288 for GPT-3.

GPT-3's embedding vector: 12,288 dimensions

> For each token ChatGPT has produced, there are 175 billion calculations to be done.

At least 175 billion calculations. 175 billion is just the number of weights.

> The main thing that's expensive about "back propagating" from the error is that each time you do this, *every* weight in the network will typically change at least a tiny bit.

*Every* weights. Otherwise training would be a lot cheaper.

> ChatGPT has a couple hundred billion weights, which is comparable to the total number of words (or tokens) of training data it's been given. It is empirically observed that the "size of the network" that seems to work so well is so comparable to the "size of the training data".

The number of tokens needs to be bigger than the number of parameters.

> When we run ChatGPT to generate text, we're basically having to use each weight once. So if there are *n* weights, we've got of order *n* computational steps to do. But if we need *n* words of training data to set up those weights, then from what we said above (two previous quotes) we can conclude that we'll need about n^2 computational steps to do the training of the network--which is why one ends up needing to talk about billion-dollar training efforts. 

*n* weights and *n* tokens of data -> n^2 computational steps

> Actual human feedback on "how to be a good chatbot" was given to ChatGPT, and another neural net model is built to attempt those ratings, essentially working like a loss function. It seems to have a big effect on the success of the system in producing "human-like" output.

[InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)

> Human brains have 100 billion or so neurons and maybe 100 trillion connections. Artificial neural net with about as many connections as brains have neurons is capable of doing a surprisingly good job of generating human language.

Human brain: 100 trillion weights

> On "paranthesis balancing" problem, even with 400,000 trained weights, there's a 15% probability that a neural net will lead to an unbalanced parenthesis.

It's surprisingly bad at it.

> Cases that a human "can solve at a glance" the neural net can solve too. But cases that require doing something "more algorithmic" the neural net tends to somehow be "too computationally shallow" to reliably do. Even ChatGPT has a hard time correctly matching parantheses in long sequences.

If a model is directed to solve a problem step by step rather than at a glance, would that approach yield different results?

> ChatGPT is just saying things that "sound right" based on what things "sounded like" in its training material. It's "merely" pulling out some "coherent thread of text" from the "statistics of conventional wisdom" that it's accumulated. Its amazing human-like results suggests that human language (and the patterns of thinking behind it) are somehow simpler and more "law like" in their structure than we thought. 

"Saying things that *"sound right"* based on what things *"sounded like"* in its training material"--I suspect that's what exactly humans do.
