## Sequence Modeling With CTC

文章地址：https://distill.pub/2017/ctc/

作者：[Awni Hannun](http://ai.stanford.edu/~awni/)

机构：[Stanford University](http://cs.stanford.edu/)

时间：Nov. 27, 2017



A visual guide to Connectionist Temporal Classification, an algorithm used to train deep neural networks in speech recognition, handwriting recognition and other sequence problems.

How CTC collapsing works

![1578382243512](D:\Notes\raw_images\1578382243512.png)

## Introduction

Consider speech recognition. We have a dataset of audio clips and corresponding transcripts. Unfortunately, we don’t know how the characters in the transcript align to the audio. This makes training a speech recognizer harder than it might at first seem.

Without this alignment, the simple approaches aren’t available to us. We could devise a rule like “one character corresponds to ten inputs”. But people’s rates of speech vary, so this type of rule can always be broken. Another alternative is to hand-align each character to its location in the audio. From a modeling standpoint this works well — we’d know the ground truth for each input time-step. However, for any reasonably sized dataset this is prohibitively time consuming.

This problem doesn’t just turn up in speech recognition. We see it in many other places. Handwriting recognition from images or sequences of pen strokes is one example. Action labelling in videos is another.

![img](https://distill.pub/2017/ctc/assets/handwriting_recognition.svg)

**Handwriting recognition:** The input can be (x,y)(*x*,*y*) coordinates of a pen stroke or pixels in an image.

![img](https://distill.pub/2017/ctc/assets/speech_recognition.svg)

**Speech recognition:** The input can be a spectrogram or some other frequency based feature extractor.

Connectionist Temporal Classification (CTC) is a way to get around (避免) not knowing the alignment between the input and the output. As we’ll see, it’s especially well suited to applications like speech and handwriting recognition.

------

To be a bit more formal, let’s consider mapping input sequences $X = [x_1, x_2, \ldots, x_T] $ , such as audio, to corresponding output sequences $Y = [y_1, y_2, \ldots, y_U] $,  such as transcripts. We want to find an accurate mapping from X’s to Y’s.

There are challenges which get in the way of us using simpler supervised learning algorithms. In particular:

- ==Both X and Y can vary in length.  （X和Y都可能是变长的）==
- ==The ratio of the lengths of X and Y can vary.==
- ==We don’t have an accurate alignment (correspondence of the elements) of X and Y.==

The CTC algorithm overcomes these challenges. For a given X it gives us an output distribution over all possible Y’s. We can use this distribution either to *infer* a likely output or to assess the *probability* of a given output.

Not all ways of computing the loss function and performing inference are tractable. We’ll require that CTC do both of these efficiently.

**Loss Function:** For a given input, we’d like to train our model to maximize the probability it assigns to the right answer. To do this, we’ll need to efficiently compute the conditional probability $p(Y \mid X)$. The function $p(Y \mid X)$ should also be differentiable, so we can use gradient descent.

**Inference:** Naturally, after we’ve trained the model, we want to use it to infer a likely Y given an X. This means solving $Y^* \enspace =\enspace {\mathop{\text{argmax}}\limits_{Y}} \enspace p(Y \mid X) $ . Ideally  $Y^*$  can be found efficiently. With CTC we’ll settle for an approximate solution that’s not too expensive to find.

## The Algorithm

The CTC algorithm can assign a probability for any Y given an X. The key to computing this probability is how CTC thinks about alignments between inputs and outputs. We’ll start by looking at these alignments and then show how to use them to compute the loss function and perform inference.

### Alignment

The CTC algorithm is *alignment-free* — it doesn’t require an alignment between the input and the output. However, to get the probability of an output given an input, CTC works by summing over the probability of all possible alignments between the two. We need to understand what these alignments are in order to understand how the loss function is ultimately calculated.

To motivate the specific form of the CTC alignments, first consider a naive approach. Let’s use an example. Assume the input has length six and Y = [c, a, t]. One way to align X  and Y is to assign an output character to each input step and collapse repeats.

![1578554791402](D:\Notes\raw_images\1578554791402.png)

This approach has two problems.

- Often, it doesn’t make sense to force every input step to align to some output. In speech recognition, for example, the input can have stretches of silence with no corresponding output.
- We have no way to produce outputs with multiple characters in a row. Consider the alignment [h, h, e, l, l, l, o]. Collapsing repeats will produce “helo” instead of “hello”.

==上述对齐方式有两个问题==：

- 通常这种对齐方式是不合理的。比如在语音识别任务中，有些音频片可能是无声的，这时候应该是没有字符输出的
- 对于一些本应含有重复字符的输出，这种对齐方式没法得到准确的输出。例如输出对齐的结果为![[h,h,e,l,l,l,o]](https://math.jianshu.com/math?formula=%5Bh%2Ch%2Ce%2Cl%2Cl%2Cl%2Co%5D)，通过去重操作后得到的不是“hello”而是“helo”。

To get around these problems, CTC introduces a new token to the set of allowed outputs. This new token is sometimes called the *blank* token. We’ll refer to it here as *ϵ*. The  *ϵ* token doesn’t correspond to anything and is simply removed from the output.

The alignments allowed by CTC are the same length as the input. We allow any alignment which maps to Y after merging repeats and removing *ϵ* tokens:

![img](https://distill.pub/2017/ctc/assets/ctc_alignment_steps.svg)

If Y has two of the same character in a row, then a valid alignment must have an *ϵ* between them. With this rule in place, we can differentiate between alignments which collapse to “hello” and those which collapse to “helo”.

Let’s go back to the output [c, a, t] with an input of length six. Here are a few more examples of valid and invalid alignments.

![1578554811283](D:\Notes\raw_images\1578554811283.png)

The CTC alignments have a few notable properties. 

- First, the allowed alignments between X and Y are monotonic. If we advance to the next input, we can keep the corresponding output the same or advance to the next one. 
- A second property is that the alignment of X to Y is many-to-one. One or more input elements can align to a single output element but not vice-versa. 
- This implies a third property: the length of Y cannot be greater than the length of X.

CTC算法的对齐方式有下列属性：

- ==输入与输出的对齐方式是单调的，即如果输入下一输入片段时输出会保持不变或者也会移动到下一个时间片段==。
- 输入与输出是多对一的关系。
- 输出的长度小于等于输入。

### Loss Function

The CTC alignments give us a natural way to go from probabilities at each time-step to the probability of an output sequence.

![1578556089593](D:\Notes\raw_images\1578556089593.png)

To be precise, the CTC objective for a single (X, Y) pair is:
$$
p(Y \mid X) = \sum_{A \in \mathcal{A}_{X,Y}}\prod_{t=1}^T p_t(a_t \mid X)
$$

- The CTC conditional **probability**  
- **marginalizes** over the set of valid alignments 
- computing the **probability** for a single alignment step-by-step

Models trained with CTC typically use a recurrent neural network (RNN) to estimate the per time-step probabilities, $p_t(a_t \mid X)$ . An RNN usually works well since it accounts for context in the input, but we’re free to use any learning algorithm which produces a distribution over output classes given a fixed-size slice of the input.

If we aren’t careful, the CTC loss can be very expensive to compute. We could try the straightforward approach and compute the score for each alignment summing them all up as we go. The problem is there can be a massive number of alignments. For most problems this would be too slow.

Thankfully, ==we can compute the loss much faster with a dynamic programming algorithm. The key insight is that if two alignments have reached the same output at the same step, then we can merge them.==

![1578557016437](D:\Notes\raw_images\1578557016437.png)

Since we can have an *ϵ* before or after any token in Y, it’s easier to describe the algorithm using a sequence which includes them. We’ll work with the sequence 
$$
Z \enspace =\enspace [\epsilon, ~y_1, ~\epsilon, ~y_2,~ \ldots, ~\epsilon, ~y_U, ~\epsilon]
$$
which is ==Y with an *ϵ* at the beginning, end, and between every character==.

Let’s let *α* be the score of the merged alignments at a given node. More precisely, $\alpha_{s, t}$ is the CTC score of the subsequence $Z_{1:s}$ after t input steps. As we’ll see, we’ll compute the final CTC score, $P(Y \mid X)$, from the *α’s* at the last time-step. As long as we know the values of *α* at the previous time-step, we can compute $\alpha_{s, t}$  There are two cases.

**Case 1:**

![img](https://distill.pub/2017/ctc/assets/cost_no_skip.svg)

In this case, we can’t jump over $z_{s-1} $, the previous token in Z. 

- The first reason is that the previous token can be an element of Y, and we can’t skip elements of Y. Since every element of Y in Z is followed by an *ϵ*, we can identify this when $z_{s} = \epsilon$. 
- The second reason is that we must have an *ϵ* between repeat characters in Y. We can identify this when $z_s = z_{s-2}$.

To ensure we don’t skip $z_{s-1}$ , we can either be there at the previous time-step or have already passed through at some earlier time-step. As a result there are two positions we can transition from.

$$
\alpha_{s, t}  = (\alpha_{s-1, t-1} + \alpha_{s, t-1})  \cdot p_t(z_{s} \mid X)
$$
The CTC probability of the two valid subsequences after t-1 input steps.

The probability of the current character at input step t.

**Case 2:**

![img](https://distill.pub/2017/ctc/assets/cost_regular.svg)

In the second case, we’re allowed to skip the previous token in Z. We have this case whenever $z_{s-1}$ is an *ϵ* between unique characters. As a result there are three positions we could have come from at the previous step.

$$
\alpha_{s, t} = (\alpha_{s-2, t-1} + \alpha_{s-1, t-1} + \alpha_{s, t-1}) \cdot p_t(z_{s} \mid X)
$$
The CTC probability of the three valid subsequences after t-1input steps.

The probability of the current character at input step t.

Below is an example of the computation performed by the dynamic programming algorithm. Every valid alignment has a path in this graph.

![1578623261590](D:\Notes\raw_images\1578623261590.png)

假设我们现在有输入音频![X](https://math.jianshu.com/math?formula=X)对应的标定输出![Y](https://math.jianshu.com/math?formula=Y)为单词“ZOO”，为了方便解释下面动态规划的思想，现在每个字符之间还有字符串的首位插入空白占位符![\epsilon](https://math.jianshu.com/math?formula=%5Cepsilon)，得到下面结果

![Z=\{\epsilon,Z,\epsilon,O,\epsilon,O,\epsilon\}](https://math.jianshu.com/math?formula=Z%3D%5C%7B%5Cepsilon%2CZ%2C%5Cepsilon%2CO%2C%5Cepsilon%2CO%2C%5Cepsilon%5C%7D)

为了便于说明，先定义好下图的横纵坐标轴的含义，横轴是![X](https://math.jianshu.com/math?formula=X)的时间片单位为 t，纵轴为![Z](https://math.jianshu.com/math?formula=Z)序列单位为 s。根据CTC的对齐方式的三个特征，输入有9个时间片，标签内容是“ZOO”，![P(Y|X)](https://math.jianshu.com/math?formula=P(Y%7CX))的所有可能的合法路径如下图: 

![img](https:////upload-images.jianshu.io/upload_images/6983308-35805e1d6a9fe9b3.png?imageMogr2/auto-orient/strip|imageView2/2/w/1054/format/webp)

![\alpha](https://math.jianshu.com/math?formula=%5Calpha)表示对齐结果合并后节点的概率。![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)表示上图中坐标为(s,t)节点的概率，该点的概率计算分为下面两种情况：
 **Case 1：**

 1）如果![\alpha_{s,t}=\epsilon](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D%3D%5Cepsilon)，则![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)只能由前一个字符![\alpha_{s-1,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs-1%2Ct-1%7D)或者本身![\alpha_{s,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct-1%7D)得到。

 2）如果![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)不等于![\epsilon](https://math.jianshu.com/math?formula=%5Cepsilon)，但是![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)为连续字符的第二个，即![\alpha_{s}=\alpha_{s-2}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%7D%3D%5Calpha_%7Bs-2%7D)(![\alpha_{s-1}=\epsilon](https://math.jianshu.com/math?formula=%5Calpha_%7Bs-1%7D%3D%5Cepsilon))，则![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)只能由一个空白符![\alpha_{s-1,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs-1%2Ct-1%7D)或者其本身![\alpha_{s,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct-1%7D)得到，而不能由前一个字符得到。

上述两种情况中，![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)可以由下式算出，其中![p_{t}(z_{s}|X)](https://math.jianshu.com/math?formula=p_%7Bt%7D(z_%7Bs%7D%7CX))表示在时刻t输出字符![z_{s}](https://math.jianshu.com/math?formula=z_%7Bs%7D)的概率。
$$
\alpha_{s,t}=(\alpha(s,t-1)+\alpha(s-1,t-1))\cdot p_{t}(z_{s}|X)
$$
 **Case 2：**

 如果![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)不等于![\epsilon](https://math.jianshu.com/math?formula=%5Cepsilon)，则![\alpha_{s,t}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct%7D)可以由![\alpha_{s,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs%2Ct-1%7D)，![\alpha_{s-1,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs-1%2Ct-1%7D)以及![\alpha_{s-2,t-1}](https://math.jianshu.com/math?formula=%5Calpha_%7Bs-2%2Ct-1%7D)得来，可以表示为：
$$
\alpha_{s,t}=(\alpha(s,t-1)+\alpha(s-1,t-1)+\alpha(s-2,t-1))\cdot p_{t}(z_{s}|X)
$$

> ZOO 中ZO是两个不同的字符组成，中间的 *ϵ* 可跳过，而OO是相同的字符，中间 *ϵ* 不能跳过。

There are two valid starting nodes and two valid final nodes since the *ϵ* at the beginning and end of the sequence is optional. The complete probability is the sum of the two final nodes.

Now that we can efficiently compute the loss function, the next step is to compute a gradient and train the model. ==The CTC loss function is differentiable with respect to the per time-step output probabilities since it’s just sums and products of them.== Given this, we can analytically compute the gradient of the loss function with respect to the (unnormalized) output probabilities and from there run backpropagation as usual.

从图中可以看到合法路径由两个起始点，输出两个终止点，最后输出的条件概率为两个终止点输出概率的和。使用这种计算方法就能高效的计算损失函数，下一步的工作表示计算梯度用于训练模型。由于P(Y|X)的计算只涉及加法和乘法，因此是可导的。

For a training set $\mathcal{D}$ , the model’s parameters are tuned to minimize the negative log-likelihood (最小化负对数似然函数) 
$$
\sum_{(X, Y) \in \mathcal{D}} -\log p(Y \mid X)
$$
instead of maximizing the likelihood directly.

### Inference

After we’ve trained the model, we’d like to use it to find a likely output for a given input. More precisely, we need to solve:

$$
Y^*  = \enspace {\mathop{\text{argmax}}\limits_{Y}} \enspace p(Y \mid X) 
$$
One heuristic is to take the most likely output at each time-step. This gives us the alignment with the highest probability:

$$
A^* \enspace = \enspace {\mathop{\text{argmax}}\limits_{A}} \enspace \prod_{t=1}^{T}  p_t(a_t \mid X)
$$
We can then collapse repeats and remove *ϵ* tokens to get Y.

For many applications this heuristic works well, especially when most of the probability mass is alloted to a single alignment. However, this approach can sometimes miss easy to find outputs with much higher probability. The problem is, ==it doesn’t take into account the fact that a single output can have many alignments==.

Here’s an example. Assume the alignments $[a, a, ϵ]$ and $[a, a, a]$ individually have lower probability than $[b, b, b]$. But the sum of their probabilities is actually greater than that of [b, b, b]. The naive heuristic will incorrectly propose Y = [b] as the most likely hypothesis. It should have chosen Y = [a]. ==To fix this, the algorithm needs to account for the fact that [a, a, a] and [a, a, *ϵ*] collapse to the same output==.

通常这种启发式的算法很有效，但是这种方法忽略了一个输出可能对应多个对齐结果。例如![[a,a,\epsilon]](https://math.jianshu.com/math?formula=%5Ba%2Ca%2C%5Cepsilon%5D)和![[a,a,a]](https://math.jianshu.com/math?formula=%5Ba%2Ca%2Ca%5D)各自的概率均小于![[b,b,b]](https://math.jianshu.com/math?formula=%5Bb%2Cb%2Cb%5D)的概率，但是他们相加的概率比![[b,b,b]](https://math.jianshu.com/math?formula=%5Bb%2Cb%2Cb%5D)概率高。简单的启发是算法得到结果为![Y=[b]](https://math.jianshu.com/math?formula=Y%3D%5Bb%5D)，但是结果为![Y=[a]](https://math.jianshu.com/math?formula=Y%3D%5Ba%5D)更为合理。

We can use a ==modified beam search to solve this==. Given limited computation, the modified beam search won’t necessarily find the most likely Y. It does, at least, have the nice property that we can trade-off more computation (a larger beam-size) for an asymptotically better solution.

A regular beam search computes a new set of hypotheses at each input step. ==The new set of hypotheses is generated from the previous set by extending each hypothesis with all possible output characters and keeping only the top candidates==.

![img](https://distill.pub/2017/ctc/assets/beam_search.svg)A standard beam search algorithm with an alphabet of $\{\epsilon, a, b\}$ and a beam size of three.

We can modify the vanilla beam search to handle multiple alignments mapping to the same output.==In this case instead of keeping a list of alignments in the beam, we store the output prefixes after collapsing repeats and removing *ϵ* characters==. At each step of the search we accumulate scores for a given prefix based on all the alignments which map to it.

![1578625801367](D:\Notes\raw_images\1578625801367.png)

A proposed extension can map to two output prefixes if the character is a repeat. This is shown at T=3 in the figure above where ‘a’ is proposed as an extension to the prefix [a]. Both [a] and [a, a] are valid outputs for this proposed extension.

When we extend [a] to produce [a,a], we only want include the part of the previous score for alignments which end in *ϵ*. Remember, the *ϵ* is required between repeat characters. Similarly, when we don’t extend the prefix and produce [a], we should only include the part of the previous score for alignments which don’t end in *ϵ*.

Given this, we have to keep track of two probabilities for each prefix in the beam. The probability of all alignments which end in *ϵ* and the probability of all alignments which don’t end in *ϵ*. When we rank the hypotheses at each step before pruning the beam, we’ll use their combined scores.

![1578625876460](D:\Notes\raw_images\1578625876460.png)

The implementation of this algorithm doesn’t require much code, but it is dense and tricky to get right. Checkout this [gist](https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0) for an example implementation in Python.

In some problems, such as speech recognition, incorporating a language model over the outputs significantly improves accuracy. We can include the language model as a factor in the inference problem.

![1578625899445](D:\Notes\raw_images\1578625899445.png)

The function L(Y) computes the length of Y in terms of the language model tokens and acts as a word insertion bonus. With a word-based language model L(Y) counts the number of words in Y. If we use a character-based language model then L(Y) counts the number of characters in Y. The language model scores are only included when a prefix is extended by a character (or word) and not at every step of the algorithm. This causes the search to favor shorter prefixes, as measured by L(Y), since they don’t include as many language model updates. The word insertion bonus helps with this. The parameters *α* and *β* are usually set by cross-validation.

The language model scores and word insertion term can be included in the beam search. Whenever we propose to extend a prefix by a character, we can include the language model score for the new character given the prefix so far.

## Properties of CTC

We mentioned a few important properties of CTC so far. Here we’ll go into more depth on what these properties are and what trade-offs they offer.

### Conditional Independence

One of the most commonly cited shortcomings of CTC is the conditional independence assumption it makes.

![img](https://distill.pub/2017/ctc/assets/conditional_independence.svg) 

Graphical model for CTC.

==The model assumes that every output is conditionally independent of the other outputs given the input.== This is a bad assumption for many sequence to sequence problems.

Say we had an audio clip of someone saying “triple A”.  Another valid transcription could be “AAA”. If the first letter of the predicted transcription is ‘A’, then the next letter should be ‘A’ with high probability and ‘r’ with low probability. The conditional independence assumption does not allow for this.

![img](https://distill.pub/2017/ctc/assets/triple_a.svg)

If we predict an ‘A’ as the first letter then the suffix ‘AA’ should get much more probability than ‘riple A’. If we predict ‘t’ first, the opposite should be true.

In fact speech recognizers using CTC don’t learn a language model over the output nearly as well as models which are conditionally dependent.  However, a separate language model can be included and usually gives a good boost to accuracy.

The conditional independence assumption made by CTC isn’t always a bad thing. Baking in strong beliefs over output interactions makes the model less adaptable to new or altered domains. For example, we might want to use a speech recognizer trained on phone conversations between friends to transcribe customer support calls. The language in the two domains can be quite different even if the acoustic model is similar. With a CTC acoustic model, we can easily swap in a new language model as we change domains.

### Alignment Properties

The CTC algorithm is *alignment-free*. The objective function marginalizes over all alignments. While CTC does make strong assumptions about the form of alignments between X and Y, the model is agnostic as to how probability is distributed amongst them. In some problems CTC ends up allocating most of the probability to a single alignment. However, this isn’t guaranteed. 

As mentioned before, ==CTC only allows *monotonic* alignments==. In problems such as speech recognition this may be a valid assumption. ==For other problems like machine translation where a future word in a target sentence can align to an earlier part of the source sentence, this assumption is a deal-breaker.==

Another important property of ==CTC alignments is that they are *many-to-one*==. Multiple inputs can align to at most one output. In some cases this may not be desirable. We might want to enforce a strict one-to-one correspondence between elements of X and Y. Alternatively, we may want to allow multiple output elements to align to a single input element. ==For example, the characters “th” might align to a single input step of audio==. A character based CTC model would not allow that.

==The many-to-one property implies that the output can’t have more time-steps than the input==. This is usually not a problem for speech and handwriting recognition since the input is much longer than the output. However, for other problems where Y is often longer than X, CTC just won’t work.

## CTC in Context

In this section we’ll discuss how CTC relates to other commonly used algorithms for sequence modeling.

### HMMs

At a first glance, a Hidden Markov Model (HMM) seems quite different from CTC. But, the two algorithms are actually quite similar. Understanding the relationship between them will help us understand what advantages CTC has over HMM sequence models and give us insight into how CTC could be changed for various use cases.

Let’s use the same notation as before, X is the input sequence and Y is the output sequence with lengths T and U respectively. We’re interested in learning $p(Y \mid X)$. One way to simplify the problem is to apply Bayes’ Rule:
$$
p(Y \mid X) \; \propto \; p(X \mid Y) \; p(Y)
$$
The $p(Y) $ term can be any language model, so let’s focus on $p(X \mid Y)$. Like before we’ll let $\mathcal{A}$ be a set of allowed alignments between X and Y. Members of $\mathcal{A}$ have length T. Let’s otherwise leave $\mathcal{A}$ unspecified for now. We’ll come back to it later. We can marginalize over alignments to get 
$$
p(X \mid Y)\; = \; \sum_{A \in \mathcal{A}} \; p(X, A \mid Y)
$$
To simplify notation, let’s remove the conditioning on Y, it will be present in every $p(\cdot)$.  With two assumptions we can write down the standard HMM.

![1578640592717](D:\Notes\raw_images\1578640592717.png)

The first assumption is the usual Markov property. The state  $a_t$ is conditionally independent of all historic states given the previous state $a_{t-1}$. The second is that the observation $x_t$ is conditionally independent of everything given the current state $a_t$.

![img](https://distill.pub/2017/ctc/assets/hmm.svg) 

The graphical model for an HMM.

Now we can take just a few steps to transform the HMM into CTC and see how the two models relate. First, let’s assume that the transition probabilities $p(a_t \mid a_{t-1})$ are uniform. This gives

$$
_p(X) \enspace \propto \enspace \sum_{A \in \mathcal{A}} \enspace \prod_{t=1}^T \; p(x_t \mid a_t)
$$
There are only two differences from this equation and the CTC loss function. 

The first is that we are learning a model of X given Y as opposed to Y given X. 

The second is how the set $\mathcal{A}$ is produced. Let’s deal with each in turn.

The HMM can be used with discriminative models which estimate $p(a \mid x)$. To do this, we apply Bayes’ rule and rewrite the model as
$$
p(X) \enspace \propto \enspace \sum_{A \in \mathcal{A}} \enspace \prod_{t=1}^T \; \frac{p(a_t \mid x_t)\; p(x_t)}{p(a_t)} \enspace \propto \enspace \sum_{A \in \mathcal{A}} \enspace \prod_{t=1}^T \; \frac{p(a_t \mid x_t)}{p(a_t)}. 
$$
If we assume a uniform prior over the states a and condition on all of X instead of a single element at a time, we arrive at 
$$
p(X) \enspace \propto \enspace \sum_{A \in \mathcal{A}} \enspace \prod_{t=1}^T \; p(a_t \mid X).
$$
The above equation is essentially the CTC loss function, assuming the set $\mathcal{A}$ is the same. In fact, the HMM framework does not specify what $\mathcal{A}$  should consist of. This part of the model can be designed on a per-problem basis. In many cases the model doesn’t condition on Y*Y* and the set $\mathcal{A}$  consists of all possible length T sequences from the output alphabet. In this case, the HMM can be drawn as an *ergodic* state transition diagram in which every state connects to every other state. The figure below shows this model with the alphabet or set of unique hidden states as \{a, b, c\}.

In our case the transitions allowed by the model are strongly related to Y. We want the HMM to reflect this. One possible model could be a simple linear state transition diagram. The figure below shows this with the same alphabet as before and Y = [a, b]. Another commonly used model is the *Bakis* or left-right HMM. In this model any transition which proceeds from the left to the right is allowed.

![1578712023780](D:\Notes\raw_images\1578712023780.png)

In CTC we augment the alphabet with *ϵ* and the HMM model allows a subset of the left-right transitions. The CTC HMM has two start states and two accepting states.

One possible source of confusion is that the HMM model differs for any unique Y. This is in fact standard in applications such as speech recognition. The state diagram changes based on the output Y. However, the functions which estimate the observation and transition probabilities are shared.

Let’s discuss how CTC improves on the original HMM model. First, we can think of the CTC state diagram as a special case HMM which works well for many problems of interest. Incorporating the blank as a hidden state in the HMM allows us to use the alphabet of Y as the other hidden states. This model also gives a set of allowed alignments which may be a good prior for some problems.

Perhaps most importantly, CTC is discriminative. It models $p(Y \mid X)$ directly, an idea that’s been important in the past with other discriminative improvements to HMMs. Discriminative training let’s us apply powerful learning algorithms like the RNN directly towards solving the problem we care about.

### Encoder-Decoder Models

The encoder-decoder is perhaps the most commonly used framework for sequence modeling with neural networks. These models have an encoder and a decoder. The encoder maps the input sequence X*X* into a hidden representation. The decoder consumes the hidden representation and produces a distribution over the outputs. We can write this as 
$$
\begin{aligned} H\enspace &= \enspace\textsf{encode}(X) \\[.5em] p(Y \mid X)\enspace &= \enspace \textsf{decode}(H). \end{aligned}
$$
The $\textsf{encode}(\cdot)$ and $\textsf{decode}(\cdot)$ functions are typically RNNs. The decoder can optionally be equipped with an attention mechanism. The hidden state sequence H*H* has the same number of time-steps as the input, T. Sometimes the encoder subsamples the input. If the encoder subsamples the input by a factor s*s* then H will have T/s time-steps.

We can interpret CTC in the encoder-decoder framework. This is helpful to understand the developments in encoder-decoder models that are applicable to CTC and to develop a common language for the properties of these models.

**Encoder:** The encoder of a CTC model can be just about any encoder we find in commonly used encoder-decoder models. For example the encoder could be a multi-layer bidirectional RNN or a convolutional network. There is a constraint on the CTC encoder that doesn’t apply to the others. The input length cannot be sub-sampled so much that T/s is less than the length of the output.

**Decoder:** We can view the decoder of a CTC model as a simple linear transformation followed by a softmax normalization. This layer should project all T steps of the encoder output H into the dimensionality of the output alphabet.

We mentioned earlier that CTC makes a conditional independence assumption over the characters in the output sequence. This is one of the big advantages that other encoder-decoder models have over CTC — they can model the dependence over the outputs. However in practice, CTC is still more commonly used in tasks like speech recognition as we can partially overcome the conditional independence assumption by including an external language model.

## Practitioner’s Guide

So far we’ve mostly developed a conceptual understanding of CTC. Here we’ll go through a few implementation tips for practitioners.

**Software:** Even with a solid understanding of CTC, the implementation is difficult. The algorithm has several edge cases and a fast implementation should be written in a lower-level programming language. Open-source software tools make it much easier to get started:

- Baidu Research has open-sourced [warp-ctc](https://github.com/baidu-research/warp-ctc). The package is written in C++ and CUDA. The CTC loss function runs on either the CPU or the GPU. Bindings are available for Torch, TensorFlow and [PyTorch](https://github.com/awni/warp-ctc).
- TensorFlow has built in CTC loss and CTC beam search functions for the CPU.
- Nvidia also provides a GPU implementation of CTC in cuDNN versions 7 and up.

**Numerical Stability:** Computing the CTC loss naively is numerically unstable. One method to avoid this is to normalize the \alpha*α*’s at each time-step. The original publication has more detail on this including the adjustments to the gradient. In practice this works well enough for medium length sequences but can still underflow for long sequences. A better solution is to compute the loss function in log-space with the log-sum-exp trick. Inference should also be done in log-space using the log-sum-exp trick.

**Beam Search:** There are a couple of good tips to know about when implementing and using the CTC beam search.

The correctness of the beam search can be tested as follows.

1. Run the beam search algorithm on an arbitrary input.
2. Save the inferred output $\bar{Y}$ and the corresponding score $\bar{c}$.
3. Compute the actual CTC score c for $\bar{Y}$.
4. Check that $\bar{c} \approx c$ with the former being no greater than the latter. As the beam size increases the inferred output $\bar{Y}$ may change, but the two numbers should grow closer.

A common question when using a beam search decoder is the size of the beam to use. There is a trade-off between accuracy and runtime. We can check if the beam size is in a good range. To do this first compute the CTC score for the inferred output $c_i$. Then compute the CTC score for the ground truth output $c_g$. If the two outputs are not the same, we should have $c_g \lt c_i$. If  $c_i << c_g$ then the ground truth output actually has a higher probability under the model and the beam search failed to find it. In this case a large increase to the beam size may be warranted.