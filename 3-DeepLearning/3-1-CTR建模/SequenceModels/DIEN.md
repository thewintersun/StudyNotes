## Deep Interest Evolution Network for Click-Through Rate Prediction

地址：https://arxiv.org/pdf/1809.03672.pdf

作者：Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, Kun Gai

机构：Alibaba Inc, Beijing, China

发表： Accepted by AAAI 2019



### 摘要

点击率（CTR）预测，其目标是估计用户点击的概率，已成为广告系统的核心任务之一。对于点击率预测模型，需要捕捉用户行为数据背后的潜在用户兴趣。此外，考虑到外部环境和内部认知的变化，用户兴趣随时间动态变化。兴趣建模的CTR预测方法有多种，但大多将行为的表征直接视为兴趣，缺乏对具体行为背后潜在兴趣的专门建模。此外，很少有工作考虑兴趣的变化趋势。

In this paper, we propose a novel model, named Deep Interest Evolution Network~(DIEN), for CTR prediction. Specifically, we ==design interest extractor layer to capture temporal interests from history behavior sequence==. At this layer, we introduce ==an auxiliary loss to supervise interest extracting at each step==. As user interests are diverse, especially in the e-commerce system, we propose interest evolving layer to capture interest evolving process that is relative to the target item. At interest evolving layer, attention mechanism is embedded into the sequential structure novelly, and the effects of relative interests are strengthened during interest evolution. 

In the experiments on both public and industrial datasets, DIEN significantly outperforms the state-of-the-art solutions. Notably, DIEN has been deployed in the display advertisement system of Taobao, and obtained 20.7% improvement on CTR.



### 介绍

相比与DIN，DIEN最大的改进insight来自于对用户潜在兴趣的刻画，DIN并没有考虑到用户兴趣的变化趋势，相当于是用过去的一个一个的点去刻画用户当前对于商品的兴趣，而DIEN则把这些点连接成了一条线，通过两层gru去动态的刻画用户兴趣。

传统的基于RNN的用户兴趣表达有两个问题，一方面，他们将RNN的隐层直接作为用户的兴趣表达，缺乏足够的监督性保证隐层与用户兴趣有直接的关系。另一方面，对于序列中的每一个item，RNN将用户前一次行为对当前行为的影响认为是相同的，然而，并不是用户的每一次行为都与上一次行为有着直接的关系。

![image-20211101113502262](D:\Notes\raw_images\image-20211101113502262.png)

而DIEN采取的两层GRU刻画用户兴趣，一层GRU是 **interest extractor layer**，用来从用户的历史行为中获取当下的兴趣，通过**auxiliary loss**来辅助学习，auxiliary loss通过用户的下一次行为去监督学习当前的隐层，当前的隐层即为用户的兴趣状态，从而使得隐层能够更好的表达用户的兴趣，另一方面，auxiliary loss可以解决序列梯度衰减的问题，当序列长度超过100时，DIN使用的attention方法与sum pooling效果基本没差。

传统的GRU：
$$
\begin{equation} \begin{split} &u_t=\sigma(W^ui_t+U^uh_{t-1}+b^u)\\ &r_t=\sigma(W^ri_t+U^rh_{t-1}+b^r)\\ &\tilde{h_t}=tanh(W^hi_t+r_t{\odot}U^hh_{t-1}+b^h)\\ &h_t=(1-u_t){\odot}h_{t-1}+t_t{\odot}\tilde{h_t}\\ \end{split} \end{equation}
$$
$\sigma$  是sigmoid激活函数，$\odot $是点乘，$i_t $是用户第$t$次的行为，$h_t  $代表第$t$个隐层。GRU的问题在于 $h_t $不能够很好的表达出 $ i_t$ ，==最终的点击行为的预估只依赖于最后一个隐层，导致前边的隐层学习不够充分，也就导致了隐层对用户兴趣的表达不够充分==。因此，DIEN在这里加入了auxiliary loss，为了这部分的学习，同时加入了一些采样的真实负样本。因此样本的合集为 $\{e^i_b,\tilde{e_b^i}\} $，其中$ e_b^i $为点击的正样本， $\tilde{e^i_b} $ ==采样的负样本==，auxiliary loss与ctr的loss结合起来，$ \alpha $作为超参进行调节
$$
L_{aux}=-\frac{1}{N}(\sum^N_{i=1}\sum_tlog\sigma(t_t^i,e_b^i[t+1])+log(1-\sigma(h_t^i,\tilde{e^i_b[t+1]})))\\ L=L_{target}+{\alpha}*L_{aux}

$$
而考虑到用户当前的行为很可能与用户很久之前的行为有关系，另外一层 $gru$ 是 **interest evolving layer**，==用来获取与当前target item相关的用户历史行为==，刻画用户潜在兴趣的变化过程。==用户兴趣的逐步变化，需要考虑到不同品类间的区别，比如服装和手机，对于target item==，我们应该更多的考虑同品类下用户兴趣的演变。在这一层，DIEN将attention与传统的gru模型结合在一起进化成了一种新的结构(AUGRU)。
$$
Attention\ function\\ a_t=\frac{exp(h_t,We_a)}{\sum^T_{j=1}exp(h_jWe_a)}
$$
$e_a $是该广告类目的embedding的concat结果，原文中说的是where ea is the concat of embedding vectors from fields in category ad，我理解的是不同体系类目的embedding，如果直接使用ad的embedding，泛化性会比较差。$W{\in}R^{n_H*n_A} $， $n_H $是隐层的纬度， $n_A $是广告embedding的维度。几种不同的结合attention与gru的方法：

- GRU with attentional input(AIGRU)：比较naive的方法，直接改变extractor layer输出的隐层，$ i_t=h_t*a_t $，这种方法的问题在于即使 $i_t $为全零的向量，依旧会对GRU的结果产生影响。
- Attention based GRU(AGRU)：这种方法首先在nlp问答领域被使用，通过改变GRU的结果，是的AGRU能够提取到问题的关键信息。AGRU中直接使用attention score替换掉了GRU中的更新门$ h_t^"=(1-a_t)*h^"_{t-1}+a_t*\tilde{h_t^"} $，AGRU的核心思想在于少提取不相关item的embedding信息。但是由于attention作为一个标量与embedding相乘，并没有考虑到embedding中不同维度的重要性。
- GRU with attentional update gate(AUGRU)：为了解决AGRU的问题，AUGRU结合了AGRU与update gate，使用attention score去scale update gate。 $\tilde{u_t^"}=a_t*u_t^"\ \ \ h_t^"=(1-\tilde{u_t^"}){\odot}h^"_{t-1}+\tilde{u_t^"}\odot{\tilde{h_t^"}}$

在线上实验中，DIEN做了很多的优化:

- GRU中隐层状态计算的并行化以及用尽可能多的kernal
- 将临近的用户请求放到一个batch中，从而更好的利用GPU模型压缩

### 总结

整篇文章的思路主要在于对DIN中attention方法的改造，采用了更复杂的模型结构从而更好的贴近自身业务。但是仅仅从线上工程的角度来讲，这么庞大的gru耗时优化就不是小厂可以轻松复制的。同时，增加这种复杂的模型结构，线上真实的受益有多少，有多少是来源于模型，有多少是来源于新的数据，都很难说的清楚。不过，整篇文章对用户建模的整体思路，auxilary loss的选择以及将attention注入GRU的方式都值得我们去借鉴，可以将这些方法融合到我们的业务中。