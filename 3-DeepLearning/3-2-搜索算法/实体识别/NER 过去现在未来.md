## NER的过去、现在和未来综述

文章地址： https://zhuanlan.zhihu.com/p/351132129

[TOC]

## NER的过去

### 背景

命名实体识别（NER, Named Entity Recognition），是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。本文主要讲解NER历史使用过的一些方法，如果更关注于现在使用功能的一些方法。

![111111111111111](D:\Notes\raw_images\111111111111111.png)

### 评价指标

使用实体级别的精确率Precision、召回率Recall、F1 Score



### 基于词典和规则的方法

利用词典，通过词典的先验信息，匹配出句子中的潜在实体，通过一些规则进行筛选。

或者利用句式模板，抽取实体，例如模板"播放歌曲${song}"，就可以将query="播放歌曲七里香"中的song=七里香抽取出来。

**正向最大匹配&反向最大匹配&双向最大匹配：**

原理比较简单，直接看代码：[ner_rule.py](https://link.zhihu.com/?target=https%3A//github.com/InsaneLife/MyPicture/blob/master/NER/ner_rule.py)

正向最大匹配：从前往后依次匹配子句是否是词语，以最长的优先。

后向最大匹配：从后往前依次匹配子句是否是词语，以最长的优先。

双向最大匹配原则：

- 覆盖token最多的匹配。

- 句子包含实体和切分后的片段，这种片段+实体个数最少的。

  

### 基于机器学习的方法

CRF，原理可以参考：Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data

<img src="https://pic4.zhimg.com/80/v2-0fac2b60b5b324bea1455e3a0514d0e7_720w.jpg" alt="img" style="zoom: 67%;" />

在随机变量X取值为x的条件下，随机变量Y取值为y的条件概率为：
$$
P(y \mid x)=\frac{1}{Z(x)} \exp \left(\sum_{i, k} \lambda_{k} t_{k}\left(y_{i-1}, y_{i}, x, i\right)+\sum_{i, l} u_{l} s_{l}\left(y_{i}, x, i\right)\right)  \\ Z(x)=\sum_{y} \exp \left(\sum_{i, k} \lambda_{k} t_{k}\left(y_{l-1}, y_{i}, x, i\right)+\sum_{i, l} u_{i} s_{l}\left(y_{l}, x, t\right)\right)
$$
其中 $t_k,s_l$ 是特征函数(如上图)， $\lambda_k,u_i$ 对应权值，$Z(x)$ 是规范化因子。

> 来自李航的统计学习方法



### 引入深度学习语义编码器

#### BI-LSTM + CRF

> [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991v1)

BI-LSTM-CRF 模型可以有效地利用过去和未来的输入特征。借助 CRF 层, 它还可以使用句子级别的标记信息。BI-LSTM-CRF 模型在POS（词性标注），chunking（语义组块标注）和 NER（命名实体识别）数据集上取得了当时的SOTA效果。同时BI-LSTM-CRF模型是==健壮的，相比之前模型对词嵌入依赖更小==。

> 文中对比了5种模型：LSTM、BI-LSTM、CRF、LSTM-CRF、BI-LSTM-CRF，LSTM:  通过输入门，遗忘门和输出门实现记忆单元，能够有效利用上文的输入特征。
>
> BI-LSTM：可以获取时间步的上下文输入特征。
>
> CRF: 使用功能句子级标签信息，精度高。
>
> 比较经典的模型，BERT之前很长一段时间的范式，小数据集仍然可以使用。

![img](https://pic2.zhimg.com/80/v2-89a852d114d5b5da24dbad6ba0f78b61_720w.jpg)

#### Stack-LSTM & char-embedding

> [Neural Architectures for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1603.01360)
>
> [Transition-based dependency parsing with stack long-short-term memory](https://www.oalib.com/paper/4074644)

Stack-LSTM 直接==构建多词的命名实体==。Stack-LSTM 在 LSTM 中加入一个栈指针。模型包含chunking和

1. 堆栈包含三个：output (输出栈/已完成的部分)，stack (暂存栈/临时部分)，buffer (尚未处理的单词栈)

2. 三种操作（action）：

3. 1. SHIFT: 将一个单词从 buffer 中移动到 stack 中；
   2. OUT: 将一个单词从 buffer 中移动到 output 中；
   3. REDUCE: 将 stack 中的单词全部弹出，组成一个块，用标签y对其进行标记， 并将其push到output中。

4. 模型训练中获取每一步的action的条件概率分布，标签是真实每一步 action 的概率分布。预测时候，同坐预测每一步action概率，用概率最大action来执行action操作。

5. 在REDUCE操作输出chunking块之后，通过LSTM对其编码输出chunk的向量表达，然后预测其标签。

举例见图示：

![img](https://pic1.zhimg.com/80/v2-0010c1fcd9afaa614d94bd31c2c44504_720w.jpg)

同时使用初始化的char-embedding，对于每一个词语，通过BI-LSTM将字符编码作为输入，输出词语的字符级别表达，然后concat词向量输入到BI-LSTM + CRF。

#### CNN + BI-LSTM + CRF

> [End-to-end Sequence Labeling via Bi-directional LSTM- CNNs-CRF](https://aclanthology.org/P16-1101.pdf)

- 通过CNN获取字符级的词表示。CNN是一个非常有效的方式去抽取词的形态信息（例如词的前缀和后缀）进行编码的方法，如图。

![img](https://pic4.zhimg.com/80/v2-9571f205b0fa897b707372ee80f8e647_720w.jpg)

- 然后将CNN的字符级编码向量和词级别向量concat，输入到BI-LSTM + CRF网络中，后面和上一个方法类似。整体网络结构：

![img](https://pic1.zhimg.com/80/v2-b1d6890f0a88c2e4f3ddc599fdd352b8_720w.jpg)

#### IDCNN

> [2017Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.02098)

针对Bi-LSTM解码速度较慢的问题，本文提出 ID-CNN 网络来代替 Bi-LSTM，在保证和 Bi-LSTM-CRF 相当的正确率，且带来了 14-20 倍的提速。句子级别的解码提速 8 倍相比于 Bi-LSTM-CRF。

CNN缺点：CNN 的上下文信息取决于窗口的大小，虽然不断地增加 CNN 卷积层最终也可以达到使每个 token 获取到整个输入句子作为上下文信息，但是其输出的分辨表现力太差。于是出现了扩张卷积（or 空洞卷积）：对于扩张卷积，有效 输入宽度可以随着深度呈指数增长，在每层不会有分辨率损失，并且可以估计 一定数量的参数

![img](https://pic2.zhimg.com/80/v2-86088020bda9021547fc5893cd603881_720w.jpg)

#### 胶囊网络

> [Joint Slot Filling and Intent Detection via Capsule Neural Networks](https://arxiv.org/abs/1812.09471)
> [Git: Capsule-NLU](https://link.zhihu.com/?target=https%3A//github.com/czhang99/Capsule-NLU)

NLU中两个重要的任务，Intent detection和slot filling，当前的无论pipeline或者联合训练的方法，没有显示地对字、槽位、意图三者之间的层次关系建模。本文提出将胶囊网络和dynamic routing-by-agreement应用于slot filling和intent detection联合任务.

1. 使用层次话的胶囊网络来封装字、槽位、意图之间的层次关系。
2. 提出rerouting的动态路由方案建模slot filling。

网络分为WordCaps、SlotCaps、IntentCaps。

**WordCaps:** 对于输入，输入BI-LSTM编码成为T个胶囊向量，和普通BI-LSTM并无区别：
$$
\overrightarrow{h_t} = LSTM_{forward}(w_t, \overrightarrow{h_{t-1}}) \\ \overleftarrow{h_t} = LSTM_{backward}(w_t, \overleftarrow{h_{t-1}})
$$
**SlotCaps:** 这里有k个slotCaps，对应k个NER的标签。作者利用第t个wordCap对于第k个slotCap的动态路由权重作为第t个字的NER预测标签的概率。初始向量：
$$
\mathbf{p}_{k | t}=\sigma\left(\mathbf{W}_{k} \mathbf{h}_{t}^{T}+\mathbf{b}_{k}\right)
$$
通过动态路由算法，更新权重：

<img src="https://pic1.zhimg.com/80/v2-4ce921909aab4a5208346f26ced65a14_720w.jpg" alt="img" style="zoom:80%;" />

输出胶囊向量：
$$
\mathbf{v}_{k}=\operatorname{squash}\left(\mathbf{s}_{k}\right)=\frac{\left\|\mathbf{s}_{k}\right\|^{2}}{1+\left\|\mathbf{s}_{k}\right\|^{2}} \frac{\mathbf{s}_{k}}{\left\|\mathbf{s}_{k}\right\|}
$$
最终slot filling的损失为：
$$
\hat{y}_{t}^{k} = {argmax}_{k \in K}(c_{kt}) \\ \mathcal{L}_{s l o t}=-\sum_{t} \sum_{k} y_{t}^{k} \log \left(\hat{y}_{t}^{k}\right)
$$
**IntentCaps**: 输入是slotCaps的输出胶囊向量，第k个slotCap对第l个intentCap的表达向量：$\mathbf{q}_{l | k}=\sigma\left(\mathbf{W}_{l} \mathbf{v}_{k}^{T}+b_{l}\right)$

同样通过动态路由算法获得输出胶囊向量，向量的模作为属于l类的概率：$u_l = DynamicRouting \  (q_{l|k, iter_{intent}})$

损失使用了max-margin Loss：
$$
\begin{aligned} \mathcal{L}_{\text {intent}} &=\sum_{l=1}^{L}\left\{\mathbb{I} z=z_{l}\left\|\cdot \max \left(0, m^{+}-\left\|\mathbf{u}_{l}\right\|\right)^{2}\right.\right.\\ &\left. +\lambda \mathbb{I} z \neq z_{l} \mathbb{U} \cdot \max \left(0,\left\|\mathbf{u}_{l}\right\|-m^{-}\right)^{2}\right\} \end{aligned}
$$
**Re-Routing:**为了将Intent的信息提供给NER使用，提出了Re-Routing机制，它和动态路由机制很像，唯一改动在于权重更新中同时使用了，其中是norm值最大的胶囊向量。
$$
\mathrm{b}_{k t} \leftarrow \mathrm{b}_{k t}+\mathbf{p}_{k | t} \cdot \mathbf{v}_{k}+\alpha \cdot \mathbf{p}_{k | t}^{T} \mathbf{W}_{R R} \hat{\mathbf{u}}_{\tilde{z}}^{T}
$$


#### Transformer

直说吧，就是BERT，Bert之前万年BI-LSTM+CRF，Bert之后，基本没它什么事儿了，Bert原理不多赘述，应用在NER任务上也很简单，直接看图，每个token的输出直接分类即可：

<img src="https://pic2.zhimg.com/80/v2-f26f9bbc740d435a1c9d85902cd94ec5_720w.jpg" alt="img" style="zoom:67%;" />

### 语义特征

#### char-embedding

> [Neural Architectures for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1603.01360)

将英文字符拆解为字母，将词语的每个字母作为一个序列编码，编码器可以使用RNN，CNN等。

<img src="https://pic2.zhimg.com/80/v2-9b7019459bb7f888c6d84161742c623d_720w.jpg" alt="img" style="zoom:67%;" />

#### Attending to Characters in Neural Sequence Labeling Models

> [Attending to Characters in Neural Sequence Labeling Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1611.04361)

使用了==单词或字符级别 embedding 组合==，并在两种embedding之间使用attention机制 “灵活地选取信息” ，而之前模型是直接将两种embedding concat。

![img](https://pic4.zhimg.com/80/v2-4e1873f4f38feaa1f88767d1c27b9143_720w.jpg)

直接看公式，z是一个动态权重：
$$
z=\sigma\left(W_{z}^{(3)} \tanh \left(W_{z}^{(1)} x+W_{z}^{(2)} m\right)\right) \\ \tilde{x}=z \cdot x+(1-z) \cdot m
$$
并交叉熵上增加额外的loss:
$$
\widetilde{E}=E+\sum_{t=1}^{T} g_{t}\left(1-\cos \left(m^{(t)}, x_{t}\right)\right) \\ g_{t}=\left\{\begin{array}{ll}{0,} & {\text { if } w_{t}=O O V} \\ {1,} & {\text { otherwise }}\end{array}\right.
$$
非OOV单词希望m和x越相似越好。

> char-embedding学习的是所有词语之间更通用的表示，而word-embedding学习的是特特定词语信息。对于频繁出现的单词，可以直接学习出单词表示，二者也会更相似。

#### Radical-Level Features (中文部首)

> [Character-Based LSTM-CRF with Radical-LevelFeatures for Chinese Named Entity Recognition](https://link.zhihu.com/?target=http%3A//www.nlpr.ia.ac.cn/cip/ZongPublications/2016/13%E8%91%A3%E4%BC%A0%E6%B5%B7Character-Based%20LSTM-CRF%20with%20Radical-Level%20Features%20for%20Chinese%20Named%20Entity%20Recognition.pdf).

也是一种char embedding方法，将每个中文字拆分为各个部首，例如“朝”会被拆分为字符：十、日、十、月。后面结构都类似。

#### n-gram prefixes and suffixes

> [Named Entity Recognition with Character-Level Models](https://link.zhihu.com/?target=https%3A//nlp.stanford.edu/manning/papers/conll-ner.pdf)

提取每个词语的前缀和后缀作为词语的特征，例如："aspirin"提取出3-gram的前后缀：{"asp", "rin"}.

包含两个参数：n、T。n表示 n-gram size，T是阈值，表示该后缀或者前缀至少在语料库中出现过T次。



### 多任务联合学习

#### 联合分词学习

> [Improving Named Entity Recognition for Chinese Social Mediawith Word Segmentation Representation Learning](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P16-2025)

将中文分词和 NER任务联合起来。使用预测的分割标签作为特征作为NER的输入之一，为NER系统提供更丰富的边界信息。

![img](https://pic3.zhimg.com/80/v2-ab9dd18c4fdff58e54a3bf404cc473fa_720w.jpg)

分词语料目前是很丰富的。如果目标域数据量比较小，不妨用分词的语料作为源域，来预训练一个底层编码器，然后再在目标域数据上联合分词任务fine-tuning.

#### 联合意图学习

##### 1. slot-gated

> Slot-Gated Modeling for Joint Slot Filling and Intent Prediction

slot-gated 这篇文章提出了slot-gate 将槽位和意图的关系建模，同时使用了attention方法，所以介绍这篇文章直接一起介绍attention，之前attention相关的就不介绍了，

<img src="https://pic2.zhimg.com/80/v2-e1b1927d7e3933dea97d37c8f612b37d_720w.jpg" alt="img" style="zoom:80%;" />

**底层特征**：使用Bi-LSTM结构，

**attention：** slot filling attention计算方式
$$
c_i^S = \sum^T_{j=1} \alpha^S_{i,j} h_j \\ \alpha^S_{i,j} = \frac{exp(e_{i,j})}{\sum_{k=1}^T exp(e_{j,k})} \\ e_{i,k} = V^T \sigma(W_{he}^S h_k + W_{ie} h_i)
$$
$其中，c^S_i \in R^{bs*T}, 和h_j一致。e_{i,k} \in R^1, e_{i,k} 计算的是h_k和当前输入向量h_i之前的关系。作者TensorFlow源码W^S_{ke}h_k用的卷积实现，而W^S_{ie}h_i用的卷积实现，而W^S_{ie}h_i$用的线性映射_linear()。T是attention维度，一般和输入向量一致。

Attention具体细节见：[Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://link.zhihu.com/?target=https%3A//blog.csdn.net/shine19930820/article/details/83052232)，

**slot-Gate:**

利用意图上下文向量来建模槽意图关系，以提高槽填充性能。如图：

<img src="https://pic3.zhimg.com/80/v2-4c9d33ad2764ad4b986aaf47eb6ca6f2_720w.jpg" alt="img" style="zoom: 33%;" />

##### 2. Stack-Propagation

> [A Stack-Propagation Framework with Token-level Intent Detection for Spoken Language Understanding](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1214/)

首先什么是Stack-Propagation呢，如下图所示：

![img](https://pic1.zhimg.com/80/v2-b2d5efed63fb8a99351403b27ea4de10_720w.jpg)

它是区别于多任务， 不同的任务通过stack（级联？）的方式一起学习优化。

然后本文将意图任务的输出stack输入给NER任务，具体做法：

- Token intent（意图阶段）：假设每个token都会有一个意图的概率分布（标签是句子的意图，通过大量数据训练，就能够学到每个token的意图分布，对于每个意图的’偏好‘ ），最终句子的意图预测通过将每个token的意图预测结果投票决定。
- Slot Filling：输入包含下面三部分： $\mathbf{h}_{i-1}^{S}, \mathbf{y}_{i-1}^{S}, \mathbf{y}_{i}^{I} \oplus \mathbf{e}_{i}$ ，其中是上一阶段 token intent 的预测结果的 intent id，然后经过一个意图向量矩阵，转化为意图向量，输入给实体预测模块，解码器就是一层 LSTM+softmax。

![img](https://pic3.zhimg.com/80/v2-f035230d0acee829f7065d6c678e9ee2_720w.jpg)

##### 3. BERT for Joint Intent Classification and Slot Filling

> [BERT for Joint Intent Classification and Slot Filling](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1902.10909)

原理如图，底层编码器使用了BERT，token的输出向量接softmax预测序列标签，cls向量预测意图。

<img src="https://pic2.zhimg.com/80/v2-68444945f03786e488a06a2b414fdbb9_720w.jpg" alt="img" style="zoom:67%;" />



## NER现在

文章：https://zhuanlan.zhihu.com/p/425268651

 过去和现在是相对于某个时间节点的，暂且以bert作为这个时间节点，本文就主要寻找NER在BERT之后的一些方法。本文将从以下方面展开：

![v2-cd5152c5d16880c227e7f59295f2f249_r](D:\Notes\raw_images\v2-cd5152c5d16880c227e7f59295f2f249_r.jpg)



### 解码框架（Framework）

> 这里归类的解码器似乎也不太合适，但是也找不到好的了。

sequence labeling（序列标注）==将实体识别任务转化为序列中每个token的分类任务==，例如softmax、CRF等。相比于sequence labeling的解码方式，最近也有很多新的解码方式。

#### Span

> [SpanNER: Named EntityRe-/Recognition as Span Prediction](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.00641v1.pdf)
> [Coarse-to-Fine Pre-training for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//aclanthology.org/2020.emnlp-main.514.pdf)

==本质是预测实体的开始和结束节点，即对于每个token都会预测它是不是某个实体的开始和结束，对于多实体类型==，有两种方式：

- 对于每个token，会预测start和end，对于start，是一个多分类(N+1)任务，N是实体个数

![img](https://pic4.zhimg.com/80/v2-db52582f55406f558b9161cbacd516df_720w.png)

- 对于每一个类别，都预测对应的start和end。

![img](https://pic3.zhimg.com/80/v2-4b48ec7e355cf231d2c631169becd88e_720w.jpg)

这种方式的优点是，==可以解决实体嵌套问题==。但是也有一个缺点，就是预测实体的start和end是独立的（理论上应该联合start和end一起考虑是否是一个实体），解码阶段容易解码出非实体，例如：

![img](https://pic1.zhimg.com/80/v2-7fee7b5f47e74d2cee8d7a89adc7abec_720w.png)

> token“林”预测为start，“伟”预测为end，那么“林丹对阵李宗伟”也可以解码为一个实体。

所以，==span更适合去做实体召回==，或者句子中只有一个实体（这种情况应该很少），所以==阅读理解任务一般会使用功能span作为解码==。

损失函数：
$$
\mathcal{L}=-\frac{1}{n} \sum_{i=1}^{n} y_{i}^{\mathrm{s}} \log \left(P_{\text {start }}\left(y_{i}^{s} \mid x_{i}\right)\right) -\frac{1}{n} \sum_{i=1}^{n} y_{i}^{\mathrm{e}} \log \left(P_{\text {end }}\left(y_{i}^{e} \mid x_{i}\right)\right)
$$

#### MRC（阅读理解）

> [A Unified MRC Framework for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.11476v6.pdf)

这个方法很有意思，当我们要识别一句话中的实体的时候，其实可以通过问题和答案的方式。解码阶段还是可以使用CRF或者span。例如问题：句子中描述的人物是？；句子：林丹在伦敦夺冠；答案：林丹；

![img](https://pic4.zhimg.com/80/v2-839cca23651122df404d988f474883bf_720w.jpg)

个人==主观意见认为不实用==，原因如下：

- 对于不同的实体，需要去构建问题模板，而问题模板怎么构建呢？人工构建的话，那么人构建问题的好坏将直接影响实体识别。
- 增加了计算量，原来输入是句子的长度，现在是问题+句子的长度。
- span的问题，它也会有（当然span的优点它也有），或者解码器使用CRF。

#### 片段排列分类

> [Span-Level Model for Relation Extraction](https://link.zhihu.com/?target=https%3A//aclanthology.org/P19-1525.pdf)
> [Instance-Based Learning of Span Representations](https://link.zhihu.com/?target=https%3A//aclanthology.org/2020.acl-main.575)

其实span还是属于token界别的分类任务，而片段排列+分类的方式，是直接对于所有可能的片段，输入是span-level的特征，输出的是实体的类别。片段排列会将所有可能的token组合作为输入进行分类，例如：

![img](https://pic3.zhimg.com/80/v2-3a9881680c82c5081104e1e25bdd1122_720w.jpg)

span-level 特征一般包含：

- 片段的编码，pooling或者start和end向量的拼接，一般比较倾向于后者。
- 片段的长度，然后通过embedding矩阵转为向量。
- 句子特征，例如CLS向量。

模型的话，参考这个模型，其中的a,b 阶段是实体识别：

![img](https://pic1.zhimg.com/80/v2-3acd875bf376945d63d7f9b532f095f0_720w.jpg)

片段排列分类

> [SpERT：Span-based Joint Entity and Relation Extraction with Transformer Pre-training](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.07755)

缺点：

- 对于长度为N的句子，如果不限制长度的话，会有N(N+1)/2，长文本的话，==片段会非常多，计算量大，而且负样本巨多==，正样本极少。
- 如果==限制候选片段长度的话，那么长度又不灵活==。

其实刚刚讲到span合适用来做候选召回，那么span的预测结果再用分类的方式进行识别，也不失为一种方式。

#### Span+片段排列

> span+分类：Boundary Enhanced Neural Span Classification for Nested Named Entity Recognition
> 序列标注（BIO）+分类：[A Boundary-aware Neural Model for Nested Named Entity Recognition](https://link.zhihu.com/?target=https%3A//aclanthology.org/D19-1034/)

然后还真就看到这篇文章，和我想的一样doge。我上面讲到的是通过pipeline的方式，那么能不能将两者联合呢？文章思路就是联合边界预测和span分类（增强边界），共享底层的编码器，通过实验证明==联合学习可以提高两者的效果==。在预测使用span的结果在进行实体分类。

![img](https://pic3.zhimg.com/80/v2-0a04d7c6d0bb89c489e8c1b400eb1a82_720w.jpg)



### 融合知识

#### 隐式融合

这部分主要指==通过预训练模型中融入知识==，一种是通过在目标域的数据上进行[adaptive pretrain](https://link.zhihu.com/?target=https%3A//medium.com/jasonwu0731/pre-finetuning-domain-adaptive-pre-training-of-language-models-db8fa9747668)，例如是对话语料，那么使用对话语料进行适配 pretrain（预训练）。

另一种是在预训练阶段引入实体、词语实体信息，这部分论文也比较同质化，例如[nezha](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.00204)/[ernie](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.09223)/[bert-wwm](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1906.08101)，以[ernie](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.09223)为例，将知识信息融入到训练任务中，ERNIE提出一种知识掩盖策略取代Bert的mask，包含实体层面和短语级别的掩盖，见下图：

![img](https://pic3.zhimg.com/80/v2-42074d685887968ee5e55349d17cea42_720w.jpg)

- Basic-Level Masking: 和bert一样，随机选取token做mask。
- Phrase-Level Masking: 会将语法中的短语做mask，例如：a series of|such as 等。
- Entity-Level Masking: 会将一些实体整个mask，主要实体包含人物、地点、组织、产品名等。

> 训练预料包括中文维基百科，百度百科，百度新闻（最新的实体信息），百度贴吧。



#### 融合分词信息

> multi-grained: fine-grained and coarse-grained
>
> multi-grained 翻译应该是多粒度，但是个人认为主要是融入了分词的信息，因为Bert就是使用字。

中文可以使用词语和字为粒度作为Bert输入，各有优劣，那么有么有可能融合两种输入方式呢:

**前期融合**

[LICHEE](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2108.00801.pdf), 前期即输入embedding 层面融合，使用max-pooling融合两种粒度（词和字粒度）embedding：
$$
\begin{aligned} \vec{e}_{i}^{f} &=\text { embedding }_{\text {fine }}\left(t_{i}^{f}\right) \\ \vec{e}_{i-k}^{c} &=\text { embedding }_{\text {coarse }}\left(t_{j-k}^{c}\right) \\ \vec{e}_{i} &=\max \text {-pool }\left(\vec{e}_{i}^{f}, \bar{e}_{j-k}^{c}\right) \end{aligned}
$$
[TNER](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1911.04474.pdf)：改进了Transformer的Encoder，更好地建模character级别的和词语级别的特征。通过引入方向感知、距离感知和unscaled的attention，改造后的Transformer encoder也能够对NER任务显著提升。

> 文章比较有意思是==分析了Transformer的注意力机制，发现其在方向性、相对位置、稀疏性方面不太适合NER任务==。

<img src="https://pic4.zhimg.com/80/v2-62e9f3f0f712c899b5c69c824787c88f_720w.jpg" alt="img" style="zoom:80%;" />

embedding中加入了word embedding和character embedding，character embedding经过Transformer encoder之后，提取n-gram以及一些非连续的字符特征。

计算self-attention包含了相对位置信息，但是是没有方向的，并且在经过W矩阵映射之后，相对位置信息这一特性也会消失。所以提出计算attention权值时，将词向量与位置向量分开计算：
$$
Q,K,V = HW_q,H_{d_k},HW_v, \\
R_{t-j} = [...sin(\frac{t-j}{10000^{\frac{2i}{d_k}}}) cos(\frac{t-j}{10000^{\frac{2i}{d_k}}})...]^T \\
A^{rel}_{t,j} = Q_tK^T_j + Q_tR^T_{t-j}+uK^T_j + vR^T_{t-j} \\
Attn(Q,K,V) = softmax(A^{rel})V
$$

> 去掉了attention计算中的scaled，即不除以 $\sqrt{d_k}$ ，认为效果更好。

[FLAT](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2004.11795.pdf), 将Lattice结构和Transformer相结合，解决中文会因为分词引入额外的误差，并且能够利用并行化，提升推理速度。如下图，通过词典匹配到的潜在词语(Lattice)，然后见词语追加到末尾，然后通过start和end位置编码将其和原始句子中的token关联起来。

![img](https://pic1.zhimg.com/80/v2-aa13a1b250c301f095e573cd8d0a4014_720w.jpg)

另外也修改了attention的相对位置编码（加入了方向、相对距离）和attention计算方式（加入了距离的特征），和[TNER](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1911.04474.pdf)类似，后续也有一篇[Lattice bert](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.07204)，内容几乎一样。

**中期融合**

> [ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1911.00720)

即==在encoder某些层中融入词语和字的输出==。在==char的中间层添加N-gram的embedding输入==。

![img](https://pic4.zhimg.com/80/v2-53c4917a715455807e5ff379097851c3_720w.jpg)

这种==n-gram加入到char的输出上，可能会找出信息泄露==，例如MLM预测粤的时候，由于融入了“港澳”、“粤港澳”、“粤港澳大湾区”，可能会对预测粤的时候泄露答案。聪明的小伙伴会说那我直接mask全词啊，那这里如果mask掉“粤港澳大湾区”，其实会找出mask大部分句子，模型很难学。另一种就是修改attention的可见矩阵。

**后期融合**

[Ambert](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2008.11869.pdf)， 字和词语各自经过一个共享的encoder，然后将粗细粒度的输出融合，看输出==不适合NER任务，更适合分类任务==。

![img](https://pic4.zhimg.com/80/v2-29eeaf4e63dc6a7dc7e90c3d9abfb0d7_720w.jpg)



#### 融合知识图谱信息

> [K-BERT: Enabling Language Representation with Knowledge Graph](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.07606.pdf)

知识图谱包含实体、实体类型、实体和实体的关系（边），怎么把这些信息融入到输入中呢？[K-BERT](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.07606.pdf)使用方式很直接，如下图：

![img](https://pic4.zhimg.com/80/v2-c51df46d5676e626ae6bc7522088eef3_720w.jpg)

例如句子中，cook在图谱中是apple的CEO，那么直接将其插入到句子中，那不就扰乱了句子顺序吗，并且对于其他token是引入了额外的信息干扰。因此它提出了两种方法解决这个问题。

- 位置编码，原始句子的位置保持不变，序列就不变，同时对于插入的“CEO”、"Apple"和“cook”的位置是连续，确保图谱知识插入的位置。

- 同时对于后面的token，“CEO”、"Apple"属于噪声，因此利用可见矩阵机制，使得“CEO”、"Apple"对于后面的token不可见，对于[CLS]也不可见。

  

### 标注缺失

首先对于NER标注，由于标注数据昂贵，所以会通过远程监督进行标注，由于远监督词典会造成高准确低召回，会引起大量未标注问题？另外即使标注，存在实体标注缺失是很正常的现象，除了去纠正数据（代价过高）之外，有么有其他的方式呢？

#### Auto-NER

> [Learning Named Entity Tagger using Domain-Specific Dictionary](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1809.03599)
> [Better Modeling of Incomplete Annotations for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//aclanthology.org/N19-1079.pdf)

当使用词典进行实体的远监督标注时，==由于词典有限，一般会造成标注数据中实体高准确，低召回（未标注）的问题==。为了解决数据中的未标注问题，提出了Auto-NER with “Tie or Break”的方法。

![img](https://pic2.zhimg.com/80/v2-bf5374dd9205e707e129ba1a7fb816f1_720w.jpg)

具体算法如图，其中：

1. Tie：对于两个相邻的token，如果他们是属于同一个实体，那么他们之间是Tie。
2. Unknow：两个相邻的token其中一个属于未知类型的高置信实体，挖掘高置信实体使用[AutoPhrase](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.04457)。
3. Break：不属于以上情况，即非同一实体。
4. 两个Break之间的tokens作为实体，需要去识别对应的类别。
5. 计算损失的时候，对于Unknow不计算损失。（主要是为了缓解漏标（false negative）问题）

解决的问题：

- 即使远监督将边界标注错误，但是实体内部的多数tie还是正确的。

> 个人理解出发点：1. 提出tie or break是为了解决边界标注错误问题，Unknow不计算损失缓解漏标（false negative）问题。
> 但是有个问题，文中提到了false negative的样本来自于high-quality phrase，但是这些high-quality phrase是基于统计，所以对于一些低频覆盖不太好。

另外一篇论文也是类似的思想：[Training Named Entity Tagger from Imperfect Annotations](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.01441)，它每次迭代包含两步：

1. 错误识别：通过交叉训练识别训练数据集中可能的标签错误。

2. 权重重置：降低含有错误标注的句子的权重。

   

#### PU learning

[Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1906.01378)

主要==解决词典漏标或者标注不连续问题==，降低对于词典构造的要求。Unbiased positive-unlabeled learning正是解决未标记样本中存在正例和负例的情况，作者定义为：
$$
R_{\ell}=\pi_{n} \mathbb{E}_{\mathbf{X} \mid \mathrm{Y}=0} \ell(f(\boldsymbol{x}), 0)+\pi_{p} \mathbb{E}_{\mathbf{X} \mid \mathrm{Y}=1} \ell(f(\boldsymbol{x}), 1)
$$
$\pi_n$是负例，未标注样本属于是正例$\pi_p$ ，解决未标注问题就是怎么不用负样本去预估 $\mathbb{E}_{\mathbf{X} \mid \mathrm{Y}=0} \ell(f(\boldsymbol{x}), 0) $  。

> 为什么不用负样本，因为负样本可能保证未标注正样本。

作者将其转化为：
$$
\begin{aligned} \pi_{n} \mathbb{E}_{\mathbf{X} \mid \mathrm{Y}=0} \ell(f(\boldsymbol{x}), 0) &=\mathbb{E}_{\mathbf{X}} \ell(f(\boldsymbol{x}), 0) -\pi_{p} \mathbb{E}_{\mathbf{X} \mid \mathrm{Y}=1} \ell(f(\boldsymbol{x}), 0) \end{aligned}
$$
所以我直接去学正样本就好了嘛，没毛病。这里大概就能猜到作者会用类似 out of domain的方法了。

> 但是我感觉哪里不对，你这只学已标注正样本，未标注的正样本没学呢。

果然，对于正样本每个标签，构造不同的二分类器，只学是不是属于正样本。

> 我不是杠，但是未标注的实体仍然会影响二分类啊。



#### 负采样

> [Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2012.05426)

未标注会造成两类问题 1）降低正样本量。2）将未标注视为负样本。问题 1 可以通过adaptive pretrain缓解，而2后果却更严重，会对于模型造成误导，怎么消除这种误导呢，那就是负采样。

本文NER框架使用了前面介绍的片段排列分类的框架，即每个片段都会有一个实体类型进行分类，也更适合负采样。

**负采样：** 即==对于所有非实体的片段组合使用功能下采样，因为非实体的片段组合中有可能存在正样本，所以负采样一定程度能够**缓解**未标注问题。注意是缓解不是解决==。损失函数如下：
$$
\left(\sum_{(i, j, l) \in \mathbf{y}}-\log \left(\mathbf{o}_{i, j}[l]\right)\right)+\left(\sum_{\left(i^{\prime}, j^{\prime}, l^{\prime}\right) \in \hat{\mathbf{y}}}-\log \left(\mathbf{o}_{i^{\prime}, j^{\prime}}\left[l^{\prime}\right]\right)\right)
$$
其中前面部分是正样本，后面部分是负样本损失，$\hat{y}$ 就是采样的负样本集合。方法很质朴，我觉得比 PU learning有效。作者还证明了通过负采样，不讲未标注实体作为负样本的概率大于(1-2/(n-5))，缓解未标注问题。



## NER 2021未来

- Few-show & zero shot：如何通过少样本，甚至零样本能够获得性能不错的模型，例如怎么引入正则表达式的模板、prompt等方式。

  > [Template-Based Named Entity Recognition Using BART](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.01760.pdf)

- 融入知识： 未来，随着预训练模型越来越大，如果能够将知识从中剥离，使用较小的语言模型来加速训练。然后通过另一些方式来融入知识，例如检索的方式，像DeepMind 的 RETRO 和 OpenAI 的WebGPT。

- 迁移学习： 这个点有点泛，怎么利用语言模型学到的知识。为什么人可以识别到其中的实体，凭借的以往经验的迁移，达到举一反三；语法信息（句式等）；特定的句式; 模仿学习等。

- 解码方式：个人觉得 span、分类、序列标注似乎都并没有完美，span方式没有考虑整体序列标签之间的依赖关系；而分类的话还需要考虑实体长度，实际情况中实体长度可以是任意长度的（讲道理极限情况会存在）；序列标注不能很好解决嵌套问题等。目前有一些结合的方法，例如[Span+片段排列](https://link.zhihu.com/?target=https%3A//ojs.aaai.org/index.php/AAAI/article/view/6434)、[BIO+分类](https://link.zhihu.com/?target=https%3A//aclanthology.org/D19-1034/)，但还是有优化空间。

个人观点，仅供参考。

