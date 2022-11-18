## Deep Interest Network for Click-Through Rate Prediction

论文地址：https://arxiv.org/abs/1706.06978

作者：Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai

发表：Accepted by KDD 2018

机构：阿里巴巴



### 摘要

Click-through rate prediction is an essential task in industrial applications, such as online advertising. Recently deep learning based models have been proposed, which follow a similar Embedding\&MLP paradigm. In these methods large scale sparse input features are first mapped into low dimensional embedding vectors, and then transformed into fixed-length vectors in a group-wise manner, finally concatenated together to fed into a multilayer perceptron (MLP) to learn the nonlinear relations among features.

 In this way, ==user features are compressed into a fixed-length representation vector==, in regardless of what candidate ads are. The use of fixed-length vector will be a bottleneck, which brings difficulty for Embedding\&MLP methods to capture user's diverse interests effectively from rich historical behaviors. 

In this paper, we propose a novel model: Deep Interest Network (DIN) which tackles this challenge by ==designing a local activation unit to adaptively learn the representation of user interests from historical behaviors with respect to a certain ad==. This representation vector varies over different ads, improving the expressive ability of model greatly. Besides, we develop two techniques: ==mini-batch aware regularization== and ==data adaptive activation function== which can help training industrial deep networks with hundreds of millions of parameters. 

Experiments on two public datasets as well as an Alibaba real production dataset with over 2 billion samples demonstrate the effectiveness of proposed approaches, which achieve superior performance compared with state-of-the-art methods. DIN now has been successfully deployed in the online display advertising system in Alibaba, serving the main traffic.



点击率预估模型在广告等业务场景中至关重要，本文提出了Deep Interest Network，DIN可以基于用户历史行为，充分挖掘用户兴趣和候选广告之间的关系，从而提升CTR。DIN已经作为阿里广告业务目前的打分模型上线。除了广告领域，DIN也可以应用于其他**有丰富用户行为数据**的场景，比如电子商务中的个性化推荐、feeds流排序等。

### 一、Motivation

论文的动机是*从用户有丰富的历史行为中捕捉用户兴趣*，从而提升广告CTR。基于对阿里电商领域广告业务的实践与思考，文中指出用户兴趣有如下两个特点：

- **多样性（Diversity）** 即用户在线上购物时往往同时表现出多种兴趣，这个很好理解，例如从一个年轻妈妈的历史行为中，可以看到她的兴趣非常广泛：羊毛衫、帆布包、耳环、童装、奶粉等等。
- **局部聚焦（Local Activation）**即用户是否会点击推荐给他的某一件商品，往往是基于他之前的部分兴趣，而非所有兴趣。例如，对一个热爱游泳与吃零食的用户，==推荐给他一个泳镜，他是否会点击与他之前购买过泳裤、泳衣等行为相关，但与他之前买过冰淇淋、旺仔牛仔等行为无关==。

受到NLP里提出的Attention机制启发，Deep Interest Network (DIN) 采用来类似机制实现用户兴趣的Diversity和Local Activation。直观理解，Attention机制就是对不同特征赋予不同weight，这样某些weight高的特征便会主导这一次的预测，就好像模型对这些特征pay attention。==DIN针对当前候选广告局部地激活用户的历史兴趣，赋予和候选广告相关的历史兴趣更高的weight，从而实现Local Activation，而weight的多样性同时也实现了用户兴趣的多样性表达==。

### 二、Model

- **特征表示**

工业界的CTR预估问题往往包含大量的稀疏特征，例如大量的id类特征，NN通常会这些特征学习到对应的embeddings。上表显示了阿里广告系统使用的一系列 categorial 特征，也是本文对应使用的特征体系。值得一提的是，文中使用的全部是稀疏特征，没有使用任何组合特征。

![image-20210913114433198](D:\Notes\raw_images\image-20210913114433198.png)

- **网络结构**

DIN的网络结构并不复杂，下图描述得非常清晰。它遵循传统的Embedding&MLP框架，即先对one-hot特征做embedding，对multi-hot特征做embeddings后再做pooling得到定长的embedding，随后将它们拼接作为DNN的输入。

![image-20210913114634339](D:\Notes\raw_images\image-20210913114634339.png)

![image-20210913114541572](D:\Notes\raw_images\image-20210913114541572.png)

**Activation Unit （AU）**

==DIN的关键点在于 AU 的设计==，DIN会计算候选广告与用户最近N个历史行为商品的相关性权重weight，将其作为加权系数来对这N个行为商品的embeddings做sum pooling，用户兴趣正是由这个加权求和后的 $embedding^*$来体现。

AU 通过 weight 的多样化实现了更Diverse的用户兴趣表达。而 weight 是根据候选广告与历史行为一起决定的，==即使用户的历史行为相同，但不同的候选广告与各个行为的weight也是不同的，即用户兴趣表示也是不同的==。DIN希望通过 Activation Unit 实现 “pay attention”，即赋予和候选广告相关的历史兴趣更高的weight，从而实现 Local Activation。AU 内部是一个简单的多层网络，输入是候选广告的embedding、历史行为商品的embedding、以及两者的叉乘。

- 为什么增加叉乘作为输入呢？因为两个embedding的叉乘是显示地反映了两者之间的相关性，加入后有助于更好学习weight。
- 为什么选叉乘而非其他形式呢？其实论文的初版使用的是两个embedding的差，发表的最新版才转为使用叉乘，相信也都是经过了一系列的尝试和实验对比。
- 为什么使用简单的MLP实现AU呢？同样是尝试出来的，作者也尝试过 LSTM 结构实现AU，效果并不理想。文中给出的一个possible解释是，文本是在语法严格约束下的有序序列，而用户历史行为序列可能包含了多个同时存在的用户兴趣点，用户会在这些兴趣点之间“随意切换”，这使得这个序列并不是那么严格的“有序”，产生了一些噪声。

**目标函数**

DIN使用Logistic Loss作为目损失函数，最终的目标函数形式如下：
$$
L = - \frac 1 N \sum_{(x,y) \in S} (ylog p(x) + (1-y) log (1 - p(x)))
$$
, 其中S是训练集，N表示数据量

### 三、训练技巧

文中提出了两个技巧来辅助工业级深度网络模型的训练，同时也提升了DIN的性能。

- **Adaptive Regularization** / **Mini-batch Aware Regularization（MAR）**

在阿里的实际业务中使用深度学习的时候，作者发现深度学习可能会过拟合，尤其是在大规模的ID特征上（例如本文介绍的 good_id 特征维度就到达了6亿），当参数量非常大，模型很复杂时，非常容易过拟合。这点在下面介绍的实验部分图1可以看出来，可以看到直接使用DIN在第一轮训练后training loss就下降得很厉害，但test loss却在之后升高。

论文初版中解释，使用的特征具有显著的“长尾效应”，即很多feature id只出现了几次，只有小部分feature id出现多次，这在训练过程中增加了很多噪声，并且加重了过拟合。正则化是处理过拟合的常见技巧，但正则方法在稀疏数据深度学习上的使用，还没有一个公认的好方法。论文使用的特征具有显著的稀疏性，对绝大部分样本来说，很多特征都是0，只有很小一部分特征非0。但直接使用正则，不管特征是不是0都是要正则的，要梯度计算。对大规模的稀疏特征，参数规模也非常庞大（最大头的参数来源就是embedding），这种计算量是不可接受的。

文中提出了自适应正则，即每次mini-batch，只在非0特征对应参数上计算L2正则（针对特征稀疏性），且正则强度与特征频次有关，频次越高正则强度越低，反之越高（针对特征长尾效应）。例如，在第m次mini-batch训练，对第 j 个特征的embedding向量 $w_j$ 的梯度更新：
$$
w_j \leftarrow w_j - \eta[\frac 1 {|B_m|} \sum_{(x,y)\in{B_m}} \frac {\partial L(p(x,y))} {\partial w_j} + \lambda \frac {\alpha_{mj}}{n_j} w_j] ，
$$
其中 $B_m$ 表示第m次mini-batch的数据集， $n_j$ 表示特征 j 在整个训练集S中出现频次， $\alpha_{mj} \in \{0,1\}$ 表示$B_m$ 中是否至少有一个样本的特征 j 取值为1，正则参数 $\lambda$ 文中取0.01。

- **Data Adaptive Activation Function （Dice）**

PRelu是很常见的激活函数，其固定的折点（hard rectified point）是0：

![image-20210918174246077](D:\Notes\raw_images\image-20210918174246077.png)

其中指示函数 $P(s) = I(s>0)$.

作者把PRelu做了一个改进，让曲线中间光滑变化，光滑方式也跟数据分布有关：

<img src="D:\Notes\raw_images\image-20210918174455886.png" alt="image-20210918174455886" style="zoom:80%;" />

常数 $\epsilon$ 在文中取 $10^{-8}$ .

在训练时，$E[s]$ 和 $Var[s]$是当前mini-batch的均值和方差，在测试时，$E[s]$ 和 $Var[s]$是通过常见的moving average计算，文中moving average超参取0.99。

<img src="D:\Notes\raw_images\image-20210913142133834.png" alt="img" style="zoom:80%;" />

Dice的主要动机是==随数据分布变化动态地调整 rectified point==，虽说是动态调整，其实它也把rectified point限定在了数据均值 $E[s]$ ，实验显示对本文的应用场景Dice比PRelu效果更好。

### 四、Experiments

- **实验设置**

**数据集** 文中使用了2个公开数据集以及阿里实际业务数据集，实验中详细设置如下

<img src="D:\Notes\raw_images\image-20210913142220176.png" alt="img" style="zoom:80%;" />

1. **Amazon** 平均每个 user / goods 对应5条review，用户行为信息丰富。特征包含good_d、cate_id, *user reviewed good_id_list*, *cate_id_list* 四类*。*假设用户对n个goods产生过行为，取用户review 过的前k个good作为训练（ K <= n-2​），目标是预测第k+1个用户 *review* 的 good，测试时，将前n-1个good作为历史行为，预测最后一个 review 的 good。使用SGD+exponential decay 优化，初始学习率1，衰减率0.1，mini-batch大小32。
2. **MovieLens** 用户对电影进行 0-5 的打分，将用户打分4-5的作为正例，其他作为负例，随机将 $10^5$ 个user划分到train，剩下划分到 test，目标是预测某部电影用户是否会打出高于3的评分。特征包含 movie_id, movie_cate_id, user rated movie_id_list, movie_cate_id_list 四类，其他配置与Amazon一致。
3. **Alibaba** 取过去某2周线上产生的数据作为train，紧接着一天的数据作为test。因为数据量巨大，mini-batch大小设为5000，使用Adam+exponential decay 作为优化器，初始学习率 0.001，衰减率 0.9。

**离线指标** 使用GAUC与RelaImpr两个指标，其表达式分别如下：
$$
AUC = \frac {\sum_{i=1}^n \#impression_i \times AUC_i } {\sum_{i=1}^n \#impression_i}
$$

$$
RelaImpr = (\frac {AUC(model) - 0.5 } {AUC(base\_model) - 0.5}) * 100\%
$$

**其他设置** 实验中Embedding的维度设为12，MLP结构设为192 × 200 × 80 × 2。

- **实验结果**

在公开数据集上的实验结果如表1所示，其中BaseModel可以看成DIN结构中去除了AU设计，是传统的Embedding&MLP框架。

![image-20210913142519227](D:\Notes\raw_images\image-20210913142519227.png)

​																								表1. 公开数据集上的结果

这两个数据集上的特征维度并不高（10万左右），过拟合问题并不明显，但在Alibaba数据集中过拟合合问题非常严重，下面在以BaseModel为例，使用不同的防过拟合方法的实验结果，本文提出的MBA效果最好：

![image-20210913142624843](D:\Notes\raw_images\image-20210913142624843.png)																						图1. Alibaba数据集，BaseModel使用不同防过拟合方法

![img](D:\Notes\raw_images\image-20210913142651515.png)

​																				表2. Alibaba数据集，BaseModel使用不同防过拟合方法

加上MBA与Dice后，在Alibaba数据集上的实验结果如表3所示。此外，在2017.5-2017.6的线上AB测试中，DIN使广告CTR与RPM分别提高了10%与3.8%，目前已经成为主流量模型。

<img src="D:\Notes\raw_images\image-20210913142745226.png" alt="img" style="zoom:80%;" />

​																									表3. Alibaba数据集上的实验结果

- **可视化实验**

以一个候选广告为例，与某用户历史行为商品的weight输出如下，可以看到相关的衣服类商品的weight较高，说明 DIN 实现了用户兴趣的 **Local Activation**。

![img](D:\Notes\raw_images\image-20210913142823766.png)

此外，论文随机选取了9个类别，各类别100个商品，同类商品用同样形状表示，通过t-SNE对它们在DIN种的embedding结果进行可视化。如下图所示，特征空间中的向量地展现出很好的聚类特性。另外，图中点的颜色代表了DIN预测的某个特定用户购买这个商品的可能性，颜色越暖表示预测值越高，下图反映了**用户兴趣分布**，可看到==该用户的兴趣分布有多个峰，说明DIN捕捉到了用户兴趣的Diversity==。

![img](D:\Notes\raw_images\image-20210913142844352.png)

### **五、Conclusion**

1. 通过对实际业务的观察思考，提出了用户兴趣具有 Diversity 与 Local Activation 两个特点。
2. 提出 Deep Interest Network，DIN 从用户历史行为中挖掘用户兴趣，针对每个候选广告，使用 Activation Unit 计算其与用户历史行为商品的相关weight，有效捕捉了用户兴趣的两个特点。
3. 在模型训练优化上，提出了Dice激活函数与自适应正则，有效提升了模型性能。
4. 在公开数据集以及Alibaba实际数据集中取得了非常有效的提升。