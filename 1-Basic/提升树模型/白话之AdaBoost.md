## 【白话机器学习】算法理论+实战之AdaBoost算法

文章地址：https://mp.weixin.qq.com/s?__biz=MzA4ODUxNjUzMQ==&mid=2247485378&idx=1&sn=e6289eafc831c281423ff9df2bfbdf94&chksm=9029b51ea75e3c086921acee6e53baa66ef4c53002409ce1958830683b3139778caf84fca3d0&scene=178&cur_album_id=1609683722293788672#rd

### 1. 写在前面

今天是白话机器学习算法理论+实战的第四篇，AdaBoost算法，这是集成方法的一种方式，通过今天的学习，快速Get到AdaBoost的原理，并最后运用AdaBoost算法实现对波士顿房价的预测。

**大纲如下**：

- AdaBoost的工作原理（三个臭皮匠，顶个诸葛亮）
- AdaBoost的实例（通过实例，迅速理解原理）
- AdaBoost的实战：对波士顿房价进行预测，并对比弟弟算法

### 2. AdaBoost ?  还是先从那句谚语开始吧！

讲AdaBoost之前，还是先了解一个故事吧：

> 小学语文课本一篇名为《三个臭皮匠顶个诸葛亮》的文章。文章中写到诸葛亮带兵过江，江水湍急，而且里面多是突出水面的礁石。普通竹筏和船只很难过去, 打头阵的船只都被水冲走触礁沉没，诸葛亮一筹莫展，也想不出好办法，==入夜来了3个做牛皮活的皮匠献策。告诉诸葛亮买牛，然后把牛从肚皮下整张剥下来，封好切口后让士兵往里吹气，做成牛皮筏子，这样的筏子不怕撞==，诸葛亮按此方法尝试并顺利过江.
>

这就是“三个臭皮匠顶个诸葛亮”的故事了，为什么先讲这个故事呢？一是怕一上来就满口官方话语，一顿数学公式的打消学习的兴趣，二是这个故事告诉了我们一个道理：**集思广益，博采众长**。这就是集成的含义。

集成的含义上面我们说了，就是集思广益，博取众长，当我们做决定的时候，我们先听取多个专家的意见，再做决定。

集成算法通常用两种：==投票选举（bagging）==和==再学习（boosting）==。

- 投票选举的场景类似把专家召集到一个会议桌前，当做一个决定的时候，让 K 个专家（K 个模型）分别进行分类（做出决定），然后选择出现次数最多的那个类（决定）作为最终的分类结果。（听说过伟大的==随机森林吧，就是训练很多棵树，少数服从多数==）
- 再学习相当于把==K 个专家（K 个分类器）进行加权融合，形成一个新的超级专家（强分类器）==，让这个超级专家做判断。（而伟大的AdaBoost就是这种方式）

> 在这里先注意下，bagging和boosting的区别吧：根据上面的描述，
>
> - Boosting 的含义是提升，它的作用是每一次训练的时候都对上一次的训练进行改进提升，在训练的过程中这 K 个“专家”之间是有依赖性的，当引入第 K 个“专家”（第 K 个分类器）的时候，实际上是对前 K-1 个专家的优化。
>- 而 bagging 在做投票选举的时候可以并行计算，也就是 ==K 个“专家”在做判断的时候是相互独立的，不存在依赖性==。
> 

### 3. AdaBoost的工作原理

AdaBoost算法是一种再学习的一种方式，英文全称是 Adaptive Boosting，中文含义是自适应提升算法。它由 Freund 等人于 1995 年提出，是对 Boosting 算法的一种实现。

什么是 Boosting 算法呢？Boosting 算法是集成算法中的一种，同时也是一类算法的总称。这类算法通过训练多个弱分类器，将它们组合成一个强分类器，也就是我们俗话说的“三个臭皮匠，顶个诸葛亮”。为什么要这么做呢？因为臭皮匠好训练，诸葛亮却不好求。因此要打造一个诸葛亮，最好的方式就是训练多个臭皮匠，然后让这些臭皮匠组合起来，这样往往可以得到很好的效果。这就是 Boosting 算法的原理。

<img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGEhnbZkTicaKwMZ2lOqqib7L0bE6biclCHZBCiccAaNSjMaAdLyzOZb11rw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

看上面这个图，我们可以用多个弱分类器来组合一个强分类器，那么就有一个问题了？怎么组合起来呢？依据是啥？看图，就会发现，这些弱分类器是根据不同的权重组合而成的。

假设弱分类器为$G_i(x)$，它在强分类器中的权重 $α_i$，那么就可以得出强分类器 f(x)：$f(x) = \sum_{i=1}^n \alpha_i G_i(x)$

看，诸葛亮就是这么来的，很多个士兵，通过重要性不同，进行加权，然后相加得出的。那么这里就有两个问题：

1. 如何得到这些弱分类器（士兵），也就是在每次迭代训练的过程中，如何得到最优的弱分类器（士兵）？
2. 每个弱分类器（士兵）的权重是如何计算的？

我们先来看一下第二个问题，如何计算权重？那第一感觉肯定是谁表现好，权重就越高啊。哈哈，还真是这样

> 实际上在一个由 K 个弱分类器中组成的强分类器中，如果弱分类器的分类效果好，那么权重应该比较大，如果弱分类器的分类效果一般，权重应该降低。所以我们需要基于这个弱分类器对样本的分类错误率来决定它的权重，用公式表示就是：
>
> $\alpha_i = \frac 1 2 log \frac {1 - e_i} {e_i}$, 其中 $e_i$代表==第 i 个分类器的分类错误率==。 
>
> 先不要管这个公式是怎么来的，只需要知道这个公式可以保证，分类器的分类错误率越高，相应的权重就越大。
>

然后我们再来看下第一个问题，如何在每次训练迭代的过程中选择最优的弱分类器？

> AdaBoost是通过改变样本的数据分布来实现的，==AdaBoost 会判断每次训练的样本是否正确分类，对于正确分类的样本，降低它的权重，对于被错误分类的样本，增加它的权重==。再基于上一次得到的分类准确率，来确定这次训练样本中每个样本的权重。然后将修改过权重的新数据集传递给下一层的分类器进行训练。==这样做的好处就是，通过每一轮训练样本的动态权重，可以让训练的焦点集中到难分类的样本上，最终得到的弱分类器的组合更容易得到更高的分类准确率==。
>

过程理解就是这样， 我的训练样本在开始的时候啊，都会有一个概率分布，也就是权重。比如n个样本，我假设每个样本的权重都是1/n，意味着同等重要， 但是我们训练出一个分类器A之后，如果这个分类器A能把之前的样本正确的分类，就说明这些正确分类的样本由A来搞定就可以了。我们下一轮训练分类器B的时候就不需要太多的关注了，让B更多的去关注A分类错误的样本？那怎么做到这一点呢？**那就把A分类正确的样本的权重减小，分类错误的样本的权重增大**。这样，B在训练的时候，就能更加的关注这些错误样本了，因为一旦把这些样本分类错误，损失就会腾腾的涨（权重大呀），为了使损失降低，B就尽可能的分类出这些A没有分出的样本，问题解决。那如果训练出来的B已经很好了，误差很小了，仍然有分不出来的怎么办？**那同样的道理，把这些的权重增大，交给下一轮的C**。每一轮的分类器各有专长的。

白话讲完了，看看怎么计算着每个样本的权重吧：

> 我们可以用 $D_{k+1}$ 代表第 k+1 轮训练中，样本的权重集合，其中 $W_{k+1,1}$ 代表第 k+1 轮中第1个样本的权重，以此类推 $W_{k+1,N}$ 代表第 k+1 轮中第 N 个样本的权重，因此用公式表示为：$D_{k+1} = (W_{k+1,1},W_{k+1,2},\cdots,W_{k+1,N})$ 第 k+1 轮中的样本权重，是根据该样本在第 k 轮的权重以及第 k 个分类器的准确率而定，具体的公式为：
>$$
> w_{k+1,i} = \frac {w_{k,i}}{z_k}exp(-\alpha_k y_iG_k(x_i)), \\ i = 1,2,\cdots,N
>$$
> 这个公式保证的就是，如果当前分类器把样本分类错误了，那么样本的w就会变大，如果分类正确了，w就会减小。 这里的$Z_k$是归一化系数。就是 $Z_k = \sum_{i=1}^N W_{k,i}exp(-\alpha_k y_i G_k(x_i))$

看到这，如果还不明白AdaBoost是怎么算的，看看下面的例子保证你神清气爽！

### 4. AdaBoost算法示例

看例子之前，我们先回忆一下，AdaBoost里面的两个问题：

1. 如何得到这些弱分类器（士兵），也就是在每次迭代训练的过程中，如何得到最优的弱分类器（士兵）？ --- 改变样本的权重或者叫数据分布
2. 每个弱分类器（士兵）的权重是如何计算的？--- 通过误差率和那个公式

好了，看下面的例子，假设有10个训练样本：

![图片](https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGe1Dd7mne69V49dwzLlk9vJSDwAEYSGm3HIBSt6jJ4pIcaRFnHziaU2A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我想通过AdaBoost构建一个强分类器（诸葛亮出来），怎么做呢？模拟一下：

- 首先，我得先给这10个样本划分重要程度，也就是权重，由于是一开始，那就平等，都是1/10。即初始权重D1=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)。假设我训练的3个基础分类器如下：

<img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGcLnq7AlhuIm3JEic2EtCz2HvGiaKFpRodrj5Io3MDD9l0NxaNnXQnMFg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

当然，这个是一次迭代训练一个，这里为了解释这个过程，先有这三个。

- 然后，我们**进行第一轮的训练**, 我们可以知道：

> 分类器 f1 的错误率为 0.3，也就是 x 取值 6、7、8 时分类错误；
>分类器 f2 的错误率为 0.4，即 x 取值 0、1、2、9 时分类错误；
> 分类器 f3 的错误率为 0.3，即 x 取值为 3、4、5 时分类错误。根据误差率最小，我训练出一个分类器来如下(选择f_1)：
> 
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGBiacNeTR6aMouXDXTJs7ACB1LfrKK7DgRW1OicDp1tSlibrE4JWtDBQvQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />
>
> 这个分类器的错误率是0.3（x取值6,  7，8的时候分类错误），是误差率最低的了（怎么训练的？可以用一个决策树训练就可以啊）， 即e1 = 0.3

- 那么根据权重公式得到第一个弱分类器的权重：

<img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGg1gcnPHskZpwta2Jz8rkIcr7hHibdRlfJw4NBOww4wtwjHUbHwou4QQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

- 然后，我们就得根据这个分类器，来更新我们的训练样本的权重了

> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeG6QwaDnyPRVxBwj2BjdStWE2iaZF8ib7yHXQpt5sic9KGElCIvv8qicXNEw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />
>
> 根据这个公式，就可以计算权重矩阵为：D_2=(0.0715, 0.0715, 0.0715, 0.0715, 0.0715, 0.0715, 0.1666, 0.1666, 0.1666, 0.0715)。
>
> 你会发现，6, 7, 8样本的权重变大了，其他的权重变小（这就意味着，下一个分类器训练的时候，重点关注6, 7, 8这三个样本，）
>

- 接着我们**进行第二轮的训练**，继续统计三个分类器的准确率，可以得到：

> 分类器 f_1 的错误率为 0.1666 * 3，也就是 x 取值为 6、7、8 时分类错误。分类器 f_2 的错误率为 0.0715 * 4，即 x 取值为 0、1、2、9 时分类错误。分类器 f_3 的错误率为 0.0715 * 3，即 x 取值 3、4、5时分类错误。
> 在这 3 个分类器中，f_3 分类器的错误率最低，因此我们选择 f_3 作为第二轮训练的最优分类器，即：
>
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGwYmsia2zurGHNjrVh5zslYfktUdnwicy8XGtBUttYITWQR5cNavYiatibg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />
>
> 根据分类器权重公式得到：
>
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGIWAZsCFMibjVyE8Via2LwO97ExMTvzg82a4DYtn9QPxhYickCPo2U6Kkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

- 同样，我们对下一轮的样本更新求权重值

> 可以得到 D_3=(0.0455,0.0455,0.0455,0.1667, 0.1667,0.01667,0.1060, 0.1060, 0.1060, 0.0455)。
>
> 你会发现， G_2分类错误的3，4， 5这三个样本的权重变大了，说明下一轮的分类器重点在上三个样本上面。
>

- 接下来**我们开始第三轮的训练**, 我们继续统计三个分类器的准确率，可以得到

> 分类器 f_1 的错误率为 0.1060 * 3，也就是 x 取值 6、7、8 时分类错误。
> 分类器 f_2 的错误率为 0.0455 * 4，即 x 取值为 0、1、2、9 时分类错误。
> 分类器 f_3 的错误率为 0.1667 * 3，即 x 取值 3、4、5 时分类错误。
> 在这 3 个分类器中，f_2 分类器的错误率最低，因此我们选择 f_2 作为第三轮训练的最优分类器，即：
>
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGibcUonicrT37za4DgRibOOFClia3gXpNrk2QvRiat2ia1apoV5YGpYFsOuTQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />
>
> 我们根据分类器权重公式得到：
>
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGXzOIPZY5HC0l7eG16BHlJVmJEokliaIHW79iar3flXcMLRd8o8Gaj3icA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

假设我们只进行 3 轮的训练，选择 3 个弱分类器，组合成一个强分类器，那么最终的强分类器

> $G(x) = 0.4236 \cdot G1(x) + 0.6496 \cdot G2(x)+0.7514 \cdot G3(x)$。
>

这样，就得到了我们想要的诸葛亮。你发现了吗？其实这个过程不难的，简单梳理就是：

1. ==确定初始样本的权重，然后训练分类器，根据误差最小，选择分类器，得到误差率，计算该分类器的权重。==
2. ==然后根据该分类器的误差去重新计算样本的权重。==
3. ==进行下一轮训练，若不停止，就重复上述过程。==

> 理解起来这其实就是一个利用敌人去使自己的士兵变强的问题，假设敌人有10个人，我这边5个人（训练5轮）。
>首先，我让这5个人分别去打那10个，选出最厉害的那一个，作为第一轮分类器， 然后10个敌人里面他能打过的，可以重要性降低，重点研究他打不过的那些人的套路。
> 
> 然后再训练，这样选出的第2个人，就可以对付一些第一个人打不过的敌人。
>同理，后面再重点研究第2个人打不过的那些人，让第3个人来打， 慢慢的下去，直到结束。
> 这样就会发现，这五个人，虽然单拿出一个来，我敌人的这10个单挑的时候，没有一个人能完胜10局，但是这5个人放在一块的组合，就可以完胜这10局。

这就是三个臭皮匠可以顶个诸葛亮的道理，诸葛亮再厉害，水平也就是单挑10局可以完胜这10局，而我用普通的五个小士兵，经过5轮训练，这个组合也可以完胜10局，而后者的培养成本远远比一个诸葛亮的培养成本低的多的多。

这也就是AdaBoost的思想核心啦。

### 5. AdaBoost实战：预测房价

懂了算法的原理之后，关键的还是实战呐。首先知道如何使用AdaBoost工具。

#### **5.1 sklearn的AdaBoost工具**

我们可以直接在 sklearn 中使用 AdaBoost。如果我们要用 AdaBoost 进行分类，需要在使用前引用代码：

```python
from sklearn.ensemble import AdaBoostClassifier
```

如果你看到了 Classifier 这个类，一般都会对应着 Regressor 类。AdaBoost 也不例外，回归工具包的引用代码如下：

```python
from sklearn.ensemble import AdaBoostRegressor
```

下面介绍一下创建AdaBoost分类器：

- 分类的时候，需要这样的函数：

```python
AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
```

下面看看这些参数的含义：

> 1. base_estimator：代表的是弱分类器。在 AdaBoost 的分类器和回归器中都有这个参数，==在 AdaBoost 中默认使用的是决策树==，一般我们不需要修改这个参数，当然你也可以指定具体的分类器。
>2. n_estimators：==算法的最大迭代次数，也是分类器的个数==，每一次迭代都会引入一个新的弱分类器来增加原有的分类器的组合能力。==默认是 50==。
> 3. learning_rate：代表学习率，取值在 0-1 之间，默认是 1.0。==如果学习率较小，就需要比较多的迭代次数才能收敛，也就是说学习率和迭代次数是有相关性的。当你调整 learning_rate 的时候，往往也需要调整 n_estimators 这个参数==。
> 4. algorithm：代表我们要采用哪种 boosting 算法，一共有两种选择：SAMME 和 SAMME.R。==默认是 SAMME.R==。这两者之间的区别在于对弱分类权重的计算方式不同。
> 5. random_state：代表随机数种子的设置，默认是 None。随机种子是用来控制随机模式的，当随机种子取了一个值，也就确定了一种随机规则，其他人取这个值可以得到同样的结果。如果不设置随机种子，每次得到的随机数也就不同。
> 

- 如何创建AdaBoost回归呢？

```python
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss=‘linear’, random_state=None)
```

> 回归和分类的参数基本是一致的，不同点在于回归算法里没有 algorithm 这个参数，但多了一个 loss 参数。
>loss 代表损失函数的设置，一共有 3 种选择，分别为==linear、square 和 exponential，它们的含义分别是线性、平方和指数。默认是线性。== 一般采用线性就可以得到不错的效果。

创建好 AdaBoost 分类器或回归器之后，我们就可以输入训练集对它进行训练。

- 我们使用 fit 函数，传入训练集中的样本特征值 train_X 和结果 train_y，模型会自动拟合。
- 使用 predict 函数进行预测，传入测试集中的样本特征值 test_X，然后就可以得到预测结果。

#### **5.2 如何对AdaBoost对房价进行预测**

我们使用sklearn自带的波士顿房价数据集，用AdaBoost对房价进行预测:

- 首先是数据集


> 这个数据集一共包括了 506 条房屋信息数据，每一条数据都包括了 13 个指标，以及一个房屋价位。
>13 个指标的含义，可以参考下面的表格：
> 
> <img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeGMU5RTkEHNqnunRCibYyogpYib7zBzYMXp8FBl4CfncZxes8ukYIHhTXQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

处理思路（还是之前的处理套路）：

> 首先加载数据，将数据分割成训练集和测试集，然后创建 AdaBoost 回归模型，传入训练集数据进行拟合，再传入测试集数据进行预测，就可以得到预测结果。最后将预测的结果与实际结果进行对比，得到两者之间的误差。
>

代码如下：

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
# 加载数据
data=load_boston()
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)
# 使用AdaBoost回归模型
regressor=AdaBoostRegressor()
regressor.fit(train_x,train_y)
pred_y = regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("房价预测结果 ", pred_y)
print("均方误差 = ",round(mse,2))
```

运行结果：

```
房价预测结果  
[20.210.413793114.6382022517.8032258124.5893129821.25076923
 27.5222222217.837209331.7964285720.8642857127.8743169431.09142857
 12.8166666724.1313131312.8166666724.5893129817.8032258117.66333333
 27.8324.5893129817.6633333320.9082352920.1055555620.90823529
 28.2087719320.1055555621.1688212924.5893129813.2761904831.09142857
 17.0809523826.192173919.97521.0340425526.7458333331.09142857
 25.8396039611.85937513.3823529424.5893129814.9793103414.46699029
 30.1277777817.6633333326.1921739120.1020618617.7054054118.45909091
 26.1921739120.1055555617.6633333333.3102564114.9793103417.70540541
 24.6442105320.9082352925.8396039617.0809523824.5893129821.43571429
 19.3161764716.3373333346.0488888921.2507692317.0809523825.83960396
 24.6442105311.8147058817.8032258127.6363636423.5973118317.94444444
 17.6633333327.725388620.2146551746.0488888914.979310349.975
 17.0809523824.1313131321.0340425513.411.85937526.19214286
 21.2507692321.0340425547.1139534916.3373333343.2111111131.65730337
 30.1277777820.1055555617.837209318.4083333314.9793103433.31025641
 24.5893129822.8881355918.2717948717.8032258114.6382022521.16882129
 26.9153846224.6442105313.0514.979310349.97526.19217391
 12.8166666726.1921428649.4651162813.2761904817.7054054125.83960396
 31.0914285724.1313131321.2507692321.0340425526.9153846221.03404255
 21.1688212917.837209312.8166666721.0340425521.0340425517.08095238
 45.16666667]
均方误差 =  18.05
```

我们下面对比一下弟弟的表现（决策树和KNN）

```python
# 使用决策树回归模型
dec_regressor=DecisionTreeRegressor()
dec_regressor.fit(train_x,train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("决策树均方误差 = ",round(mse,2))
# 使用KNN回归模型
knn_regressor=KNeighborsRegressor()
knn_regressor.fit(train_x,train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN均方误差 = ",round(mse,2))
```

运行结果：

```
决策树均方误差 =  23.84
KNN均方误差 =  27.87
```

这里就会发现，AdaBoost 的均方误差更小，也就是结果更优。虽然 AdaBoost 使用了弱分类器，但是通过 50 个甚至更多的弱分类器组合起来而形成的强分类器，在很多情况下结果都优于其他算法。因此 AdaBoost 也是常用的分类和回归算法之一。

#### **5.3 AdaBoost与决策树模型的比较**

在 sklearn 中 AdaBoost 默认采用的是决策树模型，我们可以随机生成一些数据，然后对比下 AdaBoost 中的弱分类器（也就是决策树弱分类器）、决策树分类器 和 AdaBoost 模型在分类准确率上的表现。

> 如果想要随机生成数据，我们可以==使用 sklearn 中的 make_hastie_10_2 函数生成二分类数据==。假设我们生成 12000 个数据，取前 2000 个作为测试集，其余作为训练集。
>

下面我们直接看代码和结果，再体验一波AdaBoost的强大：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier
# 设置AdaBoost迭代次数
n_estimators=200
# 使用
X,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
# 从12000个数据中取前2000行作为测试集，其余作为训练集
train_x, train_y = X[2000:],y[2000:]
test_x, test_y = X[:2000],y[:2000]
# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
dt_stump.fit(train_x, train_y)
dt_stump_err = 1.0-dt_stump.score(test_x, test_y)
# 决策树分类器
dt = DecisionTreeClassifier()
dt.fit(train_x,  train_y)
dt_err = 1.0-dt.score(test_x, test_y)
# AdaBoost分类器
ada = AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
ada.fit(train_x,  train_y)
# 三个分类器的错误率可视化
fig = plt.figure()
# 设置plt正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
ax = fig.add_subplot(111)
ax.plot([1,n_estimators],[dt_stump_err]*2, 'k-', label=u'决策树弱分类器 错误率')
ax.plot([1,n_estimators],[dt_err]*2,'k--', label=u'决策树模型 错误率')
ada_err = np.zeros((n_estimators,))
# 遍历每次迭代的结果 i为迭代次数, pred_y为预测结果
for i,pred_y in enumerate(ada.staged_predict(test_x)):
     # 统计错误率
    ada_err[i]=zero_one_loss(pred_y, test_y)
# 绘制每次迭代的AdaBoost错误率
ax.plot(np.arange(n_estimators)+1, ada_err, label='AdaBoost Test 错误率', color='orange')
ax.set_xlabel('迭代次数')
ax.set_ylabel('错误率')
leg=ax.legend(loc='upper right',fancybox=True)
plt.show()
```

运行结果：

<img src="https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqDmxGldNoccfLZOHTy5qeG3MMDicgrc8c6Xm29NRic9EKktxPdBU91OKFibswRXax3Cfo4SibXtiaxWGQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

从图中你能看出来，弱分类器的错误率最高，只比随机分类结果略好，准确率稍微大于 50%。决策树模型的错误率明显要低很多。而 AdaBoost 模型在迭代次数超过 25 次之后，错误率有了明显下降，经过 125 次迭代之后错误率的变化形势趋于平缓。

因此我们能看出，虽然单独的一个决策树弱分类器效果不好，但是多个决策树弱分类器组合起来形成的 AdaBoost 分类器，分类效果要好于决策树模型。

### 6. 总结

今天，学习了AdaBoost算法，从集成到AdaBoost的原理到最后的小实战，全都过了一遍，通过今天的学习，我们会发现，集成算法的强大和成本小。现在很多应用都使用的集成技术，AdaBoost现在用的不多了，无论是打比赛还是日常应用，都喜欢用xgboost，lightgbm，catboost这些算法了。当然，虽然学习的深入，这些算法肯定也会大白话出来。但是出来之前，还是先搞懂AdaBoost的原理吧，这样也好对比，而对比，印象也就越深刻。
