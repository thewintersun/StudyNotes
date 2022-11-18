## Wide & Deep Learning for Recommender Systems

论文地址：https://arxiv.org/abs/1606.07792

作者：Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah

机构：Google

文章地址: https://zhuanlan.zhihu.com/p/132708525

开源代码：https://github.com/kaitolucifer



### 摘要

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of ==feature interactions== through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the ==user-item interactions are sparse and high-rank==. 

In this paper, we present Wide & Deep learning---==jointly trained wide linear models and deep neural networks==---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

在这里有两个词非常重要，一个是记忆，一个是泛化。“记忆能力”可以被理解为模型直接学习并利用历史数据中物品或者特征的“共现频率”的能力。“泛化能力”可以被理解为模型传递特征的相关性，以及发掘稀疏甚至从未出现过的稀有特征与最终标签相关性的能力。



### 介绍

wide&deep框架，同时训练带有Embedding的前馈神经网络和带有特征变换的线形模型，通用于稀疏输入的推荐系统。

![image-20210922154631635](D:\Notes\raw_images\image-20210922154631635.png)

**Wide部分**

Wide部分的作用是让模型具有较强的“记忆能力”。“记忆能力”可以被理解为模型直接学习并利用历史数据中物品或者特征的“共现频率”的能力。一般来说，协同过滤、逻辑回归等简单模型有较强的“记忆能力”。由于这类模型的结构简单，原始数据往往可以直接影响推荐结果，产生类似于“如果点击过A，就推荐B”这类规则式的推荐，这就相当于模型直接记住了历史数据的分布特点，并利用这些记忆进行推荐。

假设在Google Play推荐模型的训练过程中，设置如下组合特征：｛user_installed*_*app=netflix, impression_app=pandora｝(简称netflix&pandora),它代表用户已经安装了netflix这款应用，而且曾在应用商店中看到过pandora这款应用。如果以“最终是否安装pandora”为数据标签(label), 则可以轻易地统计出netflix&pandora这个特征和安装pandora这个标签之间的共现频率。假设二者的共现频率高达10%（全局的平均应用安装率为1%），这个特征如此之强，以至于在设计模型时，希望模型一发现有这个特征，就推荐pandora这款应用（像一个深刻的记忆点一样印在脑海中），这就是所谓的模型的“记忆能力”。像逻辑回归这类简单模型，如果发现这样的“强特征”，则其相应的权重就会在模型训练过程中被调整得非常大，这样就实现了对这个特征的直接记忆。

**Deep部分**

Deep部分的主要作用是让模型具有“泛化能力”。“泛化能力”可以被理解为模型传递特征的相关性，以及发掘稀疏甚至从未出现过的稀有特征与最终标签相关性的能力。深度神经网络通过特征的多次自动组合，可以深度发掘数据中潜在的模式，即使是非常稀疏的特征向量输入，也能得到较稳定平滑的推荐概率，这就是简单模型所缺乏的“泛化能力”。

**Wide部分和Deep部分的结合**

wide的部分和deep的部分==使用其输出对数几率的加权和作为预测==，然后将其输入到联合训练的一个共同的逻辑损失函数。注意到这里的联合训练和集成学习是有区别的。集成学习中，每个模型是独立训练的，而且他们的预测是在推理时合并而不是在训练时合并。相比之下，==联合训练在训练时同时考虑wide和deep模型以及加权和来优化所有参数==。这对模型大小也有影响：对于集成学习而言，由于训练是独立的，因此每个模型的大小通常会更大（例如：更多特征和交叉特征）来实现一个集成模型合理的精确度。相比之下，在联合训练中，wide部分只需要通过少量的cross-product特征变换来补充深度模型的不足，而且不是全量的模型。

wide和deep模型的联合训练是通过使用小批量随机优化同时将输出的梯度反向传播到模型的wide和deep部分来完成的。 在实验中，我们使用带L1正则的FTRL算法作为wide部分的优化器，AdaGrad作为deep部分的优化器。

这个联合模型如图1（中）所示。对于逻辑回归问题，模型的预测是：

![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3D1+%7C+%5Cmathbf%7Bx%7D%29%3D%5Csigma%5Cleft%28%5Cmathbf%7Bw%7D_%7B%5Ctext+%7Bwide%7D%7D%5E%7BT%7D%5B%5Cmathbf%7Bx%7D%2C+%5Cphi%28%5Cmathbf%7Bx%7D%29%5D%2B%5Cmathbf%7Bw%7D_%7B%5Ctext+%7Bdeep%7D%7D%5E%7BT%7D+a%5E%7B%5Cleft%28l_%7Bf%7D%5Cright%29%7D%2Bb%5Cright%29)

其中，Y是二值分类标签， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28.%29) 是sigmoid函数， ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28x%29) 是原始特征x的跨产品变换，b是偏置项， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bwide%7D) 是wide模型的权重向量， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bdeep%7D) 是用于最终激活函数 ![[公式]](https://www.zhihu.com/equation?tex=a%5E%7B%28l_f%29%7D) 的权重。

Wide&Deep模型把单输入层的Wide部分与由Embedding层和多隐层组成的Deep部分连接起来，一起输入最终的输出层。单层的Wide部分善于处理大量稀疏的id类特征；Deep部分利用神经网络表达能力强的特点，进行深层的特征交叉，挖掘藏在特征背后的数据模式。最终，利用逻辑回归模型，输出层将Wide部分和Deep部分组合起来，形成统一的模型。



### 实验结果

实验场景 Google Play商店的app推荐中，当一个user访问Google Play，会生成一个包含user和contextual信息的query，推荐系统的精排模型会对于候选池中召回的一系列apps（即item，文中也称 impression）进行打分，按打分生成app的排序列表返回给用户。Deep&Wide对应这里的精排模型，输入$x$ 包括<user，contextual，impression>的信息，$y=1$ 表示用户下载了impression app，打分即 $p(y|x)$ 。

**实验细节**

- 训练样本约5000亿
- Categorical 特征（sparse）会有一个过滤阈值，即至少在训练集中出现m次才会被加入。
- Continuous 特征（dense）通过CDF被归一化到 [0,1] 之间。
- Categorical 特征映射到32维embeddings，和原始Continuous特征共1200维作为NN输入。
- Wide部分只用了一组特征叉乘，即被推荐的app ☓ 用户下载的app。
- 线上模型更新时，通过“热启动”重训练，即使用上次的embeddings和模型参数初始化。

Wide部分设置很有意思，作者为什么这么做呢？结合业务思考，在Google Play商店的app下载中，不断有新的app推出，并且有很多“非常冷门、小众”的app，而现在的智能手机user几乎全部会安装一系列必要的app。联想前面对Memorization和Generalization的介绍，此时的Deep部分无法很好的为这些app学到有效的embeddding，而这时Wide可以发挥了它“记忆”的优势，作者在这里选择了“记忆”user下载的app与被推荐的app之间的相关性，有点类似“装个这个app后还可能会装什么”。对于Wide来说，它现在的任务是弥补Deep的缺陷，其他大部分的活就交给Deep了，所以这时的Wide相比单独Wide也显得非常“轻量级”，这也是Join相对于Ensemble的优势。

实验的Depp&Wide模型结构如下：

![image-20210922155529360](D:\Notes\raw_images\image-20210922155529360.png)

上图展现了Google Play的推荐团队对业务场景的深刻理解。下图中可以详细地了解到Wide&Deep模型到底将哪些特征作为Deep部分的输入，将哪些特征作为Wide部分的输入。Deep部分的输入是全量的特征向量，包括用户年龄(Age)、已安装应用数量(#App Installs)、设备类型(Device Class)、已安装应用（User Installed App）、曝光应用(Impression App)等特征。已安装应用、曝光应用等类别特征，需要经过Embedding层输入连接层，拼接成1200维的Embedding向量，再经过3层ReLU全连接层，最终输入LogLoss输出层。

**实验结果** 通过3周的线上A/B实验，实验结果如下，其中Acquisition表示下载。

<img src="D:\Notes\raw_images\image-20210922201751455.png" alt="image-20210922201751455" style="zoom:80%;" />

