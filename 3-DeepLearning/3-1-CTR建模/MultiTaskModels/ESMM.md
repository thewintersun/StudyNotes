## Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

论文地址：https://arxiv.org/abs/1804.07931

作者：Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai

机构：阿里巴巴

发表：SIGIR-2018



### 摘要

准确估计点击后转化率 (CVR) 对于推荐和广告等行业应用中的排序系统至关重要。传统的 CVR 建模应用流行的深度学习方法并实现最先进的性能。然而，它在实践中遇到了几个特定于任务的问题，使 CVR 建模具有挑战性。

> 例如，传统的 CVR 模型使用点击印象的样本进行训练，同时利用所有印象的样本对整个空间进行推断。这会导致==样本选择偏差==问题。
>
> 此外，存在==极端数据稀疏问题==，使得模型拟合相当困难。

在本文中，我们通过充分利用用户操作的顺序模式，即印象 -> 点击 -> 转化，以全新的视角对 CVR 进行建模。所提出的整个空间多任务模型 (Entire Space Multi-task Model, ESMM)  可以通过

 i) 直接在整个空间上对 CVR 进行建模。

ii) 采用==特征表示迁移学习策略==来同时消除这两个问题。

对从淘宝推荐系统收集的数据集进行的实验表明，ESMM 明显优于竞争方法。我们还发布了此数据集的采样版本，以支持未来的研究。据我们所知，这是第一个公共数据集，其中包含用于 CVR 建模的点击和转换标签的顺序依赖性样本。



### 介绍

不同于CTR预估问题，CVR预估面临两个关键问题：

- ==Sample Selection Bias (SSB)== 转化是在点击之后才“有可能”发生的动作，传统CVR模型通常以点击数据为训练集，其中点击未转化为负例，点击并转化为正例。但是训练好的模型实际使用时，则是对整个空间的样本进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据来自不同分布，这个偏差对模型的泛化能力构成了很大挑战。
- ==Data Sparsity (DS)== 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

<img src="D:\Notes\raw_images\image-20210922120455666.png" alt="image-20210922120455666" style="zoom:80%;" />

认识到点击（CTR）、转化（CVR）、点击然后转化（CTCVR）是三个不同的任务后，我们再来看三者的关联：

<img src="D:\Notes\raw_images\image-20210922121526163.png" alt="image-20210922121526163" style="zoom:80%;" />

### 建模

仔细观察上图，留意以下几点：

> 1）共享Embedding CVR-task 和 CTR-task使用相同的特征和特征embedding，即两者从Concatenate之后才学习各自部分独享的参数；
>
> 2）隐式学习 pCVR 啥意思呢？这里 pCVR（粉色节点）仅是网络中的一个variable，没有显示的监督信号。

具体地，反映在目标函数中：

<img src="D:\Notes\raw_images\image-20210922143255277.png" alt="image-20210922143255277" style="zoom:80%;" />

即利用CTCVR和CTR的监督信息来训练网络，隐式地学习CVR，这正是ESMM的精华所在。

<img src="D:\Notes\raw_images\image-20210922121327499.png" alt="image-20210922121327499" style="zoom:80%;" />

### 实验结果

1. 对比方法：BASE——图1左部所示的CVR结构，训练集为点击集；
2. AMAN——从未点击样本中随机抽样作为负例加入点击集合；
3. OVERSAMPLING——对点击集中的正例（转化样本）过采样；
4. UNBIAS——使用rejection sampling；
5. DIVISION——分别训练CTR和CVCTR，相除得到pCVR；
6. ESMM-NS——ESMM结构中CVR与CTR部分不share embedding。
7. 上述方法/策略都使用NN结构，RELU激活函数，embedding维度为18，MLP结构为360\*200\*80\*2，Adam优化器 with  $\beta_1=0.9, \beta_2=0/999, \epsilon = 10^{-8}$ 。
8. 按时间分割，1/2数据训练，其余测试集

**衡量指标**：在点击样本上，计算CVR任务的AUC；同时，单独训练一个和BASE一样结构的CTR模型，除了ESMM类模型，其他对比方法均以pCTR*pCVR计算pCTCVR，在全部样本上计算CTCVR任务的AUC。

<img src="D:\Notes\raw_images\image-20210922143846714.png" alt="image-20210922143846714" style="zoom:80%;" />

ESMM显示了最优的效果。这里有趣的一点可以提下，ESMM是使用全部样本训练的，而CVR任务只在点击样本上测试性能，因此这个指标对ESMM来说是在biased samples上计算的，但ESMM性能还是很牛啊，说明其有很好的泛化能力。

<img src="D:\Notes\raw_images\image-20210922144511732.png" alt="image-20210922144511732" style="zoom:80%;" />

在Product数据集上，各模型在不同抽样率上的AUC曲线如图2所示，ESMM显示的稳定的优越性，曲线走势也说明了Data Sparsity的影响还是挺大的。

