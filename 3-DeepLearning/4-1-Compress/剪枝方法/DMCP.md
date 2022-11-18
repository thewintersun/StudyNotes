## DMCP: Differentiable Markov Channel Pruning for Neural Networks

地址：https://arxiv.org/pdf/2005.03354.pdf

作者：Shaopeng Guo, Yujie Wang, Quanquan Li, Junjie Yan

机构：纯商汤

发表：CVPR2020 Oral

代码：https://github.com/zx55/dmcp



### 摘要

Recent works imply that the channel pruning can be regarded as searching optimal sub-structure from unpruned networks. However, existing works based on this observation require training and evaluating a large number of structures, which limits their application. 

In this paper, we propose a novel ==differentiable method== for channel pruning, named Differentiable Markov Channel Pruning (DMCP), to efficiently search the optimal sub-structure. Our method is differentiable and ==can be directly optimized by gradient descent== with respect to ==standard task loss and budget regularization== (e.g. FLOPs constraint). 

作者提出了一个新奇的可微的方法来做通道剪枝，叫做可微马尔科夫通道剪枝，来搜索最优子结构。

该方法是可微的，而且可以通过梯度下降来直接进行优化，优化项为：分类的Loss， 资源预算（Flops）的正则。

In DMCP, we model the channel pruning as a Markov process, in which each state represents for retaining the corresponding channel during pruning, and transitions between states denote the pruning process. In the end, our method is able to implicitly select the proper number of channels in each layer by the Markov process with optimized transitions. 

To validate the effectiveness of our method, we perform extensive experiments on Imagenet with ResNet and MobilenetV2. Results show our method can achieve consistent improvement than state-of-the-art pruning methods in various FLOPs settings. 

![1589963547121](D:\Notes\raw_images\1589963547121.png)

With our method, MobileNetV2 has 0.1% accuracy drop with 30% FLOPs reduction： **AMC 1% drop**

the FLOPs of ResNet-50 is reduced by 44% with only 0.4% drop: **AMC Flops 20% No Drop**

在ImageNet上的实验结果：

![1591263508069](D:\Notes\raw_images\1591263508069.png)

The groups marked by * indicate the pruned model is trained by slimmable method proposed in [21] (Uniform Slim)

### 介绍

我们的方法通过将通道剪枝建模为马尔可夫过程，使其可微化。

对于每一个卷积层，$S_k$ 代表第 $k^{th}$个通道被保留， 而$S_k$ 到 $S_{k+1}$的转移（transition）表示在 $k^{th}$ 通道保留的情况下第 $(k+1)^{th}$ 个通道被保留的概率。 

注意,在我们的方法中开始状态总是$S_1$。然后, 状态 $S_k$的边际概率, 即保持第$k^{th}$ 通道的概率,可以通过转换概率的乘积来计算,也可以被看作是一个缩放系数。在网络Forward的过程中,每个缩放系数都乘以相应的通道的FeatureMap。

 transition probabilities 可以参数化表示，通过训练进行学习。 

这个过程的特性是，==如果在第L层保留k个Cout通道，它们必须是前k个通道==。

![1591259791929](D:\Notes\raw_images\1591259791929.png)

转移概率公式（从k-1个通道，到保留k个通道的概率）：

![1591327723927](D:\Notes\raw_images\1591327723927.png)

transition probabilities $P = \{ p_1, p_2, \cdot\cdot\cdot \  p_{Cout} \}$

![1591328092639](D:\Notes\raw_images\1591328092639.png)

通道数至少为1，所以k=1的时候，设置 $p_1$=1。

那么从1保留k个通道的概率，就是从1到k的 $p_i$ 的乘积。

![1591328361344](D:\Notes\raw_images\1591328361344.png)

网络结构的权重输出：输出*概率 = 新的输出。

![1591339858152](D:\Notes\raw_images\1591339858152.png)

这个操作不能直接应用在卷积层上，因为BN层会改变通道的值，所以操作在BN层之后。如下图所示。

![1590139414262](D:\Notes\raw_images\1590139414262.png)

Figure (b) is a detail illustration of the wrapped block in figure (a). The “Fuse” layer shows the incorporate details of architecture parameters $\alpha$ and outputs of unpruned networks O. 

Residual 模块的解决：结构参数共享的方式。

#### Budget Regularization

通道数预估：所有的概率之和。分别计算E(in) 和 E(out)的值。

![1591340796412](D:\Notes\raw_images\1591340796412.png)

Flops预估方法：

![1591340868456](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1591340868456.png)

这里 where groups = 1 for normal convolution and groups = E(in) for depth-wise convolution. 

$S_I$ and $S_K$ indicate input width/height and kernel width/height respectively, while $S_P$ is padding size and stride is convolution stride.

最后累加所有层的Flops数，通过梯度下降优化。

![1591341074643](D:\Notes\raw_images\1591341074643.png)

#### Loss Function

资源限制的Loss：log(预计的Flops - 目标Flops）

![1591341145390](D:\Notes\raw_images\1591341145390.png)

最终目标函数：

![1591341245855](D:\Notes\raw_images\1591341245855.png)

结构参数不加Weight Decay.

#### Training Pipeline

DMCP 首先运行几个Epochs 的 Stage1 未剪枝的网络来warm up 未剪枝网络的参数，然后循环（iteratively ）跑Stage1和Stage2 分别更新网络权重和结构参数。  

阶段1： 只更新未剪枝的参数。

这部分类似论文：Universally slimmable networks and improved training techniques。

三明治法则 - > 变体三明治法则: 越靠前的未剪枝通道更重要。

![1590139396919](D:\Notes\raw_images\1590139396919.png)

In figure (a), each rectangle represents a convolution block, e.g. Conv-BN-ReLU. Four sub-structures, represented by the blue parts of the rectangle, are sampled from the unpruned net: 

(1) the whole unpruned net (Max. Arch.), 

(2) structure with the minimum number of channels (Min. Arch.), 

(3) two structures randomly sampled by Markov process (Rand. Arch.). 

Each of these structures is forwarded independently, and the gradients in four sub-structure are accumulated to update the weights.  每个网络结构都是独立的Forward，梯度是累加所有的结构然后再进行更新。

阶段2：结构参数更新

![1591345654255](D:\Notes\raw_images\1591345654255.png)

为了进一步缩小搜索空间，我们将通道统一分成组（>=10），每个架构参数负责一组，而不是只负责一个通道。每一层都有相同数量的组。

#### Pruned Model Sampling

在DMCP训练完成后，我们得到满足给定成本约束的模型。介绍两种生成方法。第一种方法，即直接抽样法(DS)，是利用具有最优转移概率的马尔可夫过程，独立地对每一层进行抽样。我们对几种结构进行采样，只保留目标预算中的结构。第二个方法,命名预期抽样(ES),是将每个层的通道数量设置为由方程8计算的预期通道。

作者比较了这两种采样方法：DS会最终精度高一点点，但是速度比较慢。所以作者更多采用ES。

![1591347081320](D:\Notes\raw_images\1591347081320.png)