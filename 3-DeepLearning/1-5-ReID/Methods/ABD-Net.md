### ABD-Net: Attentive but Diverse Person Re-Identification

论文地址：https://arxiv.org/abs/1908.01114

作者：Tianlong Chen, Shaojin Ding, Jingyi Xie, Ye Yuan, Wuyang Chen, Yang Yang

机构： Texas A&M University, 中科大，Walmart Technology, Wormpex AI Research

代码地址：https://github.com/TAMU-VITA/ABD-Net



#### 摘要

所以作者提出了一个既包含Attention又Diverse的网络结构，结合了 Attention 模块和 diversity regularization 到网络结构中。

1. 具体来说，提出了一对互补的Attention模块，分别关注Channel聚合和位置感知。

2. 此外，提出了一种新的有效的正交性约束（spectral value difference orthogonality， SVDO)，对特征值和权重都实施正交正则化。目的是用来降低特征的相关性，也就是更为多样化。

   

#### 介绍

##### Channel Attention Module

![1567568735708](D:\Notes\raw_images\1567568735708.png)

​	基于Attention机制的Channel Attention 模块，Channel affinity matrix X 的大小为 C*C， X 矩阵代表的是Channel i 对 Channel j的影响是多大。

![1567669997156](D:\Notes\raw_images\1567669997156.png)

最后CAM模型的输出E定义为：

![1567670097141](D:\Notes\raw_images\1567670097141.png)

##### Position Attention Module

![1567568770889](D:\Notes\raw_images\1567568770889.png)

这个模块PAM的计算方法，和CAM的区别在于，首先都会先做 [卷积-BN-ReLU ] 模块，然后再做reshape操作。然后Pixel affinity matrix S 的大小为NxN，N=WxH, 是关于像素的转换矩阵。与上面的 X 矩阵大小不同。

##### 正交性正则化

作者认为：==特征空间正交正则化(以下简称O.F.)是为了减少特征相关性，直接有利于匹配结果==。权值（Weight）的正交正则化器(O.W.)可以提高Filters的多样性[39]，提高学习能力。

许多正交性方法[34,35,36,37]，包括之前对person Re-ID[13]的研究，都对权重的正交性施加了硬约束，而权重的计算依赖于SVD。 然而，在高维矩阵上计算奇异值分解(SVD)代价昂贵，迫切需要开发软正交正则化器。

> [34] St´efan van der Walt, S. Chris Colbert, and Ga¨el Varoquaux. The numpy array: a structure for efficient numerical computation. CoRR, abs/1102.1523, 2011. 2, 3
>
> [35] Mehrtash Harandi and Basura Fernando. Generalized backpropagation,n’fEg tude de cas: Orthogonality. arXiv preprint arXiv:1611.05927, 2016. 2, 3
>
> [36] Mete Ozay and Takayuki Okatani. Optimization on submanifolds of convolution kernels in cnns, 2016. 2, 3
>
> [37] Lei Huang, Xianglong Liu, Bo Lang, Adams Wei Yu, Yongliang Wang, and Bo Li. Orthogonal weight normalization: Solution to optimization over multiple dependent stiefel manifolds in deep neural networks, 2017. 2, 3
>
> [13] Yifan Sun, Liang Zheng, Weijian Deng, and Shengjin Wang. SVDnet for pedestrian retrieval. 2017 IEEE International Conference on Computer Vision (ICCV), Oct 2017. 1, 2, 3, 5, 6, 7

Many existing soft regularizers [38, 41] restrict the Gram matrix of F to be close to an identity matrix under Frobenius norm that can avoid the SVD step while being differentiable. However, the gram matrix for an overcomplete F cannot reach identity because of rank deficiency, making those regularizers biased. [39] hence introduced the spectral norm-based regularizer that effectively alleviates the bias.

方法：对于一个特征图 M (CxHxW) 首先Reshape到 CxN  大小，N=HxW。 

![1567674311041](D:\Notes\raw_images\1567674311041.png)

##### 整体网络结构

![1567568839491](D:\Notes\raw_images\1567568839491.png)

Figure 4. Architecture of ABD-Net: O.W. is applied on all ResNet layers. O.F. is applied after CAM on res_conv_2 and after res_conv_5 in the Attentive Branch. l The feature vectors from both attentive and global branches are concatenated as the final feature embedding.

损失函数包含： a cross entropy loss, a hard mining triplet loss, and orthogonal constraints on feature (O.F.) and on weights (O.W.)

![1567672333825](D:\Notes\raw_images\1567672333825.png)

#### 实验结果

在训练过程中，输入图像的大小被调整到384 x 128，然后通过随机水平翻转，归一化，和随机擦除来增强。

训练过程分两步，首先，我们将Backbone权值冻结，只训练约简层、分类器和所有注意模块10个Epochs，只应用交叉熵损失和三重损失。然后，所有层都被释放用于另外60个Epochs的训练，并应用全部损失函数。

![1567673924801](D:\Notes\raw_images\1567673924801.png)

效果图：

![1567674114450](D:\Notes\raw_images\1567674114450.png)

目前最佳方法对比:

![1567674021490](D:\Notes\raw_images\1567674021490.png)