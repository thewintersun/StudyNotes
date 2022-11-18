### Deep Layer Aggregation

论文地址：https://arxiv.org/abs/1707.06484

作者：Fisher Yu, Dequan Wang, Evan Shelhamer, Trevor Darrell

机构：UC Berkeley

发表：CVPR2018

代码地址：https://github.com/ucbdrive/dla

介绍文章：https://zhuanlan.zhihu.com/p/28563169



### 摘要

Visual recognition requires rich representations that span levels from low to high, scales from small to large, and resolutions from fine to coarse. Even with the depth of features in a convolutional network, a layer in isolation is not enough: compounding and aggregating these representations improves inference of what and where. Architectural efforts are exploring many dimensions for network backbones, designing deeper or wider architectures, but how to best aggregate layers and blocks across a network deserves
further attention. Although skip connections have been incorporated to combine layers, these connections have been “shallow” themselves, and only fuse by simple, one-step operations. We augment standard architectures with deeper aggregation to better fuse information across layers. ==Our deep layer aggregation structures iteratively and hierarchically merge the feature hierarchy to make networks with better accuracy and fewer parameters==. Experiments across architectures and tasks show that deep layer aggregation
improves recognition and resolution compared to existing branching and merging schemes.

T2T/T2T/Placeholder:0", shape=(1, ?)

### 介绍

一个CNN是由多个conv block组成，最简单的conv block由conv层+非线性层组成。其他的conv block有如下几种（不完全枚举）：

![img](https://pic4.zhimg.com/80/v2-c3a2d4b6ef096225cfa59945f5d32707_hd.png)

上图中方框里的标注，第一个表示输出通道，中间表示卷积核尺寸，最后表示输入通道。(a)和(b)来自何恺明的ResNet，(c)来自CVPR2017的文章《Aggregated residual transformations for deep neural networks》。

![1578726878440](D:\Notes\raw_images\1578726878440.png)

Figure 1: Deep layer aggregation unifies semantic and spatial fusion to better capture what and where. Our aggregation architectures encompass and extend densely connected networks and feature pyramid networks with hierarchical and iterative skip connections that deepen the representation and refine resolution.

连续的几个conv block可以组成一个subnetwork。要怎么来划分subnetwork？普遍的做法是按分辨率来划分，如ResNet101的res1~res5 block。

这些conv block一个接着一个，只在最后得到prob map。那么前面的block或者subnetwork的输出特征呢？如果能利用上，那岂不是锦上添花？当然，在这篇论文之前就已经有各类研究在做各个层的融合了，但都是“shallow aggregation”，如下图(b)。

![1578726333043](D:\Notes\raw_images\1578726333043.png)

Figure 2: Different approaches to aggregation.

 (a) composes blocks without aggregation as is the default for classification and regression networks. 

(b) combines parts of the network with skip connections, as is commonly used for tasks like segmentation and detection, but does so only shallowly by merging earlier parts in a single step each. 

We propose two deep aggregation architectures: 

(c) aggregates iteratively by reordering the skip connections of (b) such that the shallowest parts are aggregated the most for further processing and 

(d) aggregates hierarchically through a tree structure of blocks to better span the feature hierarchy of the network across different depths. 

(e) and (f) are refinements of (d) that deepen aggregation by routing intermediate aggregations back into the network and improve efficiency by merging successive aggregations at the same depth. 

Our experiments show the advantages of (c) and (f) for recognition and resolution.

(b)比较常见的，逐级上采还原，如U-Net。但是，(b)这种结构，梯度反向传播经过一个聚合点便能传回到第一个subnetwork，所以称为“shallow aggregation”。

论文提出“deep layer aggregation”（DLA），有两种：(c) iterative deep aggregation(IDA)和 (d) hierarchical deep aggregation(HDA)。

IDA如(c)所示，逐级融合各个subnetwork的特征的方向和(b)是相反的，先从靠近输入的subnetwork引出特征，再逐步聚合深层的特征。这样，梯度反向传导时再也不能仅经过一个聚合点了。上图 (b)相当于对浅层加了监督，不利于优化，DLA就避免了此问题。

IDA是针对subnetwork的，而HDA 则是针对conv block。(d) 每个block只接收上一个block传过来的feature，为HDA的基本结构；(e)block有融合前面block的feature，为HDA的变体；(f) 也是一种变体，但减少了聚合点。

上文提到了很多次聚合点，在论文里它是怎样的一种结构？如下：

![img](https://pic3.zhimg.com/80/v2-480d57e2ff2f61fda6925803857e337e_hd.png)

(b)普通的三输入的聚合点；(c)引入了残差结构，为了更好的进行梯度传导。

作者在分类和分割两类任务做了验证实验。从结果上来看，效果还是比较好的。

### 实验结果

![1578726713111](D:\Notes\raw_images\1578726713111.png)

