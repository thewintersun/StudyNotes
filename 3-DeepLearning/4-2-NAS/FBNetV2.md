## FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions

论文地址: https://arxiv.org/abs/2004.05565

作者：Alvin Wan, Xiaoliang Dai, Peizhao Zhang, Zijian He, Yuandong Tian, Saining Xie, Bichen Wu, Matthew Yu, Tao Xu, Kan Chen, Peter Vajda, Joseph E. Gonzalez

机构：UC Berkeley, Facebook Inc.

代码：https://github.com/facebookresearch/mobile-vision

文章地址：https://zhuanlan.zhihu.com/p/132533771



### 摘要

可微神经结构搜索(DNAS)在设计最先进、高效的神经网络方面表现出了巨大的成功。然而，基于DNAS的搜索空间与其他搜索方法相比较小，因为所有候选网络层都必须显式地在内存中实例化。为了解决这个瓶颈，我们提出了一种内存和计算效率高的DNAS变体: DMaskingNAS。

该算法将搜索空间扩展到传统DNAS的10^14倍，支持在空间和通道维度( spatial and channel dimensions )上进行搜索，代价高昂: 输入分辨率和过滤器数量。我们提出了一种用于特征图重用的掩蔽机制，使内存和计算成本在搜索空间扩展时保持不变。此外，我们采用有效的形状传播来最大化每个FLOP或每个parameter的精度。

To address this bottleneck, we propose a memory and computationally efficient DNAS variant: DMaskingNAS. This algorithm expands the search space by up to 10^14× over conventional DNAS, ==supporting searches over spatial and channel dimensions== that are otherwise prohibitively expensive: ==input resolution and number of filters==. We propose ==a masking mechanism for feature map reuse==, so that memory and computational costs stay nearly constant as the search space expands. Furthermore, we employ ==effective shape propagation== to maximize per-FLOP or per-parameter accuracy. The searched FBNetV2s yield state-of-the-art performance when compared with all previous architectures. 

与所有以前的架构相比，所搜索的FBNetV2s具有最先进的性能。高达421倍更少的搜索成本，DMaskingNAS发现模型有0.9%的准确率，15%的失败比mobilenetv3 -小; 准确率和效率- b0差不多，但失败次数比效率- b0少20%。此外，我们的FBNetV2在精确度上比MobileNetV3高出2.6%，而且模型大小相当。

With up to 421× less search cost, DMaskingNAS finds models with 0.9% higher accuracy, 15% fewer FLOPs than MobileNetV3-Small; and with similar accuracy but 20% fewer FLOPs than Efficient-B0. Furthermore, our FBNetV2 outperforms MobileNetV3 by 2.6% in accuracy, with equivalent model size. 

### Motivation

首先回顾一下基于 DARTS 的这种可微分的网络结构搜索方法 (Differentiable Neural Architecture Search) ：一般是构造一个包含所有候选结构的超图 (super graph)，然后从中选一条 single path 做为最终的网络结构。

作者指出，基于 DARTS 的方法有两个缺点：

1. 搜索空间相对较小。由于要把很大的 super graph 以及 featuremap 存在 GPU 上，显存的限制就使得 DARTS 系列方法的搜索空间，比基于 RL，进化算法方法的搜索空间要小；
2. 搜索代价会随着每层的选择的增加线性增长。每在 supergraph 中增加一个新的搜索维度，显存和计算量都会大幅增长，这也就制约了搜索空间不能太大。

作者的改进方向就是在几乎不引入显存和计算量代价的情况下，相比于 FBNet，搜索空间加入了 channels 和 input resolution 两个维度，从而把搜索空间提升了将近10^14倍。怎么做到的呢？主要是两点改进：

1. ==对 supergraph 中的 channel / input resolution 选项加入 mask==
2. ==复用 supergraph 中所有选项的 feature-map==

下面进行详细的介绍。

### Channel Search

![img](https://pic2.zhimg.com/80/v2-7b056aa376890417d4bb4942690e186d_720w.jpg)

channel masking机制

把不同的 channel 加入搜索空间，之前的 DNAS 系列方法就是把不同的选项融进 supergraph，这样会带来接近 ![[公式]](https://www.zhihu.com/equation?tex=O%28N%5E%7B2%7D%29) 种选择的可能。为了减少搜索 channel 时候的计算量，作者构造了 channel masking的机制，把不同 channel 的最终输出，表征为和一个 mask 相乘的形式。

具体的做法如上图所示。右边那个灰色的长方体表示一个 shape 为 (c, h, w) 的 tensor，和左边的 mask 向量 M 相乘的结果。M 可以拆解为多个 mask ![[公式]](https://www.zhihu.com/equation?tex=m_%7B1%7D%2C+m_%7B2%7D%2C+m_%7B3%7D%2C+...) 和对应的 Gumbel Softmax 的系数 ![[公式]](https://www.zhihu.com/equation?tex=g_%7B1%7D%2C+g_%7B2%7D%2C+g_%7B3%7D%2C+...) 的乘积和。通过调节左边的 mask，就能得到等价的不同 channel 的结果。相当于对一个大的 tensor，mask 掉额外的 channel，得到相应的别的 channel 的结果。

![img](https://pic3.zhimg.com/80/v2-b4aa1786dafbcefaf971697f4d03c05e_720w.jpg)

想要进行加权和，首先就会遇到上图 Step A 中的问题：channel 不同，对应的 tensor shape 不用，无法直接相加。为了解决这个问题，可以引入 Step B 中的方法：对输出做 zero padding，使之shape 对齐（也就是图中蓝色部分），然后就可以加权和了。Step B 和 Step C 是等价的。Step C 相当于对卷积的 filter 进行 mask。随后作者又进行了一个简化的假设，假设所有的 weighting 共享，也就是 Step D 的形式。Step E 和 Step D 是等效的，即为最终的 channel masking 机制。

### Input Resolution Search

![img](https://pic2.zhimg.com/80/v2-3afd669826a05441bf1d6023625d3671_720w.jpg)

​																			spatial subsampling 机制

上面说了在 channel 维度的做法。 在 Spatial 维度的做法也是类似的，作者也想构造一种加权和的形式表征不同分辨率的 feature。如上图 A 所示，不同分辨率的 tensor 不能直接相加。图 B 说明了在边缘 padding 的方式不行，pixel 无法对齐。图 C 说明了为了让 pixel 对齐，要采用这种 Interspersing zero-padding 的形式。但是图 C 这种方式会又带来==感受野 mis-alignment 的问题==：如图 D 所示，Interspersing zero-padding 之后，一个 3x3 的 kenel 有效感受野变成了 2x2。所以图 E 才是作者最终的解决方法：和 F 运算完之后再 padding。

### Search Space

通过上述的Channel masking 和 Resolution subsampling 机制，FBNet V2 的搜索空间就可以在 channel 和 spatial 维度扩展了。FBNet V2有三个不同的系列：FBNetV2-F, FBNetV2-P, FBNetV2-L。分别对应的优化目标为Flops，参数量和大模型。下表是大模型 FBNetV2-P 的搜索空间：

![img](https://pic3.zhimg.com/80/v2-cde9541155248aba9148a815112436aa_720w.jpg)

输入为288x288x3。channel 维度的搜索体现在表格种的 number of filters f，比如说可能是 16 到 28 之间步长为 4 的一个值。spatial 维度的搜索体现在表格中的block expansion rate e，从 0.75 到 4.5，步长为 0.75。

TBS 就是指待搜的 block。这些 TBS 对应的 block 具体的搜索空间如下表：

![img](https://pic1.zhimg.com/80/v2-75bd9f27dd3910333c582bd5215c6e78_720w.jpg)

可以看出，组成的 block type 就是由 3x3 或者 5x5 的 depth-wise conv + SE + relu / hswith 组成。

这三个系列最后搜出来的结果，在 ImageNet 上的结果如下表：

![img](https://pic1.zhimg.com/80/v2-aafdc611221478ff4033d54be77c491c_720w.jpg)

![img](https://pic3.zhimg.com/80/v2-c4628f99d5702ae1ca6f07c341d15aae_720w.jpg)

同样是以 FBNetV2-P 为例子，和其他方法的对比如下图：

![img](https://pic3.zhimg.com/80/v2-19fc0972f7ddb6fd805614ad84271032_720w.jpg)

总结一下，本文的==贡献点主要在于提出了一种 channel 维度的 mask 机制 和 spatial 维度的 subsampling 机制，能扩大 DNAS 系列方法的搜索空间，同时几乎不增加显存和计算开销==。