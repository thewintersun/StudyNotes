## MobileNet V3

论文题目：Searching for MobileNetV3 

论文地址：[https://arxiv.org/pdf/1905.02244.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1905.02244.pdf)

机构：Google AI

文章来源：https://zhuanlan.zhihu.com/p/70703846



MobileNetV3，是谷歌在2019年3月21日提出的网络架构。首先，引入眼帘的是这篇文章的标题，“searching”一词就把V3的论文的核心观点展示了出来——用**神经结构搜索（NAS）**来完成V3。虽然本人没有接触过NAS，但是我已经闻到了金钱的味道。由于真的没有接触过NAS，所以V3就讲讲其他的，除NAS之外的东西吧。先上结果：

![img](https://pic2.zhimg.com/80/v2-2b70050de6f762ad8a5f5636dc38f481_hd.jpg)

可以看到，在同一大小的计算量下，V3在ImageNet上的结果都是最好的。我们先来看看V3做了什么？

### MobileNetV3的相关技术

- 网络的架构基于NAS实现的MnasNet（效果比MobileNetV2好）
- 引入MobileNetV1的深度可分离卷积
- 引入MobileNetV2的具有线性瓶颈的倒残差结构
- 引入基于squeeze and excitation结构的轻量级注意力模型(SE)
- 使用了一种新的激活函数h-swish(x)
- 网络结构搜索中，结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt
- 修改了MobileNetV2网络端部最后阶段

### Squeeze-and-Excitation Networks

SENet由一些列SE block组成，一个SE block的过程分为Squeeze（压缩）和Excitation（激发）两个步骤。其中Squeeze通过在Feature Map层上执行Global Average Pooling得到当前Feature Map的全局压缩特征向量，Excitation通过两层全连接得到Feature Map中每个通道的权值，并将加权后的Feature Map作为下一层网络的输入。从上面的分析中我们可以看出SE block只依赖与当前的一组Feature Map，因此可以非常容易的嵌入到几乎现在所有的卷积网络中。论文中给出了在当时state-of-the-art的Inception和残差网络插入SE block后的实验结果，效果提升显著。


### 激活函数h-swish

### swish

h-swish是基于swish的改进，swish最早是在谷歌大脑2017的论文[Searching for Activation functions](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.05941) 所提出。是搜索出来的激活函数。

![img](https://pic2.zhimg.com/80/v2-6db0add6ef6230d6b0223ea5678530cd_hd.jpg)

swish论文的作者认为，==Swish具备无上界有下界、平滑、非单调的特性==。并且==Swish在深层模型上的效果优于ReLU==。仅仅使用Swish单元替换ReLU就能把MobileNet,NASNetA在 ImageNet上的top-1分类准确率提高0.9%，Inception-ResNet-v的分类准确率提高0.6%。

V3也利用swish当作为ReLU的替代时，它可以显著提高神经网络的精度。但是呢，==作者认为这种非线性激活函数虽然提高了精度，但在嵌入式环境中，是有不少的成本的。原因就是在移动设备上计算sigmoid函数是非常不明智的选择==。所以提出了h-swish。

### h-swish

可以==用一个近似函数来逼急这个swish，让swish变得硬(hard)==。作者选择的是基于ReLU6，作者认为几乎所有的软件和硬件框架上都可以使用ReLU6的优化实现。其次，它能在特定模式下消除了由于近似sigmoid的不同实现而带来的潜在的数值精度损失。

![img](https://pic4.zhimg.com/80/v2-48f15917f813fa435d6268489a13977f_hd.jpg)

下图是Sigmoid和swish的hard、soft形式：

![img](https://pic3.zhimg.com/80/v2-eebd40a2dcd533d203ef2914f7fe501a_hd.jpg)



我们可以简单的认为，hard形式是soft形式的低精度化。作者认为swish的表现和其他非线性相比，能够将过滤器的数量减少到16个的同时保持与使用ReLU或swish的32个过滤器相同的精度，这节省了3毫秒的时间和1000万MAdds的计算量。

并且同时，作者认为==随着网络的深入，应用非线性激活函数的成本会降低，能够更好的减少参数量==。作者发现swish的大多数好处都是通过在更深的层中使用它们实现的。因此，**在V3的架构中，只在模型的后半部分使用h-swish(HS)**。

### 网络结构搜索NAS

主要结合两种技术：**资源受限的NAS（platform-aware NAS）**与 **NetAdapt**。

**资源受限的NAS**，用于在计算和参数量受限的前提下搜索网络来优化各个块（block），所以称之为**模块级搜索（Block-wise Search）** 。

**NetAdapt**，用于对各个模块确定之后网络层的微调每一层的卷积核数量，所以称之为**层级搜索（Layer-wise Search）**。

一旦通过体系结构搜索找到模型，我们就会发现最后一些层以及一些早期层计算代价比较高昂。于是作者决定对这些架构进行一些修改，以减少这些慢层(slow layers)的延迟，同时保持准确性。这些修改显然超出了当前搜索的范围。

### 对V2最后阶段的修改

作者认为，当前模型是基于V2模型中的**倒残差结构**和相应的变体（如下图）。==使用**1×1卷积**来构建最后层，这样可以便于拓展到更高维的特征空间。这样做的好处是，在预测时，有更多更丰富的特征来满足预测，但是同时也引入了额外的计算成本与延时==。

![img](https://pic1.zhimg.com/80/v2-bd723b14a2a6f27f5e2705f701f3acac_hd.jpg)

所以，需要改进的地方就是要**保留高维特征的前提下减小延时**。首先，还是将1×1层放在到最终平均池之后。这样的话最后一组特征现在不是7x7（下图V2结构红框），而是以1x1计算（下图V3结构黄框）。

![img](https://pic2.zhimg.com/80/v2-d025a7f5d607874c8aaa25fec172cd35_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-fdb67f911587a9c69a2ce35c7c771650_hd.jpg)

这样的好处是，**在计算和延迟方面，特征的计算几乎是免费的**。最终，重新设计完的结构如下：

![img](https://pic4.zhimg.com/80/v2-a0d7d1e9a080f77e64d6737a6d793e77_hd.jpg)

在不会造成精度损失的同时，减少10ms耗时，提速15%，减小了30m的MAdd操作。

### V3的Block

综合以上，V3的block结构如下所示：



![img](https://pic3.zhimg.com/80/v2-3c6a66ab35022a2423639143f3a999be_hd.jpg)

与V2的block相比较：

![img](https://pic1.zhimg.com/80/v2-3b180e852d2362d9cc65343084836e6c_hd.jpg)



### MobileNetV3的网络结构

MobileNetV3定义了两个模型: **MobileNetV3-Large**和**MobileNetV3-Small**。V3-Large是针对**高资源**情况下的使用，相应的，V3-small就是针对**低资源**情况下的使用。两者都是基于之前的简单讨论的NAS。

#### MobileNetV3-Large

![img](https://pic2.zhimg.com/80/v2-9bc09dc0561fd0e257f9ed89a84e4165_hd.jpg)

#### MobileNetV3-Small

![img](https://pic3.zhimg.com/80/v2-1cf7926727b238557cfeb6aa284b7336_hd.jpg)

就像之前所说的：==只有在更深层次使用h-swish才能得到比较大的好处。所以在上面的网络模型中，不论大小，作者只在模型的后半部分使用h-swish==。

用谷歌pixel 1/2/3来对大小V3进行测试的结果。

![img](https://pic3.zhimg.com/80/v2-0afa10b5e1541bf86e5aa0316193e44e_hd.jpg)

### 实验结果

#### **Image Classification**

![img](https://pic4.zhimg.com/80/v2-a863abb81dd4bb65aa916b92290aae97_hd.jpg)

#### **Detection**![img](https://pic3.zhimg.com/80/v2-50cbd9f97e7fd71cbf1df1b17d91cd7a_hd.jpg)

#### Semantic Segmentation

![img](https://pic1.zhimg.com/80/v2-1bcd243060f2cc2ac4e9e6ac84f40428_hd.jpg)

![img](https://pic2.zhimg.com/80/v2-eacde59bd8778e3010e7c92e6ad68b51_hd.jpg)

感觉实验结果没什么好说的。对了，有一点值得说一下，训练V3用的是4x4 TPU Pod，batch size 4096 (留下了贫穷的泪水)

