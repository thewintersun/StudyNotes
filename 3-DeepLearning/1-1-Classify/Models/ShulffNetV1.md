## ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

论文地址：https://arxiv.org/abs/1707.01083

作者：Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun

机构：旷视

文章来源：https://zhuanlan.zhihu.com/p/51566209



### 摘要

> We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, ==pointwise group convolution== and ==channel shuffle==, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ~13x actual speedup over AlexNet while maintaining comparable accuracy.



**实验结果**

![1598861313657](D:\Notes\raw_images\1598861313657.png)



### 介绍

We propose using ==pointwise group convolutions== to reduce computation complexity of 1 x 1 convolutions.
To overcome the side effects brought by group convolutions, we come up with ==a novel channel shuffle operation== to help the information flowing across feature channels. 

Based on the two techniques, we build a highly efficient architecture called ShuffleNet.

在ResNeXt中DW只应用在3X3的卷积中，在非常小的网络中1x1的卷积也是很大的消耗，所以对1x1卷积做了Group操作，并且做了ChannelShuffle来解决Group操作带来的问题。

在ResNeXt的文章中，分组卷积作为传统卷积核深度可分离卷积的一种折中方案被采用。这时大量的对于整个Feature Map的Pointwise卷积成为了ResNeXt的性能瓶颈。一种更高效的策略是在组内进行Pointwise卷积，但是这种组内Pointwise卷积的形式不利于通道之间的信息流通，为了解决这个问题，ShuffleNet v1中提出了通道洗牌（channel shuffle）操作。假设分组Feature Map的尺寸为 $w\times h \times c_1$ ，把 $c_1 = g\times n$ ，其中 g 表示分组的组数。Channel Shuffle的操作细节如下：

- 将Feature Map展开成 $g\times n\times w\times h$ 的四维矩阵（为了简单理解，我们把 $w\times h$ 降到一维，表示为s）；
- 沿着尺寸为 $g\times n\times s$ 的矩阵的 g 轴和 n 轴进行转置；
- 将 g 轴和 n 轴进行平铺后得到洗牌之后的Feature Map；
- 进行组内 $1\times1$ 卷积。

shuffle的结果如图1.(c)所示，具体操作细节示意图见图2。

![img](https://picb.zhimg.com/80/v2-ea1b29e7ea3609d1001b14faa1ba3547_720w.jpg)

​																		图2：通道洗牌过程详解

从代码中我们也可以看出，channel shuffle的操作是步步可微分的，因此可以嵌入到卷积网络中。   



![1598862197512](D:\Notes\raw_images\1598862197512.png)

Figure 2. ShuffleNet Units. a) bottleneck unit [9] with depthwise convolution (DWConv) [3, 12]; b) ShuffleNet unit with pointwise group convolution (GConv) and channel shuffle; c) ShuffleNet unit with stride = 2.

介绍了ShuffleNet v1全部的实现细节，我们仔细分析之：

- 上下两个红色部分的 $1\times1$ 卷积替换为 $1\times1$ 的分组卷积，分组 g 一般不会很大，论文中的几个值分别是1，2，3，4，8。当 g=1 时，ShuffleNet v1退化为Xception。 g 的值确保能够被通道数整除，保证reshape操作的有效执行。
- 在第一个 $1\times1$ 卷积之后添加一个1.1节介绍的Channel Shuffle操作。

- 如图3.(c)中需要降采样的情况，左侧shortcut部分使用的是步长为2的 $3\times3$ 平均池化，右侧使用的是步长为2的 $3\times3$ 的Depthwise卷积。

- 去掉了 $3\times3$ 卷积之后的ReLU激活，目的是为了减少ReLU激活造成的信息损耗，具体原因见MobileNet v2[7]。

- 如果进行了降采样，为了保证参数数量不骤减，往往需要加倍通道数量。所以==在3.(c)中使用的是拼接（Concat）==操作用于加倍通道数，而3.(b)中则是一个单位加。


最后基于ShuffleNet v1 单元，我们计算一下ResNet，ResNeXt，ShuffleNet v1的FLOPs，即执行一个单元需要的计算量。Channel Shuffle处的操作数非常少，这里可以忽略不计。假设输入Feature Map的尺寸为 $w\times h\times c$ ，bottleneck的通道数为 m 。

ResNet： $F_{\text{ResNet}} = hwcm +3\cdot3\cdot hwmm  + hwcm = hw(2cm + 9m^2)$
ResNeXt： $F_{\text{ResNeXt}} = hwcm +3\cdot3\cdot hw\frac{m}{g}\frac{m}{g}\cdot g  + hwcm = hw(2cm + \frac{9m^2}{g})$ 
ShuffleNet v1：  $F_{\text{ShuffleNet v1}} = hw\frac{c}{g}\frac{m}{g}\cdot g + 3\cdot 3 h w m + hw\frac{c}{g}\frac{m}{g}\cdot g = hw(\frac{2cm}{g} + 9m)$ 

我们可以非常容易得到它们的FLOPs的关系：

$F_{\text{ResNet}} < F_{\text{ResNeXt}} < F_{\text{ShuffleNet v1}} $

