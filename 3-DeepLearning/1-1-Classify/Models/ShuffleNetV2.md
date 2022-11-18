## ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

论文地址：[https://arxiv.org/abs/1807.11164](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1807.11164) 

会议：ECCV2018

作者:  Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

机构：清华大学，旷视

文章来源：

https://zhuanlan.zhihu.com/p/51566209



### 摘要

> Currently, the neural network architecture design is mostly guided by the indirect metric of computation complexity, i.e., FLOPs. However, the direct metric, e.g., speed, also depends on the other factors such as memory access cost and platform characterics. Thus, this work proposes to evaluate the direct metric on the target platform, beyond only considering FLOPs. Based on a series of controlled experiments, this work derives several practical guidelines for efficient network design. Accordingly, a new architecture is presented, called ShuffleNet V2. Comprehensive ablation experiments verify that our model is the state-of-the-art in terms of speed and accuracy tradeoff.



在ShuffleNet v2的文章中作者指出现在普遍采用的FLOPs评估模型性能是非常不合理的，因为一批样本的训练时间除了看FLOPs，还有很多过程需要消耗时间，例如文件IO，内存读取，GPU执行效率等等。作者从内存消耗成本，GPU并行性两个方向分析了模型可能带来的非FLOPs的行动损耗，进而设计了更加高效的ShuffleNet v2。ShuffleNet v2的架构和[DenseNet](https://zhuanlan.zhihu.com/p/42708327)[4]有异曲同工之妙，而且其速度和精度都要优于DenseNet。



### 实验结果

![1598863768938](D:\Notes\raw_images\1598863768938.png)



### 介绍

![1598863394131](D:\Notes\raw_images\1598863394131.png)

Fig. 3: Building blocks of ShuffleNet  v1 [15] and this work. (a): the basic ShuffleNet unit; (b) the ShuffleNet unit for spatial down sampling (2x);  (c) our basic unit; (d) our unit for spatial down sampling (2x).   

DWConv: depthwise convolution. GConv: group convolution.

仔细观察(c)，(d)对网络的改进我们发现了以下几点：

1. 在(c)中ShuffleNet v2使用了一个通道分割（Channel Split）操作。这个操作非常简单，即将 c 个输入Feature分成 c-c' 和 c' 两组，一般情况下 $c' = \frac{c}{2}$ 。这种设计是为了尽量控制分支数，为了满足G3。
2. 在分割之后的两个分支，左侧是一个直接映射，右侧是一个输入通道数和输出通道数均相同的深度可分离卷积，为了满足G1。

3. 在右侧的卷积中， $1\times1$ 卷积并没有使用分组卷积，为了满足G2。

4. 最后在合并的时候均是使用拼接操作，为了满足G4。

5. 在堆叠ShuffleNet v2的时候，通道拼接，通道洗牌和通道分割可以合并成1个element-wise操作，也是为了满足G4。

最后当需要降采样的时候我们通过不进行通道分割的方式达到通道数量的加倍，如图6.(d)，非常简单。



### 2 高效模型的设计准则

**G1）：当输入通道数和输出通道数相同时，MAC最小**

假设一个卷积操作的输入Feature Map的尺寸是 $w\times h\times c_1$ ，输出Feature Map的尺寸为 $w\times h\times c_2$ 。卷积操作的FLOPs为 $F = hwc_1 c_2$ 。在计算这个卷积的过程中，输入Feature Map占用的内存大小是 $hwc_1$ ，输出Feature Map占用的内存是 $hwc_2$ ，卷积核占用的内存是 $c_1 c_2$ 。总计：
$$
MAC = hw(c_1 + c_2) + c_1 c_2 \\  = \sqrt{(hw(c_1 + c_2))^2} + \frac{B}{hw} \\  = \sqrt{(hw)^2 \cdot (c_1 + c_2)^2} + \frac{B}{hw}\\  \geq \sqrt{(hw)^2\cdot 4 c_1 c_2} + \frac{B}{hw} \\  = s\sqrt{hw \cdot (hwc_1 c_2)} + \frac{B}{hw} \\  = 2\sqrt{hwB} + \frac{B}{hw} \tag4
$$
当 $c_1 = c_2$ 时上式取等号。也就是说当FLOPs确定的时候，时 $c_1 = c_2$ 模型的运行效率最高，因为此时的MAC最小。

**G2）：MAC与分组数量** g **成正比**

在分组卷积中，FLOPs为 
$$
F = hw\frac{c_1}{g} \frac{c_2}{g} g = \frac{hwc_1 c_2}{g}
$$
  ，其MAC的计算方式为：
$$
MAC = hw(c_1 + c_2) + \frac{c_1}{g} \frac{c_2}{g} g \\  = hw(c_1 + c_2) + \frac{c_1 c_2 }{g} \\  = B g (\frac{1}{c_1} + \frac{1}{c_2}) + \frac{B}{hw} \tag5
$$
根据G2，我们在设计网络时 g 的值不应过大。

**G3）：网络的分支数量降低并行能力**

分支数量比较多的典型网络是Inception，NasNet等。作者证明这个一组准则是设计了一组对照试验：如图4所示，通过控制卷积的通道数来使5组对照试验的FLOPs相同，通过实验我们发现它们按效率从高到低排列依次是 (a) > (b) > (d) > (c) > (e)。

![img](https://pic1.zhimg.com/80/v2-ab32d82950bcc0a9f0519a3b765f1270_720w.jpg)

图4：网络分支对比试验样本示意图

造成这种现象的原因是更多的分支需要更多的卷积核加载和同步操作。

**G4）：Element-wise操作是非常耗时的**

我们在计算FLOPs时往往只考虑卷积中的乘法操作，但是一些Element-wise操作（例如ReLU激活，偏置，单位加等）往往被忽略掉。作者指出这些Element-wise操作看似数量很少，但它对模型的速度影响非常大。尤其是深度可分离卷积这种MAC/FLOPs比值较高的算法。图5中统计了ShuffleNet v1和MobileNet v2中各个操作在GPU和ARM上的消耗时间占比。

![img](https://pic3.zhimg.com/80/v2-7894170975e17aaf071e1081edb97f4c_720w.jpg)

图5：模型训练时间拆分示意图

总结一下，在设计高性能网络时，我们要尽可能做到：

- G1). 使用输入通道和输出通道相同的卷积操作；
- G2). 谨慎使用分组卷积；
- G3). 减少网络分支数；
- G4). 减少element-wise操作。

例如在ShuffleNet v1中使用的分组卷积是违背G2的，而每个ShuffleNet v1单元使用了bottleneck结构是违背G1的。MobileNet v2中的大量分支是违背G3的，在Depthwise处使用ReLU6激活是违背G4的。

从它的对比实验中我们可以看到虽然ShuffleNet v2要比和它FLOPs数量近似的的模型的速度要快。

