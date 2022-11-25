## MobileNet V1

论文题目：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications 

论文地址：[https://arxiv.org/abs/1704.04861](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1704.04861) 

作者：Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

收录：CVPR2017

机构： Google

文章来源：https://zhuanlan.zhihu.com/p/70703846



其实介绍MobileNetV1（以下简称V1）只有一句话，==MobileNetV1就是把VGG中的标准卷积层换成深度可分离卷积就可以了。==那么，这个深度可分离卷积是什么？

### 深度可分离卷积

可分离卷积主要有两种类型：**空间可分离卷积**和**深度可分离卷积**。

#### 空间可分离

顾名思义，空间可分离就是将一个大的卷积核变成两个小的卷积核，比如将一个3×3的核分成一个3×1和一个1×3的核。由于空间可分离卷积不在MobileNet的范围内，就不说了。

#### 深度可分离卷积



![img](https://pic1.zhimg.com/80/v2-30b6ec010424b349e16529607e1d0c2c_hd.jpg)

==深度可分离卷积就是将普通卷积拆分成为一个深度卷积和一个逐点卷积==。

我们先来看一下标准的卷积操作：

![img](https://pic4.zhimg.com/80/v2-f471bdb9191d0c8b65688ececbe935fb_hd.jpg)

输入一个12×12×3的一个输入特征图，经过5×5×3的卷积核卷积得到一个8×8×1的输出特征图。如果此时我们有256个特征图，我们将会得到一个8×8×256的输出特征图。

以上就是标准卷积做干的活。那深度卷积和逐点卷积呢？

#### 深度卷积

![img](https://pic2.zhimg.com/80/v2-b74a5e8241eb500949d8dcc47558d035_hd.jpg)

与标准卷积网络不一样的是，我们将**卷积核拆分成为但单通道形式**，在**不改变输入特征图像的深度**的情况下，对**每一通道进行卷积操作**，这样就得到了**和输入特征图通道数一致的输出特征图**。如上图：输入12×12×3的特征图，经过5×5×1×3的深度卷积之后，得到了8×8×3的输出特征图。输入个输出的维度是不变的3。这样就会有一个问题，通道数太少，特征图的维度太少，能获取到足够的有效信息吗？

#### 逐点卷积

**逐点卷积就是1×1卷积**。主要作用就是对特征图进行升维和降维，如下图：

![img](https://pic1.zhimg.com/80/v2-f480c4453e9b7915c88d34ca79288e20_hd.jpg)



在深度卷积的过程中，我们得到了8×8×3的输出特征图，我们用256个1×1×3的卷积核对输入特征图进行卷积操作，输出的特征图和标准的卷积操作一样都是8×8×256了。

**标准卷积与深度可分离卷积的过程对比**如下：



![img](https://pic3.zhimg.com/80/v2-e123df730cbb163fff15987638bfb03e_hd.jpg)

#### 为什么要深度可分离卷积？

这个问题很好回答，如果有一个方法能让你用更少的参数，更少的运算，但是能达到差的不是很多的结果，你会使用吗？

深度可分离卷积就是这样的一个方法。我们首先来计算一下**标准卷积的参数量与计算量**（只考虑MAdd）：

![img](https://pic2.zhimg.com/80/v2-eb2f07d6b8bc4c8a90c5daafbba683dd_hd.jpg)

标准卷积算完了，我们接下来计算**深度可分离卷积的参数量和计算量**：

![img](https://pic1.zhimg.com/80/v2-2d11a371ccccc4716958e752ce6d423c_hd.jpg)

**总的来说：**

![img](https://pic4.zhimg.com/80/v2-abb36872e97253589e659e6484e63423_hd.jpg)

可以参数数量和乘加操作的运算量均下降为原来的$\frac1N+\frac1{D_K^2}$。

我们通常所使用的是3×3的卷积核，也就是会下降到原来的**九分之一到八分之一**。

**举例说明：假设**

输出为一个224×224×3的图像，VGG网络某层卷积输入的尺寸是112×112×64的特征图，卷积核为3×3×128，**标准卷积的运算量**是：

3×3×128×64×112×112 = 924844032

**深度可分离卷积的运算量**是：

3×3×64×112×112+128×64×112×112 = 109985792

这一层，MobileNetV1所采用的**深度可分离卷积计算量与标准卷积计算量的比值**为：

109985792 /924844032 = 0.1189

与我们所计算的九分之一到八分之一一致。

---

### V1卷积层

![img](https://pic4.zhimg.com/80/v2-9e51f4ea53e3ca29e134d913db8ead87_hd.jpg)

上图左边是标准卷积层，右边是V1的卷积层，虚线处是不相同点。V1的卷积层，首先使用3×3的深度卷积提取特征，接着是一个BN层，随后是一个ReLU层，在之后就会逐点卷积，最后就是BN和ReLU了。这也很符合深度可分离卷积，**将左边的标准卷积拆分成右边的一个深度卷积和一个逐点卷积**。

#### ReLU6

![img](https://pic3.zhimg.com/80/v2-9f1a722b2eceaa84169300521843bdfa_hd.jpg)

上图左边是普通的ReLU，对于大于0的值不进行处理，==右边是ReLU6，当输入的值大于6的时候，返回6==，relu6“具有一个边界”。作者认为**ReLU6作为非线性激活函数，在低精度计算下具有更强的鲁棒性**。（这里所说的“低精度”，我看到有人说不是指的float16，而是指的定点运算(fixed-point arithmetic)）

现在就有一个问题，**标准卷积核深度可分离卷积层到底对结果有什么样的影响呢？**上实验。

![img](https://pic4.zhimg.com/80/v2-60a70f311b302cc51d95b8d43d8c19bf_hd.jpg)

可以看到使用深度可分离卷积与标准卷积，**参数和计算量能下降为后者的九分之一到八分之一**左右。但是**准确率只有下降极小的1％**。

#### V1网络结构

![img](https://pic4.zhimg.com/80/v2-d39fb64f504e0b4364d6a67a15eb39d7_hd.jpg)

MobileNet的网络结构如上图所示。首先是一个3x3的标准卷积，s2进行下采样。然后就是堆积深度可分离卷积，并且其中的部分深度卷积会利用s2进行下采样。然后采用平均池化层将feature变成1x1，根据预测类别大小加上全连接层，最后是一个softmax层。整个网络有28层，其中深度卷积层有13层。

#### 实验结果

V1论文中还有一部分对V1网络再进行调整，在此就不赘述了，感兴趣的同学可以去看看原论文。

V1的效果到底好不好，作者将V1与大型网络GoogleNet和VGG16进行了比较：

![img](https://pic3.zhimg.com/80/v2-eabe379daab0594762692acd648764aa_hd.jpg)

可以发现，作为轻量级网络的V1在计算量小于GoogleNet，参数量差不多是在一个数量级的基础上，在分类效果上比GoogleNet还要好，这就是要得益于深度可分离卷积了。VGG16的计算量参数量比V1大了30倍，但是结果也仅仅只高了1%不到。

目标检测，在COCO数据集上的结果：

![img](https://pic2.zhimg.com/80/v2-bc69c2120e73a131384533d00b81d715_hd.jpg)

对了，作者还在论文中分析整个了网络的参数和计算量分布，如下图所示。可以看到整个计算量基本集中在1x1卷积上。对于参数也主要集中在1x1卷积，除此之外还有就是全连接层占了一部分参数。

![img](https://pic3.zhimg.com/80/v2-507740df0945f193c3e9ee6739451f06_hd.jpg)

