## Weight Standardization

地址：https://arxiv.org/abs/1903.10520

作者：Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, Alan Yuille

机构：Johns Hopkins University

代码地址：https://github.com/joe-siyuan-qiao/WeightStandardization



### 摘要

In this paper, we propose Weight Standardization (WS) to accelerate deep network training. ==WS is targeted at the micro-batch training setting where each GPU typically has only 1-2 images for training==. The micro-batch training setting is hard because small batch sizes are not enough for training networks with Batch Normalization (BN), while other normalization methods that do not rely on batch knowledge still have difficulty matching the performances of BN in large-batch training. Our WS ends this problem because when used with Group Normalization and trained with 1 image/GPU, ==WS is able to match or outperform the performances of BN trained with large batch sizes with only 2 more lines of code.== In micro-batch training, WS significantly outperforms other normalization methods. ==WS achieves these superior results by standardizing the weights in the convolutional layers, which we show is able to smooth the loss landscape by reducing the Lipschitz constants of the loss and the gradients==. The effectiveness of WS is verified on many tasks, including image classification, object detection, instance segmentation, video recognition, semantic segmentation, and point cloud recognition.  



#### Torch 实现

```python
class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
```

### 实验效果

![1593331285576](D:\Notes\raw_images\1593331285576.png)



![1593331302977](D:\Notes\raw_images\1593331302977.png)



**知乎评论, 粗看了论文，个人理解如下：**

（1）基本思想和BN应该是一致的，就是尽量保证映射的平滑性。不过BN是通过对反馈的信号的约束来间接调整w，这里是直接调整w. 从效率上说，的确是直接约束w更加快速，这可能是系统收敛比BN更快的原因。实际上，简单的类比，最优化的网络构造的映射应该是映射空间的测地线，其基本特征就是‘匀速’，这里的WS就是在直接去保证映射满足在数据空间的变换‘匀速’（假定网络结构足够平滑，否则必须考虑卷积核尺寸因素），直接看速度表，而BN是通过要求对速度调整（油门刹车操作）操作的均匀来达到‘匀速’。

（2）直觉上如果网络足够宽足够深且网络结构平滑，则基本按照这个算法得到的网络配置会接近满足dynamical isometry特征，这样的网络应该有更好的收敛特性。

（3）如果网络的深度不太深，而且结构平滑性不好（比如每层卷积核的尺度变化较大），直觉上这个约束不一定会有好的效果。



文章来源：https://zhuanlan.zhihu.com/p/61783291

#### **一、温故而知新**

先简要复习一下BN的概念：

![img](https://pic4.zhimg.com/80/v2-a5ee32d57bc428e9b32bcf620f3cee03_hd.jpg)

**（1）主要解决了：ICS (Internal Covariate Shift)现象**

Google原文这样定义：

> We define *Internal Covariate Shift* as the change in the distribution of network activations due to the change in network parameters during training.

简单理解，就是激活函数势必会改变各层数据的分布，那么随着网络的加深，这种改变的趋势也会加剧，数据分布会越来越偏移。我们知道像sigmoid、tanh这一类激活函数在x绝对值较大时梯度较小，当x逐渐向函数两端区域靠拢时，模型将很难继续优化更新。

因此，BN试图将各层数据的分布固定住（目前能肯定的是，确实固定了数据的范围，但不一定真的调整了分布），使得各层数据在进入激活函数前，保持均值为0，方差为1的正态分布。这样就能使数据尽量靠近激活函数的中心，在反向传播时获得有效的梯度。

当然，由于BN的本质还是根据激活函数调整数据分布范围，所以增加了偏移量 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 来应对Relu等不以0为中心的激活函数，增加网络稳定性。

![img](https://pic3.zhimg.com/80/v2-b8eac7a0f203727331e6ed68f70ce3d6_hd.jpg)

​		sigmoid函数

**（2）计算方式**

BN，batch normalization，即是对batch进行归一化。如图所示，一个batch取N张图片。

对于这N张图片的同一个位置的像素点来说，在每一层总会经过同一个神经元。（如图 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B5%7D) 位置像素点，都会与卷积核的 ![[公式]](https://www.zhihu.com/equation?tex=w_%7B1%7D) 参数进行计算）。所谓的BN就是针对每一个神经元来说的。在训练开始前，会先计算得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5B+w_%7B1%7Dx_%7B5%5C_1%7D%2Cw_%7B1%7Dx_%7B5%5C_2%7D%2Cw_%7B1%7Dx_%7B5%5C_3%7D%2C...%2Cw_%7B1%7Dx_%7B5%5C_N%7D+%5Cright%5D) 的均值和方差。（所以说BN非常依赖batch_size）

依次输入每张图片， ![[公式]](https://www.zhihu.com/equation?tex=+x_%7B5%5C_1%7D%2Cx_%7B5%5C_2%7D%2Cx_%7B5%5C_3%7D%2C...%2Cx_%7B5%5C_N%7D+) 依次经过该神经元，每个 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B5%5C_i%7D) 都会得到一个卷积结果 ![[公式]](https://www.zhihu.com/equation?tex=w_%7B1%7Dx_%7B5%5C_i%7D%2Bb_%7B0%7D) ，按照之前计算的μ和δ对卷积结果进行归一化，再进入激活函数。后面每一层的BN操作同理，都是对前一层的卷积结果进行归一化，这样就保证了每层激活函数的输入都是均值为0，方差为1的分布。

![img](https://pic1.zhimg.com/80/v2-5077eb13b39406f94f71eab540527334_hd.jpg)

Batch Normalization示意图

Layer normalization（LN）可以更好地帮助我们理解BN。BN是把整个batch的数据看做整体，针对每个神经元的输入来归一化。LN则是把每一层的数据看做整体，针对每一层进行归一化。解决了BN不能很好适用于RNN的问题。

![img](https://pic4.zhimg.com/80/v2-3feb1807cbdf15c85c2f73d26987385f_hd.jpg)

Layer Normalization示意图

**（3）存在的问题**

batch_size过小：由于BN是以整个batch来计算均值和方差，所以batch size不能设置过小，失去BN的意义。

batch_size过大：①超过内存容量 ②跑一次epoch迭代次数少，达到相同精度所需要的迭代次数（参数调整的次数）是差不多的，所以大的batch size需要跑更多的epoch，导致总的训练时间变长。③过大的batch size会直接固定下降方向，导致很难更新。

### 二、What is Weight standarization？

于是，为了像BN一样加速训练过程，又能够摆脱对于large batch size的限制，WS（Weight standarization）横空出世。

常见的normalization方式（e.g. BN,LN,IN,GN）都是从激活函数的输入来考虑，以不同的方式对激活函数的输入进行标准化；WS则想，我们这么费心费力地去处理卷积后的结果来加速训练，那为什么不直接去处理这个卷积的weight呢。最后实验表明，确实直接向weight下手对速度的影响更加直观。同时，直接处理weight，也很好地规避了对batch size的依赖，使得真正的mini batch size成为可能。

![img](https://pic2.zhimg.com/80/v2-4da3a07ff6e6dc7152e667dafd2d404d_hd.jpg)

Weight standarization示意图

通过以下的公式再结合上图 可以很好地理解Weight Standarization:

重点在于对每个卷积核进行一个归一化。有 ![[公式]](https://www.zhihu.com/equation?tex=x) 个卷积核，即输出 ![[公式]](https://www.zhihu.com/equation?tex=x) 个channel，就会进行 ![[公式]](https://www.zhihu.com/equation?tex=x) 次归一化，产生 ![[公式]](https://www.zhihu.com/equation?tex=x) 个 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) （注意原论文公式(4)求 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma+W_%7Bi%2C.%7D) 时有一个笔误，把求和符号下面的 ![[公式]](https://www.zhihu.com/equation?tex=j) 写成了 ![[公式]](https://www.zhihu.com/equation?tex=i) ）。

![1579053896259](D:\Notes\raw_images\1579053896259.png)

其中 ![[公式]](https://www.zhihu.com/equation?tex=I%3DC_%7Bin%7D%5Ctimes+Kernel%5C_Size) 。对weight进行标准化后得到卷积结果，此时再加上GN或者BN，就逆天了。。。

另外，作者在分析WS的好处时提到了能够获得更小的L约束。

谈及L约束，我们可以先回顾一下扰动敏感：（1）参数的扰动（2）输入的扰动。就是说，当 模型的参数![[公式]](https://www.zhihu.com/equation?tex=w)或者输入 ![[公式]](https://www.zhihu.com/equation?tex=x) ，变为 ![[公式]](https://www.zhihu.com/equation?tex=w%2B%5CDelta+w) 或者 ![[公式]](https://www.zhihu.com/equation?tex=x%2B%5CDelta+x) 时，模型能否对这种扰动不敏感，给出同样的输出。可以理解为是一种抗"噪声"的鲁棒性。（注意此处之所以称之为“扰动”，就是暗含**变化不大，轻微的一点点变化**。不然输入都从猫变成狗了，模型输出还不变，那这个模型就有点儿问题了23333）

我们希望它不敏感，就是当 ![[公式]](https://www.zhihu.com/equation?tex=%E2%80%96x1%E2%88%92x2%E2%80%96) 很小的时候（变化不大~）， ![[公式]](https://www.zhihu.com/equation?tex=%E2%80%96f_w%28x1%29%E2%88%92f_w%28x2%29%E2%80%96) 也尽可能小。为了定义这个“尽可能”，Lipschitz提出了L约束，存在某个常数C，使得下式恒成立：

![[公式]](https://www.zhihu.com/equation?tex=%E2%80%96f_w%28x1%29%E2%88%92f_w%28x2%29%E2%80%96%E2%89%A4C%28w%29%E2%8B%85%E2%80%96x1%E2%88%92x2%E2%80%96)

于是，我们常常会用L约束来衡量一个模型的好坏，如果能找出这么一个C，我们就说它合格了，满足L约束。如果还能让这个C比较小，我们就说它比较优秀，抗扰动的能力强。

常用的L2正则，其实就是对 ![[公式]](https://www.zhihu.com/equation?tex=%E2%80%96f_w%28x1%29%E2%88%92f_w%28x2%29%E2%80%96) 进行一阶近似后，取的C的最小值![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7B%5Csum_%7Bi%2Cj%7D%5E%7B%7D%7Bw%5E%7B2%7D_%7Bij%7D%7D%7D) .

在这篇文章中，作者也是利用同样的思路，证明了经过WS的loss能够比没用WS的loss继续降低这个C值，从而验证WS对于提升模型性能的作用。

![1579053957835](D:\Notes\raw_images\1579053957835.png)

经过Weight standarization的loss能够比原先的loss继续降低L约束的C值。

#### 三、实验结果

实验非常全面，在ImageNet、COCO、VOC、ModelNet40都对GN+WS和BN+WS做了详细的实验分析，具体可以去原文仔细阅读。

最后的结论是GN+WS可以在micro batch（1~2images per GPU）的条件下，匹配甚至超越BN在large batch size下的性能。

![img](https://pic1.zhimg.com/80/v2-a3a2c3c9d05756904c0bae4da3e33984_hd.jpg)

![img](https://pic4.zhimg.com/80/v2-90c718e80e05398d7e54a42deb1e351f_hd.jpg)