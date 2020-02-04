

### Omni-Scale Feature Learning for Person Re-Identification

论文地址: https://arxiv.org/pdf/1905.00953v2.pdf

代码地址：https://github.com/KaiyangZhou/deep-person-reid

作者：Kaiyang Zhou^1， Yongxin Yang^1, Andrea Cavallaro^2, Tao Xiang^1,3

机构：University of Surrey, Queen Mary University of London， Samsung AI Center, Cambridge



#### **摘要**

作为实例级识别问题，人员重新识别（ReID）==依赖于判别特征==，其不仅捕获不同的空间尺度而且还封装多个尺度的任意组合。我们将这些特征称为同构和异构尺度的==全尺度特征==。在这篇论文中，一部新颖的深刻CNN被设计为全称网络（Omni-Scale Network， OSNet），用于ReID中的全方位特征学习。这是通过设计由==多个卷积特征流==组成的==残差块==来实现的，每个卷积特征流==检测特定尺度的特征==。重要的是，==引入了一种新颖的统一聚合门==，以动态融合多尺度特征和输入相关的信道方向权重。为了有效地学习空间通道相关性并避免过度拟合，构建块使用逐点和深度卷积。通过逐层堆叠这些块，我们的OSNet非常轻量级，可以在现有的ReID基准测试中从头开始进行训练。尽管模型尺寸较小，但我们的**OSNet在六个ReID数据集上实现了最先进的性能**。



#### **介绍**

行人重识别包含两类挑战：

1）类内差异大 - 相机视角的变化带来的行人变化，

2）类间差异小 - 穿着类似衣服的不同的人，

作者认为解决方法：学习判别性特征，而且是全尺度的特征。

特点： 

1）学习全尺度特征表示。不同尺度的特征，通过Chanel-Wise Weights 进行混合，而这些权重是通过一个独立的子网生成，叫做Aggregation Gate, ==AG是跨所有分支的共享参数的子网==。

![1564740192055](C:\Users\j00496872\Desktop\Notes\raw_images\1564740192055.png)

2）设计了一个轻量级的网络。带来的好处：中小型数据集（reid数据集一般都不大）不至于过拟合；大规模监控应用中reid在设备端提取特征。

效果：比流行的基于ResNet50的模型小一个数量级，但却非常强大，并在6个reid数据集上实现先进性能。

To this end, in our building block, we factorise standard convolutions with **pointwise** and **depthwise** convolutions [9, 10].

> [9] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam,
> “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.
> [10] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in CVPR, 2018.



#### **分解卷积**

在本节中，我们介绍了OSNet，它专门研究人ReID任务的全方位特征表示。 我们从分解卷积层开始，然后引入全尺寸残差块，和统一聚合门。

![1564740031788](C:\Users\j00496872\Desktop\Notes\raw_images\1564740031788.png)

分解卷积：为了减少参数的数量，我们采用深度可分离的卷积，将标准卷积分成两个单独的层：逐点卷积和深度卷积。

为了减少参数量，采用 depthwise separable convolutions，即把标准卷积分成 pointwise convolutions 和depthwise convolutions.

标准卷积由4D张量$w \in R^{k×k×c×c^{'}} $ 参数化，其中k是内核大小，c是输入通道的深度，$c^{'}$是输出通道的深度。 为了学习输入张量 $x\in R^{h×w×c}$ 的空间通道相关性，其中h是高度，w是宽度，卷积运算可以表示为$x^{'} =φ（w * x）$，其中 φ 是非线性的映射（ReLU）*表示卷积。 为清楚起见，省略了偏差。图3（a）描绘了标准的实际实现3×3卷积层

 ![1564975889386](C:\Users\j00496872\Desktop\Notes\raw_images\1564975889386.png)

设 $u \in R^{1×1×c×c^{'}} $ 是一个逐点卷积核，密集地连接到信道维，

而 $v \in R^{k×k×1×c^{'}} $ 是深度卷积核，它将每个feature map的局部信息与感知域 k 聚合在一起。

==我们通过将 $w$ 分解为 $v$, $ u$ 来解开空间信道相关性的学习==，导致 $x^{'} = φ（（vou）* x）$，如图3（b）所示。

 结果，计算成本从 $h·w·k^2·c·c^{'}$ 减小到 $h·w·（k^2 + c）·c^{'}$，以及参数的数量 $k^2·c·c^{'}$ 至$（k^2 + c）·c^{'}$。 

当我们分解3×3卷积时，我们将这些层称为Lite 3×3。请注意，我们的实现不同于原始的深度可分离卷积[39]，它在逐点卷积之前应用深度卷积。 根据经验，我们发现与原始版本（深度方向->逐点）相比，我们的设计（逐点->深度）对于全方位特征学习更有效。

#### **OS残差块**

为了实现全方位表示学习，我们通过引入表示特征尺度的新维度指数 $t$ 来扩展残差函数 $F$. 对于$F^t$，当 $t> 1$时，我们堆叠 $t$ Lite $3×3$ 层，这导致大小$（2t + 1）×（2t + 1）$ 的感受野。 然后，要学习的残差 $\bar{x}$ 是直到 $T$ 的表示的增量比例的总和。本论文将T设为4，也就是说感受野最大为9.

![1565920293410](C:\Users\j00496872\Desktop\Notes\raw_images\1565920293410.png)

where $F$ represents a Lite $3 × 3$ layer that learns single scale features (scale = 3).

![1565920880852](C:\Users\j00496872\Desktop\Notes\raw_images\1565920880852.png)

#### 统一聚合门

每个流可以为我们提供特定比例的特征，即它们是Scale均匀的。 为了学习全尺寸特征，我们==提出以动态方式组合不同流的输出，即根据输入图像将不同的权重分配给不同的尺度==，==而不是在训练之后固定==。 更具体地说，动态规模融合是通过新颖的方式实现的聚合门（AG），这是一个可学习的神经网络。

加入此聚合门之后的残差项公式为：

![1565921318382](C:\Users\j00496872\Desktop\Notes\raw_images\1565921318382.png)

G is implemented as a mini-network composed of ==a non-parametric global average pooling layer== [41] and
a ==multi-layer perceptron (MLP)== with one ==ReLU-activated== hidden layer, followed by the ==sigmoid activation==. 

To reduce parameter overhead, we follow [42, 43] to reduce the hidden dimension of the MLP with a reduction ratio, which is set to 16.

值得指出的是，与使用提供粗尺度融合的单个标量输出函数相比，我们==选择使用channel-wise weights，即AG子网 $G（x^t）$的输出是第t个流的向量而不是标量==。 ==这种设计可以产生更精细的融合，可以调整每个特征通道==。 另外，通过调节来动态地计算权重输入数据。 这对于ReID作为测试图像至关重要包含与培训中不同身份的人; 因此，更期望自适应/输入相关的特征尺度融合策略。

Note that in our architecture, the ==AG is shared for all feature streams== in the same omni-scale residual block (dashed box in Fig. 4(b)).

> 代码中聚合门实现：先对输入AG的特征X(n, 64, 128, 64)做全局平均池化操作，得到(n, 64)的特征，然
> 后将此特征通过全连接映射成shape为(n, 4)的特征，最后再将(n, 4)的特征映射回来成shape为(n, 64)的
> 特征Y，最后将AG输出的Y(n, 64)和X(n, 64, 128, 64)通过broadcast机制相乘。本质上就是包含一个隐藏
> 层的全连接网络，其中隐藏层包含四个神经元。

```Python
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels  # 64
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, 									 bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, 
                             bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()
      
    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x
```



#### 网络结构

![1564977001159](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1564977001159.png)

#### Ablation Study

![1565924796114](C:\Users\j00496872\Desktop\Notes\raw_images\1565924796114.png)

#### 结论

我们介绍了OSNet，一种轻量级CNN架构，能够学习全方位的特征表示。 对六个ReID数据集进行的大量实验表明，尽管OSNet具有轻量级设计，但仍能实现最先进的性能。 我们还在单标签对象分类任务和多标签属性识别任务上评估了OSNet。 OSNet在这些任务上的卓越性能表明，OSNet对ReID之外的视觉识别问题具有广泛的兴趣。

**实验结果：**

![1564736939687](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1564736939687.png)

Table 3. Results (%) on big ReID datasets. It is clear that OSNet achieves the best performance on all datasets, surpassing the published state-of-the-art ReID methods by a clear margin. It is noteworthy that OSNet has only 2.2 million parameters, which are far less than current best-performing ResNet-based methods. -: not available. y: model trained from scratch. z: reproduced by us.

