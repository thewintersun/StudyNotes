## An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

论文地址：https://arxiv.org/abs/2010.11929

作者：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

发表：ICLR2021

机构：Google Research，Google Brain

项目：Fine-tuning code and pre-trained models are available at [this https URL](https://github.com/google-research/vision_transformer). 



### 摘要

虽然 Transformer 架构已成为 NLP 任务的事实标准，但它在 CV 中的应用仍然有限。在视觉上，注意力要么与卷积网络结合使用，要么用于替换卷积网络的某些组件，同时保持其整体结构。我们证明了这种对 CNNs 的依赖是不必要的，直接应用于图像块序列的纯 Transformer 可以很好地执行图像分类任务。当对大量数据进行预训练并迁移到多个中小型图像识别基准时，与 SOTA 的 CNN 相比，Vision Transformer可获得更优异的结果，同时仅需更少的训练资源。



### 引文

近年来，Transformer已经成了NLP领域的标准配置，但是CV领域还是卷积架构占主导地位。

最近CV界也有很多文章将transformer迁移到CV领域，这些文章总的来说可以分为两个大类：

1. 将自注意力机制与常见的CNN架构结合；
2. 用自注意力机制完全替代CNN。

本文采用的是第2种思路。虽然已经有很多工作用自注意力机制完全替代CNN，且在理论上效率比较高，但是它们用了特殊的注意力机制，无法从硬件层面加速。

**文章不同于以往工作的地方，就是尽可能地将NLP领域的transformer不作修改地搬到CV领域来。**

受NLP中Transformer扩展成功的启发，作者尝试将标准Transformer直接应用于图像，并进行最少的修改。模型通过将图像拆分为小块，并提供这些小块的线性嵌入序列作为Transformer的输入。图像图块与NLP应用程序中的 token（words）的处理方式相同，以监督方式对模型进行图像分类训练。

可能由于==Transformer缺乏CNN固有的一些inductive biases ，例如平移不变性和局部性，因此在训练不足的数据量时不能很好地概括==。所以模型在对中等规模的数据集（例如ImageNet）进行训练时，所产生的精度要比同等规模的ResNet低几个百分点。

但是，如果在更大的数据集上训练模型（14M-300M图像），则这种情况会发生变化，large scale training要胜过inductive bias。ViT经过足够的预培训并转移到数据点较少的任务时，可以获得出色的结果。当在公共ImageNet-21k数据集或内部JFT-300M数据集上进行预训练时，ViT在多个图像识别基准上达到或超越了最新水平。特别是，最佳模型在ImageNet上达到88.55％的精度，在ImageNet-ReaL上达到90.72％的精度，在CIFAR-100上达到94.55％的精度，在19个任务的VTAB上达到77.63％的精度。



### 核心思想

由于NLP处理的语言数据是序列化的，而CV中处理的图像数据是三维的，所以我们需要一个方式将图像这种三维数据转化为序列化的数据。

如何将2D图片数据转换成 1D数据？目前 BERT 能够处理的序列长度是512，如果直接将图像像素转换成 1D。即使是 224 × 224 大小的图片，其序列长度也有5万多，计算复杂度将是 BERT 的100倍，如果是检测或分割任务，那计算复杂度就更大了。

所以将自注意力机制应用在CV领域，关键在于将图片分割等大的patch，添加位置信息，然后按序排成一排，输入进Vision Transformer进行训练：

<img src="D:\Notes\raw_images\image-20221010170615019.png" alt="image-20221010170615019" style="zoom:80%;" />

图 1：模型概述。我们将图像分割成固定大小的块，线性嵌入每个块，添加位置嵌入，并将生成的向量序列送到标准的 Transformer 编码器。为了执行分类，我们使用序列添加额外可学习的“分类标记”的标准方法。 

### 一、预处理

#### 1.1 图像块嵌入

标准 Transformer 接受一维标记嵌入序列作为输入。为处理 2D 图像，我们将图像 reshape 为一个展平的2D块序列。将一张图片 $(H×W×C)$ 分割为$N$个像素点个数为 $(P×P×C)$ 的图像块，图像块长宽为$P$，那么 $N=\frac {H×W}{P×P}$ ，再把每个图像块展平后连接得到一个 $N×(P^2⋅C)$ 的二维矩阵，相当于NLP中输入transformer的词向量。

但有一个问题，因为P是不固定的，那么得到的的每个patch的向量长度也是不一样的，为了模型不受patch大小的影响，作者引入了线性映射来把每个 $P^2⋅C$ 维的patch大小线性映射成固定的$D$ 维。

综上，原本$H×W×C$维的图片被转化为了一个$N×D$维的二维矩阵。

#### 1.2 可学习的嵌入

类似BERT的[class] token，作者在嵌入的补丁序列序列中（$z_0^0=x_{class}$）准备了可学习的embedding向量，该序列在Transformer编码器的输出（$z_L^0$）的状态用作图像表示$y$。 在预训练和fine-tuning期间，都将分类head连接到$z_L^0$。分类head是通过在预训练时具有一个隐藏层的MLP以及在微调时通过一个线性层的MLP来实现的。

位置embedding会添加到patch embedding中，以保留位置信息。通过使用标准的可学习1D位置embedding，因为没有观察到使用更高级的2D感知位置embedding可显着提高性能。embedding向量的结果序列用作编码器的输入。

Tranformer编码器由 multi-head self-attention（MSA）和MLP块的层组成。在每个块之前应用 Layernorm（LN），在每个块之后应用残差连接。MLP包含具有GELU非线性的两全连接层。

<img src="D:\Notes\raw_images\image-20221010171543250.png" alt="image-20221010171543250" style="zoom:67%;" />

#### 1.3 位置嵌入

由于transformer模型本身是没有位置信息的，和NLP中一样，我们需要用位置嵌入将位置信息加到模型中去。

如上图所示1，编号有0-9的紫色框表示各个位置的位置嵌入，而紫色框旁边的粉色框则是经过线性映射之后的展平的向量。文中采用将位置嵌入和图像块嵌入相加的方式结合位置信息。

ViT 论文中对比了几种不同的位置编码方案：

1. 无位置嵌入
2. 1-D 位置嵌入：考虑把 2-D 图像块视为 1-D 序列
3. 2-D 位置嵌入：考虑图像块的 2-D 位置 (x, y)
4. 相对位置嵌入：考虑图像块的相对位置

最后发现如果不提供位置编码效果会差，但其它各种类型的编码效果效果都接近，这主要是因为 ViT 的输入是相对较大的图像块而非像素，所以学习位置信息相对容易很多。

论文中也对学习到的位置编码进行了可视化，发现相近的图像块的位置编码较相似，且同行或列的位置编码也相近：

<img src="D:\Notes\raw_images\image-20221010180023500.png" alt="image-20221010180023500" style="zoom:67%;" />

#### **Inductive bias**

Vision Transformer的image-specific inductive bias[[1\]](https://zhuanlan.zhihu.com/p/380417641#ref_1)比CNN小得多。在CNN中，局部性、二维邻域结构、平移等因素贯穿于整个模型的每一层。在ViT中，只有MLP层是局部的、平移等价的，而自注意层是全局的。二维邻域结构的使用非常少：在模型的开始通过将图像切割成小块，在微调阶段调整不同分辨率的图像的位置嵌入(如下所述)。除此之外，初始化时的位置嵌入不携带任何关于补丁二维位置的信息，所有补丁之间的空间关系都必须从头开始学习。

#### **Hybrid Architecture**

作为原始图像图块的替代，可以从CNN的特征图形成输入序列。在该混合模型中，将patch embedding投影E应用于从CNN特征图提取的图块。作为一种特殊情况，patch的空间大小可以为1x1，这意味着通过简单地展平特征图的空间尺寸并投影到transformer尺寸即可获得输入序列。如上所述，添加了分类输入embedding和位置embedding。

#### **FINE-TUNING AND HIGHER RESOLUTION**

通常，我们在大型数据集上对ViT进行预训练，并微调(fine-tuning)到数据集较小的下游任务中。为此，作者删除了预训练的预测head，并附加了一个初始化为$0$的维度为$D×K$的feed-forward layer，其中$K$是下游类的数量。使用比预训练更高的分辨率进行微调通常是有益的，当提供更高分辨率的图像时，将图块大小保持不变，这会导致更大的有效序列长度。ViT可以处理任意序列长度（直到内存限制），但是，预训练的位置embedding可能不再有意义。因此，根据预先训练的位置嵌入在原始图像中的位置执行2D插值，并只有在分辨率调整和色块提取中，将有关图像2D结构的感应偏差手动注入到Vision Transformer中。



### **实验**

评估了ResNet、ViT和混合模式的表示学习能力。为了了解每个模型的数据要求，对不同大小的数据集进行了预训练，并评估了许多基准任务。当考虑预训练模型的计算成本时，ViT的性能非常好，以较低的预训练成本在大多数识别基准上达到了最先进的水平。最后使用自监督进行了一个小实验，并证明了自监督的ViT对未来充满希望。

**数据集：**为探索模型的可扩展性，作者使用具有1k类和1.3M图像的ILSVRC-2012 ImageNet数据集，具有21k类和14M图像的superset ImageNet-21k和具有18k类和303M高分辨率图像的JFT。遵循Kolesnikov等人的下游任务的测试集，在这些数据集上训练的模型转移到几个基准任务上：原始验证标签上的ImageNet和清理后的ReaL标签（CIFAR-10 / 100）、Oxford-IIIT Pets。

**模型变种：**我们将ViT配置基于BERT所使用的配置，如下表所示。“Base”和“ Large”模型直接由继承而来，还加入了一个较大的“Huge”模型。值得注意的是：transformer的序列长度与图块大小的平方成反比，因此图块大小较小的模型在计算上会更昂贵。

<img src="D:\Notes\raw_images\image-20221010175611254.png" alt="image-20221010175611254" style="zoom:67%;" />

<img src="D:\Notes\raw_images\image-20221010175643745.png" alt="image-20221010175643745" style="zoom:80%;" />



- 而且随着网络深度的增加，MSA的注意力距离达到平衡

<img src="D:\Notes\raw_images\image-20221010180042564.png" alt="image-20221010180042564" style="zoom: 80%;" />

- 另外transformer学习到的感兴趣区域和图像中的物体轮廓基本吻合，进一步验证了在CV领域使用transformer的合理性。

<img src="D:\Notes\raw_images\image-20221010180105985.png" alt="image-20221010180105985" style="zoom:80%;" />