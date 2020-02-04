#### Object Detection based on Region Decomposition and Assembly
- 论文地址：https://arxiv.org/abs/1901.08225 
- 作者： Seung-Hwan Bae
- 机构： Incheon National University, 119 Academy-ro, Yeonsu-gu, Incheon, 22012, Korea
- 文章介绍：https://mp.weixin.qq.com/s/-G47vOGx2iNQCarYRAiNPg 
- 发表： AAAI 2019 

本文解读的是一篇发表于 AAAI 2019 的 paper，文章提出了一种 R-DAD 的方法来对 RCNN 系列的目标检测方法进行改进。

#### 研究动机

目前主流的目标检测算法分为 1 stage 和 2 stage 的，而 2 stage 的目标检测方法以 Faster-RCNN 为代表是需要 RPN（Region Proposals Network）生成 RoI（Region of Interests，感兴趣区域）的，文章认为正是因为被遮挡了的或者不精确的 Region Proposals 导致目标检测算法的不准确。

作者的想法动机其实很简单，就是假如一辆车的左边被人遮挡了，那么这辆车的右边带来的信息其实才是更可信的。基于这个想法，文章提出 R-DAD（Region Decomposition and Assembly Detector），即区域分解组装检测器，来改善生成的 Region Proposals。

#### R-DAD的网络结构

文章以 Faster-RCNN 的网络结构为例，修改成它提出的 R-DAD 结构：

![](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1565863306622.png)

Figure 1: Proposed R-DAD architecture: In the MRP network, rescaled proposals are generated. For each rescaled proposal, we decompose it into several part regions. We design a region assembly block (RAB) with 3x3 convolution filters, ReLU, and max units. In the RDA network, by using RABs we combine the strong responses of decomposed object parts stage by stage, and then learn the semantic relationship between the whole object and part-based features.

R-DAD 网络架构主要分成两个模块 MRP 和 RDA：

**MRP（Multi-Scale Region Proposal）模块**，用来改善 RPN 生成的 Region Proposals 的准确率。 

![1565863672634](C:\Users\j00496872\Desktop\Notes\raw_images\1565863672634.png)

图MRP模块，框内分别对应S=0.7,1,1.2的Region Proposals

MRP 表面意思就是生成多尺度的 Region Proposal，方法很简单，就是使用传统的 RPN 生成一些建议框，然后用不同的缩放因子（文章使用了 5 种缩放因子作为一组 s=[0.5,0.7,1,1.2,1.5]）对生成出的建议框进行不同比例的缩小放大，从而提高 Region Proposals 的多样性。

如图一，生成了不同尺度的区域，有一些仅仅是局部有一些是大于目标本身的，但是这也带来了一个问题，就是原来的 Region Proposals 已经可以说是极大的数量了，再乘以五倍，想要网络能够完全利用这些建议框是不切实际的，作者最后还添加了 RoI 的采样层，对分数低的和跟 ground truth 重叠率低的进行了筛选。 

由 MRP 网络生成的各种 Region Proposals 可以进一步适应目标之间因为空间变化所导致的特征变化，提高结构的鲁棒性。 

**RDA（Region Decomposition and Assembly）模块**，作者也称它为 mutil-region-based appearance model，即基于多区域的外观模型，它可以同时描述一个物体的全局外观和局部外观，RDA 分为目标分解和目标区域集成的两部分，目标分解如图二所示，把一个目标分为上下左右四个方向的分解部分。

![1565863413441](C:\Users\j00496872\Desktop\Notes\raw_images\1565863413441.png)

一般会先用线性插值两倍上采样之后再分解，后面作者给出了表格表示这样效果更好。左右刚好是特征图的左右一半，上下也同理，都会送入 RAB 模块，RAB 模块如图三所示：

![1565863691740](C:\Users\j00496872\Desktop\Notes\raw_images\1565863691740.png)

▲ 图3. RAB模块

其实就是下面这个函数：

![1565863722069](C:\Users\j00496872\Desktop\Notes\raw_images\1565863722069.png)

![1565863732581](C:\Users\j00496872\Desktop\Notes\raw_images\1565863732581.png)

其中 p 代表着上下左右的每一个部分或者组合后的部分如左-右 (l/r)、下-上 (b/u) 和 comb（l/r 与 b/u 的组合），*是卷积操作，f() 是 ReLU 单元。最后再取 max，是为了融合和的信息，生成同样大小的。

最后就是代表着全局信息的 scale 为 1 生成的 Region Proposals，一起送进 RAB 模块。这样整个网络结构就可以做到既捕捉到局部信息的同时，也不丢失全局信息。 

RAB 模块是一个类似 maxout 的单元，理论上它可以逼近任何连续的函数，所以我们使用 RAB 而不是直接使用 ReLU。这表明可以通过配置不同的分层地组合 RAB 模块来表示各种各样的目标特征。

#### 损失函数
对每一个框 (box) d，我们都会通过 IoU 筛选出跟 GT (ground truth) 最匹配的 d*，如果 d 跟任何的 d* 的 IoU 超过 0.5，给予正标签，若在 0.1 到 0.5 之间的，给予负标签。R-DAD 的输出层对每一个框 d 都有四个参数化坐标和一个分类标签。对于 box regression 来说，我们与以往目标检测的参数化一致如下：

![1565863749533](C:\Users\j00496872\Desktop\Notes\raw_images\1565863749533.png)

同理，是用来评估预测框和 GT 的差距的。 

跟训练 RPN 网络相似，R-DAD 也需要最小化分类损失和回归损失，如下：

![1565863762006](C:\Users\j00496872\Desktop\Notes\raw_images\1565863762006.png)

#### 实验结果
文章中做了各种设置的组合，关于 MRP 里缩放因子的组合、是否有 RDA 模块以及是否上采样，得分如下表所示：

![](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1565863800053.png)

与 Faster-RCNN 对比，作者使用了 VOC07trainval 和 VOC12trainval 数据集训练，再在 VOC07test 上测试，并且用了不同的特征提取器（VGG、ZF、Res101），得分均比 Faster-RCNN 高。

![1565863822096](C:\Users\j00496872\Desktop\Notes\raw_images\1565863822096.png)

在速度方面均比 Faster-RCNN 慢。

![1565863844396](C:\Users\j00496872\Desktop\Notes\raw_images\1565863844396.png)


与没有上下区域分解集成的 R-DAD 对比，有上下分解集成的误判率低很多，因为它在复杂情形下被遮挡物体会更有选择地相信得到的信息。

![1565863868381](C:\Users\j00496872\Desktop\Notes\raw_images\1565863868381.png)

![1565863895061](C:\Users\j00496872\Desktop\Notes\raw_images\1565863895061.png)

#### R-DAD的优点

1. 文章提出因为我们最大化目标在横向空间位置上局部特征的语义响应，与使用支持小区域的最大池化相比，在没有深层次结构的情况下，我们可以改善特征位置的空间不变性。我的理解就是作者取了上下左右四个方向的特征模板，最后对四个方向进行了融合语义信息，利用了横向空间上的空间不变性，揭示了不同方向上的语义关系。 

2. 在复杂场景下，如有目标对象被另一目标对象遮挡时，通过左右上下模板筛选出来的特征是更符合真实场景的，这样的 Region Proposals 也更加可信。 

3. 同时描述了全局特征和局部特征的语义信息，在 RAB 的组装上具有很强的可操作性，通过配置分层式地组装 RAB 模块，以及修改特征模板，特征的表达会更加灵活。

#### 点评
这个区域分解集成的算法令我觉得跟以前传统的人脸识别算法提取 Haar-like 特征有点异曲同工之处，同样都是把特征图分成上下两部分，然后做特征提取操作，都是定义了特定的特征模板，这就很容易理解为什么作者要做 multi scale 的操作了，因为在以前使用 Haar/SIFT/HoG 的时候，往往都需要使用 muti scale 来检测。 

但是 R-DAD 为什么对特征只分成上下各一半，左右各一半这种特征模板，文章并没有给出令人信服的理由。尽管如此，这也是一个对目标检测的改进方向，通过 MRP 和 RDA 模块代替了之前的单纯的 RPN 网络，而且在不使用 FPN (Feature Pyramid Networks) 的情况下取得了不错的 mAP，这样看来 R-DAD 是 2 stage 目标检测系列的另一种技巧，综合了横向空间上的语义信息。

