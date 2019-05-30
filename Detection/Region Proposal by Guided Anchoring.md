#### Region Proposal by Guided Anchoring
- 论文地址：https://arxiv.org/abs/1901.03278
- 作者： Jiaqi Wang, Kai Chen, Shuo Yang, Chen Change Loy, Dahua Lin
- 机构： 商汤，香港中文大学，南洋理工大学
- 代码地址： https://github.com/open-mmlab/mmdetection  尚未开放
- 作者介绍地址： https://zhuanlan.zhihu.com/p/55854246 


#### 概述
我们提出了一种新的 anchor 生成方法 —— Guided Anchoring，即通过图像特征来指导 anchor 的生成。通过预测 anchor 的位置和形状，来生成稀疏而且形状任意的 anchor，并且设计了 Feature Adaption 模块来修正特征图使之与 anchor 形状更加匹配。在使用 ResNet-50-FPN 作为 backbone 的情况下，Guided Anchoring 将 RPN 的 recall（AR@1000） 提高了 9.1 个点，将其用于不同的物体检测器上，可以提高 mAP 1.2 到 2.7 个点不等。

下图是我们的方法和传统 RPN 的性能和速度对比，可以看到要显著优于传统 RPN。

<img src='https://pic1.zhimg.com/80/v2-6dd934af891028e745860b6bc9bf1b58_hd.jpg' />

下面是应用在不同检测方法上的结果，backbone 均为 ResNet-50-FPN。

<img src='https://pic3.zhimg.com/80/v2-f19c8cc61d062a7efc126088a60f998a_hd.jpg' />

#### 背景
Anchor 是物体检测中的一个重要概念，通常是人为设计的一组框，作为分类（classification）和框回归（bounding box regression）的基准框。无论是单阶段（single-stage）检测器还是两阶段（two-stage）检测器，都广泛地使用了 anchor。例如，两阶段检测器的第一阶段通常采用 RPN 生成 proposal，是对 anchor 进行分类和回归的过程，即 anchor -> proposal -> detection bbox；大部分单阶段检测器是直接对 anchor 进行分类和回归，也就是 anchor -> detection bbox。

常见的生成 anchor 的方式是滑窗（sliding window），也就是首先定义 k 个特定尺度（scale）和长宽比（aspect ratio）的 anchor，然后在全图上以一定的步长滑动。这种方式在 Faster R-CNN，SSD，RetinaNet 等经典检测方法中被广泛使用。

#### Motivation
通过 sliding window 生成 anchor 的办法简单可行，但也不是完美的，不然就不会有要讲的这篇 paper 了。首先，anchor 的尺度和长宽比需要预先定义，这是一个对性能影响比较大的超参，而且对于不同数据集和方法需要单独调整。如果尺度和长宽比设置不合适，可能会导致 recall 不够高，或者 anchor 过多影响分类性能和速度。一方面，大部分的 anchor 都分布在背景区域，对 proposal 或者检测不会有任何正面作用；另一方面，预先定义好的 anchor 形状不一定能满足极端大小或者长宽比悬殊的物体。所以我们期待的是稀疏，形状根据位置可变的 anchor。

