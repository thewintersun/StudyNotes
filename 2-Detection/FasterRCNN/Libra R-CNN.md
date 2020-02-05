### Libra R-CNN: Towards Balanced Learning for Object Detection
- 论文地址：https://arxiv.org/abs/1904.02701
- 作者：Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin
- 机构： 商汤，香港中文大学
- 代码地址： https://github.com/open-mmlab/mmdetection
- 介绍文章： https://mp.weixin.qq.com/s/ts4WFnuN4cHLfUh8--96Kw
- CVPR 2019 

#### 摘要
In this work, we carefully revisit the standard training practice of detectors, and find that the detection performance is often limited by the imbalance during the training process, which generally consists in three levels – sample level, feature level, and objective level. To mitigate the adverse effects caused thereby, we propose Libra R-CNN, a simple but effective framework towards balanced learning for object detection. It integrates three novel components: **IoU-balanced sampling, balanced feature pyramid, and balanced L1 loss**, respectively for reducing the imbalance at sample, feature, and objective level.

现有的Faster RCNN检测框架的问题有：

1. 正负样本不均。
2. 难易样本不均。
3. 视野域问题。
4. FPN特征层选择问题。

这篇论文是修补类型的，主要分析三个方面的不均衡问题：sample level, feature level, and objective level。从而提出了三个方面的改进：

#### 三个不平衡

纵观目前主流的目标检测算法，无论 SSD、Faster R-CNN、Retinanet 这些的 detector 的设计其实都是三个步骤： 
- 选择候选区域 
- 提取特征 
- 在 muti-task loss 下收敛 

往往存在着三种层次的不平衡： 
- sample level
- feature level
- objective level 

这就对应了三个问题： 
- 采样的候选区域是否具有代表性？
- 提取出的不同 level 的特征是怎么才能真正地充分利用？
- 目前设计的损失函数能不能引导目标检测器更好地收敛？

![1565861170709](D:\Notes\raw_images\1565861170709.png)

**1. IoU-balanced sampling**

RPN Proposal过程中，作者统计得到60% 的**hard negative**都落在 IoU 大于 0.05 的地方，但是RPN随机采样机制只提供了 30%，也就是说70%随机采样提供的候选框IoU值都在[0, 0.05] 之间的，这是因为，目标检测中背景往往占了一张图绝大多数的位置，这就导致了大量的易负样本。 *所以这本质上还是难易样本不均的问题。*

Ross Girshick 后面提出了 OHEM（online hard example mining，在线困难样本挖掘）是一个hard negative mining的一种好方法，但是**这种方法对噪音数据会比较敏感**。

作者解决方法：按IoU值划分区域，如果一共还是要采样 N 个，通过 IoU 的值划分为 K 个区间，每个区间中的候选采样数为$M_k$ ，实验表明K参数不敏感。

**2. balanced feature pyramid**

FPN 结构中含有多层特征，低层特征分辨率高往往学习到的是细节特征，高层特征分辨率低学习到语义特征，如何更有效地利用不同层的特征信息？

作者解决方法：1） 通过插值或池化，resize 到中间层大小，比如C[2-5] 取C4层。 2）特征相加取平均值。 3）Refine，可以通过卷积或者non-local模块，但non-local模块更稳定。4）最后通过插值或池化，加到各个层中去。简单而有效。

**3. balanced L1 loss**

在Faster RCNN中目标函数 $L = L_{cls}+\lambda L_{loc}$ , 设定 L 值大于1.0的样本为outliers（困难样本）, 小于1.0的为inliers（容易样本）。 Loss为分类和回归两者相加，如果分类做得很好，回归的重要性就会被忽略，如果通过调节 $\lambda$ 的值来解决，又会使得网络对outliers样本更为敏感。并且作者分析得出，inliers样本对Loss值只贡献30%的梯度。

首先我们看 Smooth L1 Loss：

![1565861267621](D:\Notes\raw_images\1565861267621.png)

作者解决方法：Balanced L1 Loss, clip the large gradients produced by outliers with a maximum value of 1.0。

![1565861254654](D:\Notes\raw_images\1565861254654.png)

根据梯度反求出 Lb(x) 表达式：

![1565861235509](D:\Notes\raw_images\1565861235509.png)

实验结果：三个改进加起来相对FPN 结构Faster RCNN有2+个点的提升。