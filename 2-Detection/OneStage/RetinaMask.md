### RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free

论文地址：https://arxiv.org/abs/1901.03353

作者：Cheng-Yang Fu, Mykhailo Shvets, Alexander C. Berg

机构：Computer Science Department of UNC at Chapel Hill

代码地址：https://github.com/chengyangfu/retinamask



#### 摘要

This paper brings singleshot detectors up to the same level as current two-stage techniques. We do this by improving training for the state-of-the-art single-shot detector, RetinaNet, in three ways: ==integrating instance mask prediction for the first time==, ==making the loss function adaptive and more stable==, and ==including additional hard examples in training==. We call the resulting augmented network RetinaMask. 

The detection component of RetinaMask has the same computational cost as the original RetinaNet, but is more accurate.



#### 介绍

![1566893665897](C:\Users\j00496872\Desktop\Notes\raw_images\1566893665897.png)

**1）IoU threshold的选择问题**

正常的过程是IoU，[0.5，1] 为正样本，[0, 0.4]为负样本， (0.4, 0.5) 舍弃。作者认为这样做舍弃了很多 outliers 样本，尤其是有一个边特别大的样本。作者做了很多实验来调整threshold的选择。

**2） Self-Adjusting Smooth L1 Loss**

正常的Smooth L1 Loss过程是这样的，a point splits the positive axis range into two parts: L2 loss is used
for targets in range $[0, \beta]$ , and L1 loss is used beyond to avoid over-penalizing outliers.

![1566894580820](C:\Users\j00496872\Desktop\Notes\raw_images\1566894580820.png)

Figure 2: Smooth L1 and Self-Adjusting Smooth L1. In Smooth L1 Loss (a)  if a fixed threshold that separates
loss into L2 and L1 regions. In the proposed Self-Adjusting Smooth L1 (b),==the  $\beta$   is calculated as the difference between running mean and running variance of L1 loss and the value is clamped to the $[0, \hat{\beta}]$ range. The $\beta$ is approaching 0 during training==.

![1566894802556](C:\Users\j00496872\Desktop\Notes\raw_images\1566894802556.png)

![1566894828943](C:\Users\j00496872\Desktop\Notes\raw_images\1566894828943.png)

**3) Mask Prediction Module**

Mask的预测过程：==将检测的结果当做RPN的Proposal结果，然后将Top-N predicted bbox 按照FPN的分配公式分配到不同的层中==，然后 RoIAlign提取特征。 

​	![1566895324682](C:\Users\j00496872\Desktop\Notes\raw_images\1566895324682.png)

In our final model, we use the { P3, P4, P5, P6, P7 } feature layers for bounding box predictions and { P3, P4, P5}  feature layers for mask prediction.

![1566895528341](C:\Users\j00496872\Desktop\Notes\raw_images\1566895528341.png)



**实验结果**

![1566895981004](C:\Users\j00496872\Desktop\Notes\raw_images\1566895981004.png)

与RetinaNet的对比，R代表RetinaNet, O(B) 代表 Ours(BBOX), O(M) 代表Ours(MASK) 。 

==可以发现Mask的结果，比BBOX的结果低很多。==

Our speed number is evaluated on Nvidia 1080 Ti / PyTorch1.0 and FocalLoss results are evaluated on Nvidia M40 / Caffe2.

![1566896224939](C:\Users\j00496872\Desktop\Notes\raw_images\1566896241185.png)

Table 5: Comparison with state-of-the-art methods on COCO test-dev. Compared to RetinaNet [25], our model based on ResNet-101-FPN is better by 2.6 mAP. Compared to Mask R-CNN [18], our model shows 3.5 mAP improvement.
