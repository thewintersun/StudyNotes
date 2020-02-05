### Objects as Points

论文地址：https://arxiv.org/abs/1904.07850

作者：Xingyi Zhou, Dequan Wang, Philipp Krähenbühl

机构：UT Austin, UC Berkeley

代码地址：https://github.com/xingyizhou/CenterNet



#### 摘要

In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. ==Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose.== 

Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. 

==CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS.== 

We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

#### 介绍

这篇文章直接预测物体的中心点、中心点的偏移量、box的长宽。相比于CornerNet、ExtremeNet没有Grouping等一系列繁琐的操作，==最大的特点是没有NMS操作==。

![1575256906974](D:\Notes\raw_images\1575256906974.png)

个人觉得这个点挺有意思的。具体来说，对于一张WxHx3的输入图片，经过FCN（ResNet、Hourglass）之后，得到 $\frac{W}{R} \times \frac{H}{R} \times C $的heatmap，其中C表示类别数，R表示网络的stride。

这个heatmap表示当前位置点是物体的中心点的概率。在计算gt的时候，首先将bounding box的中心点按照stride映射到heatmap上，然后==利用高斯核函数（如下公式）将中心点周围的像素点给不同的惩罚值，这样heatmap上每个点都有对应的gt==。==如果在同一个类中，heatmap上的点的高斯核函数有重合部分，取最大值。==

![1575257420217](D:\Notes\raw_images\1575257420217.png)

这里像素点的目标函数采用FocalLoss：

![1575257537773](D:\Notes\raw_images\1575257537773.png)

由于网络stride的存在，在计算量化的时候会带来误差，这个同RoI pooling一样的道理。为了消除量化误差，网络额外输出通道数为2的heatmap，尺寸大小为$\frac{W}{R} \times \frac{H}{R} \times 2$，在WxH空间上一个点有两个数值分别表示偏差的（x,y）坐标。在计算gt的时候是直接根据量化的误差来算的。

这里Offset的目标函数采用L1Loss：

![1575257687357](D:\Notes\raw_images\1575257687357.png)

有了中心点就有了物体的位置坐标，但还需要物体的长宽，所以网络还会输出通道数为2的heatmap，尺寸大小同样为$\frac{W}{R} \times \frac{H}{R} \times 2$，其对应的gt就是原来的bounding box的长和宽。

这里Size的目标函数采用L1Loss：

![1575257845446](D:\Notes\raw_images\1575257845446.png)

总的来说，网络输出是一张尺寸为$\frac{W}{R} \times \frac{H}{R} \times \left( C + 4 \right)$的heatmap。==在inference的时候，分别在每个类别中取topk个峰值点，这个峰值点的定义是周围8个点都比它小或相等。==这就像关键点检测一样，每一个关键点都是一个局部峰值。==这些峰值点的数值表示检测框的score==，检测框的坐标即decode过程很简单直接：峰值点的坐标加上预测的偏移量再加上预测的长宽。

总体的Loss函数：![1575258044075](D:\Notes\raw_images\1575258044075.png)， $\lambda_{size} =0.1, \lambda_{off}=1$

文章利用关键点检测的思路，将检测问题转化为求物体中心点位置的问题。一般情况下，两个很相近的物体中心点（距离大于4像素）会映射到heatmap上不同的位置，==同样的，heatmap上不同位置的点对应不同的物体，挑选峰值点的过程相当于实现了NMS操作。==之前一直在思考如何将NMS操作融合进网络中，让网络自动实现NMS，也许这个思路有助于启发，未来不需要NMS也不一定。==另外，对于两个物体中心点高度重合的情况，这种方法没法解决，但这种情况出现的很少，也不需要关注这种极端情况。==

![1575258183278](D:\Notes\raw_images\1575258183278.png)

Figure 4: Outputs of our network for different tasks: top for object detection, middle for 3D object detection, bottom:for pose estimation. All modalities are produced from a common backbone, with a different 3\*3 and 1\*1 output convolutions separated by a ReLU. The number in brackets
indicates the output channels. 

#### 实验结果

最后从效果来看，CenterNet超过了绝大部分two-stage和one-stage模型，如下图所示，CenterNet将RetinaNet和YOLOv3都包裹在内，在精度和速度上都取得了不错的效果。

![1568178024922](D:\Notes\raw_images\1568178024922.png)

![1575257118273](D:\Notes\raw_images\1575257118273.png)

