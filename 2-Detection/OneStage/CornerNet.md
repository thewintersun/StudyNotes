### CornerNet: Detecting Objects as Paired Keypoints

论文地址：https://arxiv.org/abs/1808.01244

作者：Hei Law, Jia Deng

机构：Princeton University, Princeton, NJ, USA

代码地址：https://github.com/princeton-vl/CornerNet

参考文章：https://zhuanlan.zhihu.com/p/58183628



#### 摘要

CornerNet将目标BBOX的检测作为Keypoint检测，即左上角和右下角。用关键点的方法来做目标检测。引入了corner pooling，这是一种新型的池化层，可以帮助网络更好地定位corners。 实验表明，CornerNet在MS COCO上实现了42.2％的AP，优于所有现有的单级探测器。



#### 介绍

基于Anchor的方法有两个缺点：

1. 造成的AnchorBox很多，导致严重的正负样本不平衡，并且训练速度减慢。
2. 引入了很多超参，需要预先定义，并且与训练集相关联。

Anchor-Free的方法，代表算法有CornerNet、ExtremeNet、FSAF（CVPR2019）

先来看看CornerNet是怎么做的。CornerNet顾名思义，预测左上角和右下角来得到最后的bounding boxes。所以最基本的需要两个feature map表示对应的Corner，文中称为heatmap。如下图所示，两个heatmap分别表示top-left（左上角）和bottom-right（右下角）的corner，作者借鉴Associative Embedding method （NIPS2017） 利用了两个embedding层做grouping操作，将属于同一个object的corner聚合都一起得到最后的bounding box。

![img](https://pic1.zhimg.com/80/v2-783a301079757990f683bddcfe98e4fc_hd.jpg)

模型的整体结构如下图，==backbone采用hourglass network==以得到高质量的feature map，接两个分支分别预测top-left Corner 和bottom-right Corner，两个分支的结构完全一样。

![img](https://pic2.zhimg.com/80/v2-c45ede4a4339d90e96e4735bb4a6e68d_hd.jpg)

**Heatmap分支的设计**

heatmap的尺寸大小为（H,W,C）,C 是类别数，一个channel对应一类物体。H和W和输入图片 $(\tilde{H}，\tilde{W}) $ 一般满足关系： $H=\frac{\tilde{H}}{Stride},W=\frac{\tilde{W}}{Stride} $，文中Stride为4。

ground-truth corner 在heatmap上会有一个对应的位置，按常理，这个位置是负责预测这个gt最好的点，然而需要注意的是，==偏离这个位置附近的点也对预测gt有贡献，即使不是百分之百重合，但围成的框和gt有较大的IoU，这些点不应该被忽视==。但偏离距离不同的点却有不同的“重要度”，==作者加入了一项penalty reduction，合理地对gt进行reweight==。

heatmap对应的loss函数如下，==采用了focal loss的变体形式==，其中 $P_{cij}$ 是位置为 $（i，j）$ 的输出类别为 c 的scores， $y_{cij}$  是对应的 gt 值，这个gt值被 penalty reduction 进行惩罚。

![img](https://pic1.zhimg.com/80/v2-eb6300c9a958bcd99344c85ffaab45f0_hd.png)

**Offset分支的设计**

由上面的介绍，==heatmap和原图输入有一个stride大小的比例关系，在将heatmap上的点映射到原图尺寸上时会有位置误差。==假设原图上位置为（i，j）的点，在heatmap上对应的位置变为 $（\lfloor \frac{x}{n} \rfloor, \lfloor \frac{y}{n}\rfloor）$ , n是下采样stride。这样在映射的过程中会产生 $（\frac{x}{n}- \lfloor \frac{x}{n} \rfloor）$  的误差。因此，模型除了预测heatmap，还同时预测了弥补映射过程中带来的误差offset分支 $O_k=\left(\frac{x_k}{n}- \lfloor \frac{x_k}{n} \rfloor， \frac{y_k}{n}- \lfloor \frac{y_k}{n} \rfloor  \right) $ ,  $O_k$ 是offset， $x_k$, $y_k$ 是object k对应的corner。并且offset分支对所有类别共享，采用smooth L1函数进行训练。

![img](https://pic3.zhimg.com/80/v2-13e10632c21362a9383b9b5ac706759a_hd.jpg)

**Grouping Corner的设计**

文章设计的出发点是：==来自同一个object的Corner距离应该尽可能接近==。为了表示Corner的距离，作者引入了一个embedding分支，该分支在空间上的尺寸和heatmap保持一样，通道数可以自己设定，即heatmap上每一个位置的Corner由一个embedding vector表示，这个vector隐式地表示了和其他Corner的距离。来自同一个object的Corner其对应的embedding vector之前的距离应该尽可能小。基于embedding vector之间距离就可以对Corner进行grouping操作。距离的具体的数值并不重要，这是一个相对概念，只要能区分不同object即可。

假设object k的 top-left Corner对应的embedding vector是 $e_{t_{k}}$  ,bottom-right Corner对应的embedding vector是 $e_{b_k}$ ，==embedding vector的训练目的就是使得同一个object的距离越来越近，不同object的距离越来越远，即类似于“类间间距大，类内间距小”==，因此最后的loss函数如下：

![img](https://pic3.zhimg.com/80/v2-db03cd6e9920ff53f45bfebf259b497e_hd.jpg)

**Corner Pooling 的设计**  ground-truth对应的Corner必然会存在不在物体上面的情况，这种情况下，==Corner周围都是背景区域，对Corner的预测很不利==。从本质上来说，左上角和右下角的Corner其实根据object的边界信息得到。为了利用物体边界信息，需要一个操作可以将物体边界信息聚合到Corner点上，因此就有了pooling操作。

pooling操作很简单，==对每个点，向水平和垂直方向进行pooling操作，这样同一个水平和垂直方向上就包含了该方向的物体边界信息。==以 top-left Corner 为例：

![img](https://pic4.zhimg.com/80/v2-3fdf3e09ce0321f517d13a9ab3198baf_hd.jpg)

**实验效果**

CornerNet在COCO上达到了SOTA效果。同时作者还在Corner polling、penalty reduction、error analysis上面进行了对比实验。

![img](https://pic1.zhimg.com/80/v2-052ecdb582c069acaed01b9f3c5afe04_hd.jpg)

**总结**

Anchor free based method一般采用bottom-up的思路，先生成带有类别信息和位置信息的feature map，再根据这些feature map得到最后的bounding boxes。总结起来，其关键部分包含一下几方面：

1. 如何生成高质量的feature map，这是得到精准的bounding boxes的前提，主要是backbone的设计。众所周知，CNN的多层stride会损失很多位置信息，而这些信息对于pixel-level的任务至关重要。
2. 因为没有提前设定feature map上哪些points负责预测哪个ground-truth box，需要一个grouping操作将属于同一个object的points划分到一起。
3. 训练过程loss的设计。loss函数既需要考虑预测的feature map能学到discriminative的类别置信度，还需要输出准确的位置信息。

CornerNet和ExtremeNet都尝试==从keypoint的角度做detection，摆脱了anchor的繁琐设置，以及先验anchor带来的bias==。从标注成本的角度来讲，CornerNet只需要bounding box标注，ExtremeNet需要instance segmentation mask的标注，标注成本相对较高。同时，keypoint需要更细节的位置信息，对backbone的要求较高，所以hourglass network这类对keypoint较友好的网络成为了第一选择，带来的==问题是检测速度较慢==。

目前anchor-free的文章越来越多，很多人开始考虑从keypoint、segmentation的角度入手做detection，但想要取得像Faster R-CNN、SSD、RetinaNet的实用性和效果还有很长的路要走。

参考文献：

- Associative embedding: End-to-end learning for joint detection and grouping
- Deep extreme cut: From extreme points to object segmentation   

