### Semantic Instance Meets Salient Object: Study on Video Semantic Salient Instance Segmentation

论文地址：https://arxiv.org/abs/1807.01452

作者：Trung-Nghia Le, Akihiro Sugimoto

网址：https://sites.google.com/view/ltnghia/research/sesiv 

项目地址：

https://github.com/ltnghia/SESIV matlab code

 https://github.com/davisvideochallenge/davis2017-evaluation



#### 摘要

文章旨在推进视频显著实例分割的发展，推出新的视觉任务： 视频语义显著实例分割（video semantic salient instance segmentation， VSSIS)， 也就是 语义实例 - 显著目标（Semantic Instance - Salient Object， SISO)，融合了两种分割任务：语义实例分割，显著目标分割。

![1568704321054](D:\Notes\raw_images\1568704321054.png)

- a) 显著区域是一个前景和背景的二分类，前景有多个类别物体并区分。
- b) 显著实例在显著区域的基础上，将多个类别的前景进行区别。
- c) 语义显著实例在区分多个类别实例的基础上，给每个类别添加语义信息，即属于哪个类别信息。也是本文的研究内容。

SISO基于两个不同的分割任务，即语义实例分割(semantic instance segmentation) 和显著物体分割(salient
object segmentation)。

- In SISO, we introduce a sequential fusion by looking at overlapping pixels between semantic instances and salient regions to have non-overlapping instances one by one. 
- We also introduce a recurrent instance propagation to refine the shapes and semantic meanings of instances, 
- and an identity tracking to maintain both the identity and the semantic meaning of instances over the entire video. 

==SISO possesses three key features: sequential fusion, recurrent instance propagation, and identity tracking.==

实验证明我们的SISO基线方法可以处理视频中的遮挡问题。为了处理VSSIS任务，我们在DAVIS-2017 benchmark dataset 上，通过为显著实例标签添加语义信息，扩展成数据集 **SEmantic Salient Instance Video (SESIV)** ，包括84个带标注的高质量视频。



#### 背景介绍

视频显著目标分割方法包括:

> T.-N. Le and A. Sugimoto. Deeply supervised 3d recurrent fcn for salient object detection in videos. In BMVC, 2017.
>
> G. Li, Y. Xie, T. Wei, K. Wang, and L. Lin. Flow guided recurrent neural encoder for video salient object detection. In CVPR, 2018.
>
> J. Li, C. Xia, and X. Chen. A benchmark dataset and saliency-guided stacked autoencoders for video-based salient object detection. IEEE TIP, 27(1):349–364, 2018.
>
> W. Wang, J. Shen, and L. Shao. Video salient object detection via fully convolutional networks. IEEE TIP, 27(1):38–49, Jan 2018.

> G. Li, Y. Xie, L. Lin, and Y. Yu. Instance-level salient object segmentation. In CVPR, 2017.

当前工作的挑战：

The problem is even more challenging on the videos because ==instances need to be tracked over the entire video to maintain their identifications== even if they are occluded at some frames. 

We remark that in this paper, ==an instance in a video is defined to be salient if it appears in the first video frame and stands out for more than 50% duration of the video in total==.

显著目标分割（SOS）的相关数据集列表：

![1568706629678](D:\Notes\raw_images\1568706629678.png)

**语义实例分割**（Semantic instance segmentation，SIS）任务包含了目标检测和语义分割任务，有两种实现方法：

1. 基于分割的方法：首先进行分割，然后将分割的实例进行聚类。

2. 基于RPN的方法：首先进行BBOX的预测，然后根据BBOX获取Mask区域，最后利用检测来对Mask区域进行分类（Mask RCNN）。

   

**视频显著目标分割**（Video salient object segmentation, VSOS）可以分为两种实现方法：

1. 基于分割的方法：首先将视频的每一帧进行分割成各个区域，然后利用每个区域提取的深层特征进行显著性推断。
2. 端到端显著性推断方法：使用全卷积网络(FCNs)，利用光流（optical flow）或三维内核（3D kernels ）来实现。

端到端显著性推理方法比基于分割的方法具有更好的性能，使用3D kernels处理的帧数比使用光流处理的帧数多，可以更好地融合时间信息。所以作者采用了下面的方法做VSOS:

> T.-N. Le and A. Sugimoto. Deeply supervised 3d recurrent fcn for salient object detection in videos. In BMVC, 2017.



#### **SESIV数据集**

由 84个视频组成，185个语义显著实例，被分为29个类。训练集包括58个视频 (136个实例)，测试集由26个视频组成(49个实例和14个类别)。 类别包含在MSCOCO的类别种类中。

> They are person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, bird, cat, dog, horse, sheep, cow, elephant, bear, backpack, snowboard, sports ball, kite, skateboard, surfboard, tennis racket, chair, tv, remote, cell phone, and clock.

对于每个视频帧，我们提供了3种基本Ground Truth 标签(即，显著性标签，实例标签，语义标签，如图3所示)。

![1568778286433](D:\Notes\raw_images\1568778286433.png)

数据集采集过程：

![1568781766655](D:\Notes\raw_images\1568781766655.png)

They are background clutter, dynamic background, deformation, appearance change, shape complexity, small instance, **occlusion**, **out of view**, motion blur, and fast motion. 实例会出现在视频中消失的现象，由于完全遮挡和离开视野。

数据集统计分析：

![1568781531684](D:\Notes\raw_images\1568781531684.png)



#### 基线方法介绍

![1568781470555](D:\Notes\raw_images\1568781470555.png)

首先SISO包含两个分支(e.g. SIS and SOS) ，==融合这两个输出流来去除非显著目标实例==，得到像素级的实例分割Mask。在融合两个特征流之前，会进行mask的Refine工作，==采用boundary snapping method [3]==, 最后采用identity tracking 保证整个视频中实例标签的一致性。

> S. Caelles, K.-K. Maninis, J. Pont-Tuset, L. Leal-Taix´e, D. Cremers, and L. Van Gool. One-shot video object segmentation. In CVPR, 2017.

SIS分支即语义实例分割, 可用的算法都是图片分割，因此针对视频可以每一帧都检测，==使用recurrent instance propagation算法==提高实例分割的准确率。 文中SIS分支采用了Mask R-CNN和MNC进行了对比实验，Mask-RCNN效果更优。

![1568793163472](D:\Notes\raw_images\1568793163472.png)

- Morg is the original model (we applied this frame-by-frame for videos), 
- Mprop is the model incorporating our identity propagation module (this is just to simply exploit temporal information), 
- MSISO is the model incorporated in our proposed SISO.

SOS分支即显著性分割, 采用了3D卷积核进行视频分割，文中==使用了3D FCN==。

> T.-N. Le and A. Sugimoto. Deeply supervised 3d recurrent fcn for salient object detection in videos. In BMVC, 2017.

#### Sequential Fusion

如何==处理不同实例的Overlap问题==，是个比较棘手的问题，作者采用SOS中的信息进行处理。

**计算过程：**首先，作者计算与Salient Object Mask IoU值最大的Instance，然后去除该Instance在Salient Object Mask的区域，然后依次计算其他Instance在Salient Mask中的区域，继续去除，直到所有的Instance都计算完成。如果，Instance与 Salient Mask 的IoU值小于0.1，作者认为该Instance其实没有被Present出来。

![1568860709825](D:\Notes\raw_images\1568860709825.png)

我们还通过对Frame中所有语义显著实例的置信度求平均值来计算每个Frame的置信度。Salient Instance的置信度是 IoU值 和 Classify 值的一个混合：此处设置$\beta^2=0.3$， 所以IoU的score值权重更重些。

![1568861554701](D:\Notes\raw_images\1568861554701.png)

#### Recurrent Instance Propagation

问题：由于剧烈运动或者相机的运动，导致有些==Instance有严重的变形==。所以作者使用Recurrent Instance Propagation来将Instance recurrently propagated to 相邻的Frame。

![1568861890070](D:\Notes\raw_images\1568861890070.png)

图7: Recurrent Instance Propagation的一次迭代的流程图。语义实例从高 frame-confidences 的视频帧传播到 frame-confidences的视频帧。带有黄色边框的视频帧具有比相邻帧更高的frame-confidences。

**计算过程：**首先，计算Video中的所有帧的置信度，上面的Salient Instance的置信度的平均值，然后降序排列。如果视频帧具有比相邻帧更大的帧置信度，则使用Flow Warping/Inverse Flow Warping将帧的实例传播到下一帧/前一帧，其中Flow使用FlowNet2计算。

> E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In CVPR, 2017.

在Recurrent Instance Propagation迭代完成后，更新被Propagated帧的新的置信度值。然后对视频中的每一帧的置信度计算平均值，作为Video的置信度，直到Video的置信度收敛。作者观察大概在5个Epoch后可以有效。

#### Identity Tracking

问题：如何维持视频中Instance的==Label的一致性问题==。短期的一致性，以及消失与重现的ReID问题。

![1568862819672](D:\Notes\raw_images\1568862819672.png)

图8:身份跟踪模块流程图。实例视频帧的Label通过Flow Wraping被传播到它的下一帧。当一个实例被遮挡时或者在框架之外,它会在下一个帧中进行一次ReID，通过在关键帧 k 中提取的特性。 

**Identity Propagation:** 计算两个帧之前的实例的IoU值，IoU值最大的两个实例，应该是同一个实例，也就是Label应该相同，如果一个实例，在目标帧中没有一个实例与它的IoU值超过0.7，那么我们就认为它消失了。。。（这里IoU=0.7是个很经验的值。。。）

**ReID 过程：** 首先从前面的帧中，选择一个包含该Instance的Key Frame，然后通过该Instance的BBOX提取出该区域的特征，作为查询特征。然后在目标帧中，采用Faster RCNN弹出各个Proposal，然后提取各个Proposal在目标帧中的特征，采用余弦相似度计算查询特征和Proposal特征的相似度，取相似度最高的那个Proposal 区域。下一步，计算Proposal region 和目标帧中已经分割好的所有的Instance的IoU，如果IoU值大于0.7，那么这个Instance的Label就应该是查询特征的对应的Label。

语义标签的统一： 对于语义显著的实例和类别，我们首先计算实例所属类别的分类分数在整个视频中的总和。然后，我们为实例选择在所有类别中达到最大值的类别的语义标签。通过这种方式，附加在显著实例上的语义标签在整个视频中是统一的。



#### 实验结果

评价标准：==语义区域相似度==和==语义轮廓相似度==. 假设 m 和 g 是预测Mask和ground truth Mask结果，那么语义区域相似度和语义轮廓相似度的计算方式如下：

![1568865167938](D:\Notes\raw_images\1568865167938.png)

注意：我们只比较具有相同的标识和相同的语义标签实例的相似性。

各个Tricks的对比实验效果：

![1568864197688](D:\Notes\raw_images\1568864197688.png)

显然，identity tracking对结果的影响是非常大的，Recurrent Instance Propagation的效果也比较明显。