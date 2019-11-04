#### Mask Scoring R-CNN (MS R-CNN)
- 论文地址： https://arxiv.org/pdf/1903.00241.pdf
- 作者： Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang
- 机构： 华科和地平线
- 代码地址： https://github.com/zjhuang22/maskscoring_rcnn 
- 介绍地址： https://zhuanlan.zhihu.com/p/58291808
- CVPR2019

#### 介绍
这篇论文从实例分割中mask 的分割质量角度出发，提出过去的经典分割框架存在的一个缺陷：用Bbox bounding box的classification confidence作为mask score，导致mask score和mask quality不配准。因此文章基于Mask R-CNN提出一个新的框架Mask Scoring R-CNN，==能自动学习出mask quality，试图解决不配准的问题==。

在实例分割（instance segmentation）中，比如Mask R-CNN，mask 分支的分割质量（quality）来源于检测分支的classification confidence。Mask R-CNN其实Faster R-CNN系列的延伸，其在Faster R-CNN的基础上添加一个新的分支用来预测object mask，该分支以检测分支的输出作为输入，mask的质量一定程度上依赖于检测分支。这种简单粗暴的做法取得了SOTA的性能，近年来COCO比赛的冠军或者前几名基本是Mask R-CNN及其变体，但依然有上升的空间。更仔细的来讲，==Mask R-CNN存在的问题是：bounding box的classification confidence不能代表mask的分割质量。==classification confidence高可以表示检测框的置信度高（严格来讲不能表示框的定位精准），但也会存在mask分割的质量差的情况。高的分类置信度也应该同时有好的mask结果。

回到原始的初衷，文章希望得到精准的mask质量，那么如何评价输出的mask质量呢？

是AP，或者说是instance-level的IoU。这个IoU和检测用到的IoU是一个东西，前者是predict mask和gt mask的pixel-level的Intersection-over-Union，而后者则是predict box和gt box的box-level的Intersection-over-Union。所以一个直观的方法就是用IoU来表示分割的质量，那么让网络自己学习输出分割的质量也是简单直观的做法。学习出mask的IoU，那么最后的mask score就等于maskIoU乘以classification score，mask score就同时表示分类置信度和分割的质量。

作者在Mask R-CNN的基础上添加了一个MaskIoU分支用于预测当前输出的mask和gt mask的IoU。MaskIoU的输入由两部分组成，一是ROIAlign得到的RoI feature map，二是mask分支输出的mask。两者concat之后经过3层卷积和2层全连接输出MaskIoU。

training过程：box分支和mask保持不变，输出的mask先经过阈值为0.5的binarize，再计算binary mask和gt的IoU作为target，采用L2 loss作为损失函数，loss weight设为1，3个分支同时end-to-end训练。

inference过程：检测分支输出score最高的100个框，再送入mask分支，得到mask结果，RoI feature map再和mask送入MaskIoU分支得到mask iou，与box的classification score相乘就得到最后的mask score.

![1565861626517](C:\Users\j00496872\Desktop\Notes\raw_images\1565861626517.png)

Figure 3. Network architecture of Mask Scoring R-CNN. The input image is fed into a backbone network to generate RoIs via RPN and  RoI features via RoIAlign. The RCNN head and Mask head are standard components of Mask R-CNN. For predicting MaskIoU, we use the predicted mask and RoI feature as input. The MaskIoU head has 4 convolution layers (all have kernel=3 and the final one uses stride=2 for downsampling ) and 3 fully connected layers (the final one outputs C classes MaskIoU.)

实验结果，在COCO 2017 test集上，相对于Mask R-CNN，mask AP有1个点多的提升。同时作者还做了对比实验，验证不同的MaskIoU输入对性能的影响。

![1565861686028](..\raw_images\1565861686028.png)

文章列举了4种输入方式：
- target mask和ROI feature concattarget 
- mask和ROI feature 
- 相乘所有mask和ROI feature concattarget 
- mask和高分辨率的ROI feature concat

其网络结构示意图如下：

![1565861733524](..\raw_images\1565861733524.png)

验证不同training target对性能的影响：
只学习target类别的MaskIoU,忽略其他类别学习所有类别的MaskIoU，相应的其他类别的MaskIoU的学习目标就是0学习出现在ROI区域的类别的MaskIoU。可以看出，setting#1的效果最好，setting#2的效果最差。

![1565861753083](..\raw_images\1565861753083.png)

同时作者还做了实验探索Mask Scoring R-CNN的性能上界。对每个输出的MaskIoU，用输出的mask 和匹配的gt mask iou来代替，MaskIoU分支就输出了mask分支的真实quality，这时就得到了Mask Scoring R-CNN的性能上界。
实验结果表明Mask Scoring R-CNN依然比Mask R-CNN更好，说明MaskIoU起到了alignment的效果，但很显然会比用gt mask iou 代替的效果差，说明一方面box的准确性和mask分支本身也会影响mask任务的性能，另一方面MaskIoU 分支的学习能力可以进一步提升，Mask Scoring R-CNN依然有提升的空间。
速度方面，作者在Titan V GPU上测试一张图片，对于ResNet18-FPN用时0.132s，Resnet101-DCN-FPN用时0.202s，Mask Scoring R-CNN和Mask R-CNN速度一样。

![1565861781714](..\raw_images\1565861781714.png)

总结：==作者motivation就是想让mask的分数更合理，从而基于mask rcnn添加一个新的分支预测来得到更准确的分数，做法简单粗暴，从结果来看也有涨点。==其实mask的分割质量也跟box输出结果有很大关系，这种detection-based分割方法不可避免，除非把detection结果做的非常高，不然mask也要受制于box的结果。这种做法与IoU-Net类似，都是希望直接学习最本质的metric方式来提升性能。
