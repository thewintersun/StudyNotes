### Repulsion Loss: Detecting Pedestrians in a Crowd

论文地址：https://arxiv.org/pdf/1711.07752.pdf

机构：同济大学，北京大学，旷视

代码地址(作者)：https://github.com/bailvwangzi/repulsion_loss_ssd

他人实现：https://github.com/rainofmine/Repulsion_Loss

知乎文章：https://zhuanlan.zhihu.com/p/41288115



本文由同济大学和北京大学合作发表于CVPR2018，聚焦于loss层面，为遮挡情况下的行人检测问题提供了一种行之有效的解决方案。

#### 背景介绍

与通用目标检测相比，遮挡情况在行人检测中更为普遍，为此也是行人检测领域最广为关注的问题之一。现实场景中行人的遮挡情况主要分为两种情况：一种是其他物体对行人的遮挡，这往往会带来目标信息的缺失，进而导致漏检；另一种是行人个体之间的相互遮挡，这往往会引入大量的干扰信息，进而导致更多的虚检。==本文重点解决的是后一种情况导致的遮挡问题==。作者将对这一问题进行了深入思考，并从loss的层面提出了一种新颖的解决方案，为我们呈现了一场思路和实验都十分精彩的盛宴。



#### 行人检测的遮挡问题

本文重点解决的问题是：在行人个体之间的相互遮挡时，如何提高行人检测的准确率。无论是在自动驾驶还是智能监控场景下，行人以群体形式出现的情况时有发生，如图1所示：两个行人个体之间存在着相互遮挡，而这种遮挡对检测算法的影响体现在两个层面：

- 一是目标框定位不准确，如图中红色框；
- 二是算法对NMS的阈值的选取更为敏感了，如果NMS阈值选取过小，则周围的目标框就被过滤掉了（图中蓝色框），但蓝色框对于图中女士而言却是正确的预测，如果NMS阈值选取过大，则又会带来更多的虚检。

对于这样一个两难问题，作者的解决思路在图中的式中也得到了直观体现：该思路的核心就是一种全新的loss，叫做Repulsion loss，其包括两个部分：==前者是使得预测框更接近目标框，后者是使得预测框要尽可能远离周围的目标框==。

![1565694254644](D:\Notes\raw_images\1565694254644.png)

作者首先研究了现有公开数据集CityPersons[1]中遮挡出现的情况以及这种情况对检测器性能的影响。在CityPersons验证集中，共有3157个行人标注框，其中48.8%的行人相互遮挡的IOU高于0.1，26.4%的行人相互遮挡的IOU高于0.3，可见这种遮挡情况的出现是极其普遍的。那么遮挡到底会给检测器带来什么样的影响？作者训练了Faster R-CNN检测器作为baseline对这个问题进行了回答，如图2所示：评估指标选用行人检测中常用的Miss Rate (MR，越低越好)，其中Reasonable-occ代表所有遮挡情况，Reasonable-crowd代表所有遮挡情况中自遮挡的部分，图中列出了在平均20，100，500个虚检情况下的missed detection，==从图中可以发现遮挡占据了近60%的席位（蓝色+橙色），而在这60%席位中，自遮挡又占据了近60%==。综上，图2足以说明：遮挡是影响行人检测性能的一个非常重要的因素，而行人之间的自遮挡更是重中之重。

![1565694392311](D:\Notes\raw_images\1565694392311.png)



#### Repulsion Loss

针对以上分析，作者决定从loss层面来解决行人之间的自遮挡问题，首先我们直观感受下本文方法RepGT的有效性，如下图所示：图（a）展示了RepGT==对漏检的有效性==，可以发现在detection score较高时，RepGT的漏检更少，图（b）展示了RepGT==对自遮挡情况下的虚检的有效性==，可以发现RepGT的虚检中自遮挡所导致的虚检比例更低。

![1565694442692](D:\Notes\raw_images\1565694442692.png)

![1565694505196](D:\Notes\raw_images\1565694505196.png)

上图实验所展示的效果验证了本文的一大贡献：Repulsion loss，其包括三个子模块：

![1565694591627](D:\Notes\raw_images\1565694591627.png)

- 其中第一个子模块LAttr是使得预测框和匹配上的目标框尽可能接近；
- 第二个子模块LRepGT是使得预测框和周围的目标框尽可能远离；
- 第三个子模块LRepBox是使得预测框和周围的其他预测框尽可能远离。

##### Attraction Term

LAttr采用通用检测框架中的回归loss，可以采用欧式距离、SmoothL1距离以及IoU距离，为了使得和其他算法具有可比性，本文这里采用的是SmoothL1距离：

![1565694642653](D:\Notes\raw_images\1565694642653.png)

##### Repulsion Term (RepGT)

因为LRepGT是==使得预测框P和周围的目标框G尽可能远离==，这里的周围的目标框是除了匹配上的目标框以外的IoU最大的目标框，也即

![1565694758811](D:\Notes\raw_images\1565694758811.png)

，受启发于IoU Loss[2]，它们之间的距离定义为Intersection over Ground-truth (IoG)，也即

![1565694799805](D:\Notes\raw_images\1565694799805.png)


，则RepGT loss定义为：

![1565694831022](D:\Notes\raw_images\1565694831022.png)


从式（4）中可以发现==当预测框P和周围的目标框G的IoG越大，则产生的loss也会越大，因此可以有效防止预测框偏移到周围的目标框上==。此外，式（5）中的sigma是一个调整LRepGT敏感程度的超参数，文中图5给出了验证性实验，这里不再赘述，详情可见论文。

值得注意的是这里的度量距离为什么选用IoG而不是IoU呢？仔细思考一下，如果是IoU的话，那么只要预测框足够大就一定能够使得RepGT loss减小，而这和我们的预期目标是不一致的，这点作者在文中也有论述。

##### Repulsion Term (RepBox)

因为LRepBox是==使得预测框Pi和周围的其他预测框Pj尽可能远离==，Pi和Pj分别匹配上不同的目标框，它们之间的距离采用的是IoU，则RepBox loss定义为：

![1565694922944](D:\Notes\raw_images\1565694922944.png)

从式（6）中可以发现当预测框Pi和周围的其他预测框Pj的IoU越大，则产生的loss也会越大，因此可以有效防止两个预测框因为靠的太近而被NMS过滤掉，进而减少漏检。



#### 实验分析

实验细节：

本文的detector采用的是改进版的Faster R-CNN，以保证对比的公平性，验证实验是在CityPersons验证集上做的。对比实验是在Caltech测试集上做的，训练和测试均采用新标注。

实验结果：

首先我们看下在CityPersons验证集上的剥离实验（如下，表3）：

![1565695097914](D:\Notes\raw_images\1565695097914.png)


可以发现加上RepGT loss和RepBox loss，都可以给baseline带来较为明显的性能提升，尤其是在遮挡情况较为严重的情况下（Heave occlusion）的效果最为显著。本文的两种loss共同将baseline在Reasonable设定下的Miss Rate从14.6减少到了13.2。最后将图像扩大1.5倍得到了最佳的10.9的表现。

![1565695170895](D:\Notes\raw_images\1565695170895.png)

同样在Caltech测试集上的表现也是state-of-the-art（如下，表4和图7）：在Caltech上再一次证明了本文方法对Heave occlusion的有效性。在Reasonable设定下取得了4.0的表现，据笔者所知，在目前已公开发表的实验结果中是最好的了。

![1565695212843](D:\Notes\raw_images\1565695212843.png)

#### 总结展望

本文贡献：

（1）深入研究了行人检测中的遮挡问题（包括物体遮挡和行人之间的遮挡），并分析了影响检测器性能的关键因素。

（2）基于对以上问题的分析，从loss层面为行人检测中的自遮挡问题提出了一种全新的解决方案，在CityPersons和Caltech两个行人检测数据集上展现了非常出色的性能。

个人见解：

（1）本文探讨了行人检测中长期以来广为关注的问题——遮挡，是极具启发性的一篇工作。已有工作中主要采用的是‘分part’，‘加语义信息’等思路加以解决，然而‘分part’毕竟需要人工设定，难以穷尽现实场景中纷繁复杂的遮挡情况，本文另辟蹊径从loss的角度，使得网络自动学习的过程中不断提升定位性能，减少了人为干预，从新的角度发挥了深度学习end-to-end的优势。

（2）值得注意的还有表3，尽管repulsion loss将检测器在Reasonable设定下的Miss Rate从14.6减少到了13.2（下降了1.4个点），==但仅仅将图像扩大1.5倍，Miss Rate又从13.2下降到了10.9（下降了2.3个点）==，我们知道图像扩大是为了检测到更多的小目标，==足以说明弱小目标的存在对检测器的性能影响同样是不容忽视的==。那么，针对弱小目标的检测，能否从loss层面找到一个合理的解决方案呢？期待你的精彩发现。

参考文献：

[1] Citypersons: A diverse dataset for pedestrian detection. CVPR (2016)

[2] Unitbox: An advanced object detection network. ACM MM (2016)