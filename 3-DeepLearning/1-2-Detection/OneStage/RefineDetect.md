### Single-Shot Refinement Neural Network for Object Detection
- 论文地址：https://arxiv.org/abs/1711.06897

- 作者：Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei, Stan Z. Li

- 机构：中科院

- 代码：https://github.com/sfzhang15/RefineDet

- 文章地址：https://mp.weixin.qq.com/s/XcR-ZAkJkM8w2hHyKlCoyA

  


#### 摘要
RefineDet consists of two inter-connected modules, namely, the anchor refinement module and the object detection module.
Specifically, the former aims to (1) filter out negative anchors to reduce search space for the classifier, and
(2) coarsely adjust the locations and sizes of anchors to provide better initialization for the subsequent regressor.

Meanwhile, we design a transfer connection block to transfer the features in the anchor refinement module to predict locations, sizes and class labels of objects in the object detection module.



**1.引言**

物体检测是视觉感知的第一步，也是计算机视觉的一个重要分支。物体检测的目标是用框去标出物体的位置，并给出物体的类别。目前，基于深度学习的物体检测算法大致分为两类：一步法检测器和二步法检测器。

一步法检测器在原图上铺设一系列锚点框，利用一个全卷积网络，对这些锚点框进行一次分类和一次回归，得到检测结果。而二步法检测器在原图上铺设一系列锚点框，先利用一个全卷积网络对这些锚点框进行第一次分类和第一次回归，得到候选区域，抠出每一个候选区域的特征后，再利用一个网络对候选区域进行第二次分类和第二次回归，得到最终的检测结果。

相对于一步法检测器，二步法检测器多了额外的第二步，若在相同的条件下，如输入、锚点框、机器等，一步法一般胜在高效率，而二步法有更高的精度，现在各个检测库上排名靠前的算法，基本都是二步法。代表性的二步法检测器有Faster R-CNN[1]、R-FCN[2]、FPN[3]、Cascade R-CNN[4]，其中Faster R-CNN是奠基性工作，基本所有的二步法检测算法，都是在它的基础上改进的。一步法检测器则有YOLO[5]、SSD[6]、RetinaNet[7]、CornerNet[8]，其中SSD是一步法检测器的集大成者，后续大部分的一步法工作都是基于它的。

二步法相对于一步法，多了后面的进一步分类和回归，这一步一般比较耗时但能显著提升精度，原因是它让二步法相对于一步法有了以下三个优势：

- 二阶段的分类：二步法中的第一步分类时，正负样本是极不平衡的，导致分类器训练困难，这也是一步法效果不如二步法的原因之一。而第二步分类时，第一步会帮第二步滤掉很多简单负样本，使得第二步分类中时正负样本比例比较平衡，即二步法可以很大程度地缓和正负样本极不平衡的问题。
- 二阶段的回归：二步法中的第一步对锚点框进行校正，然后把校正后得到的候选框送给第二步做进一步的校正。
- 二阶段的特征：在二步法中，第一步和第二步法除了共享的特征外，都有自己独有的特征，专注于自身不同难度的任务，如第一步的特征专注于二分类和初步回归，第二步的特征处理多分类和精确回归。

为了能让一步法也具备二步法的这些优势以提升检测精度，同时能够保持原有的检测速度，作者提出了RefineDet这一通用物体检测算法。能够让一步法具备二阶段特征、二阶段分类、二阶段回归，从而在保持一步法速度前提下，获得二步法的精度。



**2.方法**

![img](https://mmbiz.qlogo.cn/mmbiz_jpg/xRp3sibCWzgFC29OnWx22PoZFTwaxvVFeiackmp3SJSibIcoJXh0xv3NFZLqaISFNEkKDDjyMQofxTn6OvfXFjS1g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=2)

​													图2 RefineDet算法的网络架构示意图

上图是RefineDet的检测框架示意图，由ARM(Anchor Refinement Module)和ODM(Object Detection Module) 模块组成，它俩由TCB(Transfer Connection Block)连接。ARM专注于二分类，为后续ODM滤掉大量的简单负样本，同时进行初级的边框校正，为后续ODM提供更好的回归起点。ODM把ARM优化过的锚点框作为输入，专注于多分类和精准的边框校正。其中ODM没有使用耗时的逐区域操作，而是直接通过TCB转换ARM特征、融合高层特征，以得到感受野丰富、细节充足、内容抽象的特征，用于进一步的分类和回归，使得一步法检测器RefineDet具备了二步法的二阶段分类、二阶段回归、二阶段特征这3个优势。



**3.实验**

![img](https://mmbiz.qlogo.cn/mmbiz_jpg/xRp3sibCWzgFC29OnWx22PoZFTwaxvVFeFJBIm8ymKVm7ktferwMicktYvaCHUicIZE7FwV7sElXiaEnp92VbMT06g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=2)

​																	表1 检测精度 vs. 检测速度

表1总结了著名检测算法的速度以及精度。RefineDet在保持一步法速度的同时，能够达到甚至超过二步法的精度。跟SSD相比，RefineDet算法框架多了下面一部分卷积层和反卷积层，速度还比SSD快一些原因是：

1. 使用较少的anchor，如512尺度下，RefineDet共有1.6W个框，而SSD有2.5W个框。RefineDet使用较少anchor也能达到高精度的原因是二阶段回归，虽然总共预设了4个尺度（32，64，128，256）和3个比例（0.5，1，2），但是经过第一阶段的回归后，预设的anchor被极大丰富，因此用于第二阶段回归的anchor，具备丰富的尺度和比例。
2. 使用较少的检测层：RefineDet在基础网络上新加了很少的卷积层，并只选了4个卷积层作为检测层。

![img](https://mmbiz.qlogo.cn/mmbiz_jpg/xRp3sibCWzgFC29OnWx22PoZFTwaxvVFemUPBxVzApiab1TPLVRSpUdsO3NWq92rRIhhXaibv0OyvjNhDxcic0VB0A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=2)

​													表2 RefineDet在各个数据库上的检测精度

表2是RefineDet算法在VOC2007，VOC2012，COCO上详细的结果。由于显存的限制，只训了输入为320和512的两个模型。从这些结果中，有以下几点结论：

1. 输入尺度与精度高度正相关：训练和测试的输入越大，小物体越少，精度就会越高。
2. BN的问题：显存的限制会导致每张GPU显卡上图片数量不足，使得BN信息统计不稳定，从而影响检测精度，可以使用同步BN或GN来解决，以训更大尺度的模型。
3. 多尺度测试很重要：由于RefineDet最大输入只有512x512，而二步法检测器有着较大的输入如1000x600或800x800。为了公平比较而使用了多尺度测试方法，以降低尺度不同的影响。



**4.扩展**

提出RefineDet算法后，作者在更加贴近实际应用的人脸检测任务上对其进行了扩展验证。目前人脸检测领域难度最大数据集是WIDER FACE，它总共有32203张图像，393703个人脸标注，包含尺度，姿态，遮挡，表情，化妆，光照等难点。WIDER FACE每张图像的人脸数据偏多，平均每张图12.2人脸，密集小人脸非常多，同时根据EdgeBox的检测率情况划分为三个难度等级：Easy, Medium, Hard。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/xRp3sibCWzgFC29OnWx22PoZFTwaxvVFeSsTMU7ibZyXBoejDgMeCGebTBtiaXJN6QzWGmS4kZ7JZIPb6Ww6PWKyw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​																		图3 人脸检测算法SRN

RefineDet在处理人脸检测这一特殊任务时，并不是在所有的检测层上做二阶段分类和二阶段回归都有提升。二阶段分类用于较浅的3个检测层上有效果，原因是98.5%的锚点框关联于较浅的3个层，这些层存在正负样本不平衡的问题，二阶段分类得以发挥作用。此外，二阶段回归用于较深的3个检测层上有效果，原因是强行让较浅层利用二阶段回归去得到更准的小人脸位置，会影响更重要的分类任务，而较深的3个层则不存在这种问题，适合做二阶段回归来提升大中尺度的人脸位置。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/xRp3sibCWzgFC29OnWx22PoZFTwaxvVFeEiblLsnoZ1tj51qz0C46TLL5EwibQ88SicAUB9Knv1KziboAQbx4sRvukA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​																		图4 人脸检测算法ISRN

此外，目前的检测器都需要加载预训练模型，这大大限制了网络的灵活性，使得不能够对网络进行任何微小的改动。作者成功地验证了不需要预训练模型也能够成功地训出检测器后，对网络进行了改进以提升小物体的检测性能。常用的ResNet网络对小物体不大友好，因为它第一个卷积层就有一个下采样操作，紧接着的第二个池化层也有一个下采样操作，这会导致小物体的信息基本丢失。为了解决这一问题，作者对ResNet网络进行了相应的改进，以对小物体更友好，并基于RefineDet成功地从零训了一个人脸检测器，在WIDER FACE数据集上达到了较高的检测精度。

