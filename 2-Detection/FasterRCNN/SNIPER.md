### SNIPER: Efficient Multi-Scale Training

- 论文地址：https://arxiv.org/abs/1711.08189
- 作者：Bharat Singh, Larry S. Davis
- 机构：University of Maryland, College Park (马里兰大学.美国)
- 代码地址：https://github.com/mahyarnajibi/SNIPER
- CVPR18 Oral Paper
- 知乎解析：https://zhuanlan.zhihu.com/p/36431183

在CNN中，我们需要不同种类的invariance（不变性）来做识别，这其中translation invariance在CNN中可以比较好地被考虑，然而另外一种重要的不变性，scale invariance就很难被CNN考虑到。为了解决这个问题，一般常用的两大种策略就是Image Pyramid或者Feature Pyramid。

#### 摘要
By evaluating the performance of different network architectures for classifying small objects on ImageNet, we show that ==CNNs are not robust to changes in scale.== Based on this analysis, we propose to train and test detectors on the same scales of an image pyramid.

Since small and large objects are difficult to recognize at smaller and larger scales respectively, we present a novel training scheme called ==Scale Normalization for Image Pyramids (SNIP)== which selectively back-propagates the gradients of object instances of different sizes as a function of the image scale.

On the COCO dataset, our single model performance is 45.7% and an ensemble of 3 networks obtains an mAP of 48.3%.



#### 介绍
Interestingly, the median scale of object instances relative to the image in ImageNet (classification) vs COCO (detection) are 0.554 and 0.106 respectively. Therefore, ==most object instances in COCO are smaller than 1%== of image area! To make matters worse, the scale of the smallest and largest 10% of object instances in COCO is 0.024 and 0.472 respectively (resulting in ==scale variations of almost 20 times!==); Moreover, differences in the scale of object instances between classification and detection datasets also results in a ==large domain-shift while finetuning from a pre-trained classification== network.

- 问题1：mscoco中的图片Instance的大小都很小。
- 问题2：Instance的大小差异很大，最大和最小相差20倍。
- 问题3：从ImageNet中与训练的网络可能没有那么适应。

To alleviate the problems arising from scale variation and small object instances, multiple solutions have been proposed. For example,

- features from the layers near to the input, referred to as shallow(er) layers, are combined with deeper layers for detecting small object instances [23, 34, 1, 13, 27],
- dilated/deformable convolution is used to increase receptive fields for detecting large objects [32, 7, 37, 8],
- independent predictions at layers of different resolutions are used to capture object instances of different scales [36, 3, 22],
- context is employed for disambiguation[38, 39, 10],
- training is performed over a range of scales[7, 8, 15] or,
- inference is performed on multiple scales of an image pyramid and predictions are combined using nonmaximum suppression [7, 8, 2, 33].

#### Image Classification at Multiple Scales

In this section we study the effect of domain shift, which is introduced when different resolutions of images are provided as input during training and testing.

我们检测的常用操作：We perform this analysis because state-of-the-art detectors are typically trained at a resolution of 800x1200 pixels, but inference is performed on an image pyramid, including higher resolutions like 1400x2000 for detecting small objects.

Multi-Scale Inference:
We observe that as the difference in resolution between training and testing images increases, so does the drop in performance. Hence, testing on resolutions on which the network was not trained is clearly sub-optimal, at least for image classification.

作者用不同Resolution的图片在一个224*224的分类网络上进行测试，发现和训练图片Resolution相近的图片准确率越高，而低Resolution的图片准确率越低。 （弱关联）

Resolution Specific Classifiers：

This result empirically demonstrates that the filters learned on high-resolution images can be useful for recognizing low-resolution images as well. Therefore, instead of reducing the stride by 2, it is better to up-sample images 2 times and then fine-tune the network pre-trained on highresolution images.

Since pre-training on ImageNet (or other larger classification datasets) is beneficial and filters learned on larger object instances help to classify smaller object instances, upsampling images and using the network pre-trained on high resolution images should be better than a specialized network for classifying small objects.![1565851841633](C:\Users\j00496872\Desktop\Notes\raw_images\1565851841633.png)

Figure 6. SNIP training and inference is shown. Invalid RoIs which fall outside the specified range at each scale are shown in purple. These are discarded during training and inference. Each batch during training consists of images sampled from a particular scale. Invalid GT boxes are used to invalidate anchors in RPN. Detections from each scale are rescaled and combined using NMS.

作者用实验证明了我们常用操作的正确性，与其去训练一个基于低像素的分类器与训练模型，不如使用一个高分辨率的分类器，然后采用放大的图片在上面做FineTuning.

值得关注的是以下三组实验反映出的问题：

- 800(all)和1400(all)的对比：训练时使用不同大小的图训练，理论上如果使用更大图，小物体检测的性能应当有显著提升。但是实验表明这个提升非常小。文章中给出的解释是虽然1400的图训练会提升小物体的性能，但是会加大大物体训练的困难，所以此消彼长，并不会有比较大提升。（其实我是不太认可这个解释的，个人理解见下面的第三条。）

- 1400(<80px)和1400(all)的对比：既然大物体太难train了，可能对小物体造成干扰，是否去掉大物体可以提升小物体的性能呢？答案也是否定的，而且损失掉的大物体的语义会让结果变得更加糟糕。

- MST：在Object Detection中，为了提升测试针对不同scale物体的性能，大家一般会使用Multi-scale training/testing这样的测试时融合的技巧来提升结果。与SNIP做法最大的区别就在于Multi-scale的做法扩充了不同scale样本的数目，但是仍然要求CNN去fit所有scale的物体。==通过这样的一个对比实验，SNIP非常solid地证明了就算是数据相对充足的情况下，CNN仍然很难使用所有scale的物体。==个人猜测由于CNN中没有对于scale invariant的结构，CNN能检测不同scale的“假象”，更多是通过CNN来通过capacity来强行memorize不同scale的物体来达到的，这其实浪费了大量的capacity，而SNIP这样只学习同样的scale可以保障有限的capacity用于学习语义信息。

  ![1565851490859](C:\Users\j00496872\Desktop\Notes\raw_images\1565851490859.png)

所以，其实SNIP做的事情是非常简单的：==在训练中，每次只回传那些大小在一个预先指定范围内的proposal的gradient，而忽略掉过大或者过小的proposal；== ==在测试中，建立大小不同的Image Pyramid，在每张图上都运行这样一个detector，同样只保留那些大小在指定范围之内的输出结果，最终在一起NMS。这样就可以保证网络总是在同样scale的物体上训练，也就是标题中Scale Normalized的意思。==

![1565851788499](C:\Users\j00496872\Desktop\Notes\raw_images\1565851788499.png)

实验结果中可以看到，对比各种不同baseline，在COCO数据集上有稳定的3个点提升，这个结果可以说是非常显著了。

简而言之，==SNIP可以看做是一个改版版本的Image Pyramid==，从本质上来讲，其实和MTCNN并无太大区别，然而训练的时候使用了全卷积网络来训练和测试，一方面可以加速，另一方面可以利用更好更大的context信息。但是其背后反映出的问题确是==十分有insight：也就是直击现在CNN网络其实仍无法解决好的一个问题，也就是scale invariance==。虽然大家通过FPN，Multi-scale Training这样的手段来减轻了这个问题，但这个事情还是不能通过现有CNN的框架来本质上描述，无论是使用Multi-scale Training来暴力扩充数据，还是使用FPN来融合底层和高层的不同分辨率特征，都很难得到一个满意的效果。最有效的方式还是这个网络在训练和测试的时候都处理scale差不多大的物体。其实沿着SNIP的思路应该后续可以有很多工作可以来做，例如如何能够加速这样的Image Pyramid，是否可以显式地拆分出CNN中对于物体表示的不同因素（semantic，scale等等）。这些都会是很有价值的研究题目。



#### 实验结果

![1565851911674](C:\Users\j00496872\Desktop\Notes\raw_images\1565851911674.png)