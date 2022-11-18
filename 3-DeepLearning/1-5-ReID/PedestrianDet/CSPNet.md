### Center and Scale Prediction: A Box-free Approach for Object Detection

High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection, CVPR, 2019

论文地址：https://arxiv.org/pdf/1904.02948v2.pdf

作者：Wei Liu · Shengcai Liao · Irtiza Hasan

机构：国防科大

发表: CVPR2019

代码地址：https://github.com/liuwei16/CSP



#### 摘要

在本文中，检测被明确为高级语义特征检测任务。本文提出的检测器==扫描整个图像上的特征点==。在本文中，目标==检测通过卷积简化为简单的中心和尺度预测任务==。这样，所提出的方法具有 box-free 设置。虽然结构简单，但它在几个具有挑战性的基准数据集测试中提供了有竞争力的准确性，包括行人检测和人脸检测。此外，进行交叉数据集评估，证明了所提出方法的优异的泛化能力。



#### 方法

Center and Scale Prediction (CSP) based detector

结构图，将中心点和大小预测分为两个分支，分别进行预测。

![1565838493899](D:\Notes\raw_images\1565838493899.png)

![1565839257893](D:\Notes\raw_images\1565839257893.png)

Fig. 2 Overall architecture of CSP, which mainly comprises two components, i.e. the feature extraction
module and the detection head. The feature extraction module concatenates feature maps of different resolutions into a single one. The detection head merely contains a 3x3 convolutional layer, followed by two
prediction layers, one for the center location and the other for the corresponding scale.

- 主干网络采用ResNet-50或者MobileNetV2 , pretrained on ImageNet.
- 卷基层被分为5层， the output feature maps are downsampled by 2, 4, 8, 16, 32 w.r.t. the input image. 
- As a common practice, the dilated convolutions are adopted in stage 5 to ==keep its output as 1/16== of the input image size.
- Since the feature maps from each stage have different scales, we ==use L2-normalization== to rescale their norms to 10, which is similar to [21]. 
- Similarly to [43], ==stride = 4 gives the best performance== as demonstrated in our experiments, because a larger r means coarser feature maps which struggle on accurate localization, while a smaller r brings more computational burdens.

![1565840527476](D:\Notes\raw_images\1565840527476.png)

![1565840987249](D:\Notes\raw_images\1565840987249.png)

#### Loss结构

![1565841453168](D:\Notes\raw_images\1565841453168.png)

![1565841384621](D:\Notes\raw_images\1565841487883.png)

![1565841562783](D:\Notes\raw_images\1565841562783.png)

In order to reduce the ambiguity of these negatives surrounding the positives, we also ==apply a 2D Gaussian mask==G(:) centered at the location of each positive, which is similar in [18,43].

If these masks have overlaps, we choose ==the maximum values for the overlapped locations==. 

To combat the extreme positive-negative imbalance problem, the ==focal weights [24] on hard examples== are also adopted. 

> Law, H., Deng, J.: Cornernet: Detecting objects as paired keypoints. In: The European Conference on
> Computer Vision (ECCV) (2018)
>
> Song, T., Sun, L., Xie, D., Sun, H., Pu, S.: Small-scale pedestrian detection based on topological line
> localization and temporal feature aggregation. In: The European Conference on Computer Vision
> (ECCV) (2018)
>
> Lin, Y.T., Goyal, P., Girshick, R., He, K., Doll´ar, P.: Focal loss for dense object detection. arXiv
> preprint arXiv:1708.02002 (2017)

![1565841749735](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1565841749735.png)

If the offset prediction branch is appended, the similar smooth L1 loss is adopted.

![1565841326867](D:\Notes\raw_images\1565841326867.png)

weights for center classification, scale regression and offset regression losses, which are experimentally set as ==0.01, 1 and 0.1,== respectively.



#### 实验结果

CityPerson 1024*2018的测试结果

![1565840601666](D:\Notes\raw_images\1565840601666.png)

![1565840728085](D:\Notes\raw_images\1565840728085.png)