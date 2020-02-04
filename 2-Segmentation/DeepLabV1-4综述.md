DeepLab：Semantic image segmentation with deep convolutional nets and fully connected CRFs

论文地址: https://arxiv.org/pdf/1606.00915.pdf

作者：Liang-Chieh Chen, George Papandreou, Senior Member, IEEE, Iasonas Kokkinos, Member, IEEE

机构： Google



**三点主要贡献：**

1. 针对信号下采样或池化降低分辨率，DeepLab是采用的atrous(带孔)算法扩展感受野，获取更多的上下文信息。
2. 论文使用Multi-Scale features。其实就是U-Net和FPN的思想，在输入图像和前四个最大池化层的输出上附加了两层的MLP(第一层是128个3×3卷积，第二层是128个1×1卷积)，最终输出的特征与主干网的最后一层特征图融合，特征图增加5×128=640个通道，实验表示多尺度有助于提升预测结果，但是效果不如CRF明显。
3. DeepLab结合了深度卷积神经网络(DCNNs)和概率图模型(DenseCRFs)的方法。在实验中发现DCNNs做语义分割时精准度不够的问题，根本原因是DCNNs的高级特征的平移不变性(由于重复的池化和下采样导致)。分类器获取以对象为中心的决策是需要空间变换的不变性，这天然的限制了DCNN的定位（Location）精度，DeepLab采用==完全连接的条件随机场(CRF)==提高模型捕获细节的能力。

**实验：**论文的模型基于VGG16，在Titan GPU上运行速度达到了8FPS，全连接CRF平均推断需要0.5s ，PASCAL VOC-2012 达到79.7% mIOU，当时的 state-of-art。



**结构图:**

![1562740663571](C:\Users\j00496872\Desktop\Notes\raw_images\1562740663571.png)

Fig. 1: Model Illustration. 主干网络采用VGG-16或者ResNet-101。使用空洞卷积，图像Stride由原来的32倍，减到8倍，然后采用二插值法，放大8倍到原图大小，之后采用全连接条件随机场改善结果。

**实验：**

![1562741281694](C:\Users\j00496872\Desktop\Notes\raw_images\1562741281694.png)

#### DeepLabV2

![1562808221506](C:\Users\j00496872\Desktop\Notes\raw_images\1562808221506.png)

输入图片中的实例对象存在多种尺度，固定的网络设置使得网络的感受野固定，在对很小的实例对象和很大的实例对象表现出的效果不好。所以这个版本将网络空洞卷积最后一部分改为多通道（空洞率不同）并行即aspp，从而获得了多个尺度的感受野，增强了对multi scale的适应性。同时本文尝试用了更深的网络结构 resnet。


#### DeepLabV3

![1562808321308](C:\Users\j00496872\Desktop\Notes\raw_images\1562808321308.png)

在原来resnet基础上再添加 5,6,7三个block， 都为block4的副本。 个人认为这里在尝试更深的网络

![1562808347937](C:\Users\j00496872\Desktop\Notes\raw_images\1562808347937.png)

这里是在改进aspp，当空洞率较大时原来3x3卷积的9个参数只剩下中间参数有效，其他参数作用于前面feature层的padding部分，为无效参数，所以在rate变大时，3x3卷积退化成1x1卷积，所以这里的aspp去掉了rate=24的分支，增加了1x1卷积分支。另外，为了获得全局信息，加入了image pooling分支，其实这个分支做的操作就是将block4输出的feature进行全局池化，然后再双线性插值到与其他分支一样的分辨率。最后将五个分支连接，再做1x1卷积（通道数变化）。v3不再使用条件随机场校正。 



#### DeepLabV3+

语义分割关注的问题: 

1、 实例对象多尺度问题。

2、 因为深度网络存在stride=2的层，会导致feature分辨率下降，从而导致预测精度降低，而造成的边界信息丢失问题。

deeplab v3+主要目的在于解决问题2, 可以使用空洞卷积替代更多的pooling层来获取分辨率更高的feature。但是feature分辨率更高会极大增加运算量。 以deeplab v3 使用的resnet101为例， stride=16将造成后面9层feature变大，后面9层的计算量变为原来的2*2=4倍大。stride=8则更为恐怖，后面7-8层的计算量都会变大很多。

解决方案：1、编解码器结构。2 Modified Aligned Xception。

![1562808541846](C:\Users\j00496872\Desktop\Notes\raw_images\1562808541846.png)

在deeplab v3 基础上 加入解码器。 A是aspp结构， A的8倍的上采样可以看做是一个naïve的解码器。 B是编解码结构，集合了高层和底层的feature。 C就是本文采取的结构，Conv2（图中红色）的提取到结果和最后提取出的feature上采样4后融合。


![1562808587604](C:\Users\j00496872\Desktop\Notes\raw_images\1562808587604.png)

解码器部分：先从低层级选一个feature，将低层级的feature用1x1的卷积进行通道压缩（原本为256通道，或者512通道），目的在于减少低层级的比重。作者认为编码器得到的feature具有更丰富的信息，所以编码器的feature应该有更高的比重。 这样做有利于训练。

再将编码器的输出上采样，使其分辨率与低层级feature一致。 举个例子，如果采用resnet conv2 输出的feature，则这里要*4上采样。 将两种feature 连接后，再进行一次3x3的卷积（细化作用），然后再次上采样就得到了像素级的预测。 后面的实验结果表明这种结构在stride=16时既有很高的精度速度又很快。stride=8相对来说只获得了一点点精度的提升，但增加了很多的计算量。

#### Modified Aligned Xception

Xception 主要采用了deepwish seperable convolution来替换原来的卷积层。简单的说就是这种结构能在更少参数更少计算量的情况下学到同样的信息。这边则是考虑将原来的resnet-101骨架网换成Xception。

![1562808927738](C:\Users\j00496872\Desktop\Notes\raw_images\1562808927738.png)

红色部分为修改（1）更多层：重复8次改为16次（基于MSRA目标检测的工作）。（2）将原来简单的pool层改成了stride为2的deepwish seperable convolution。 （3）额外的RELU层和归一化操作添加在每个 3 *×* 3 depthwise convolution之后（原来只在1x1卷积之后）

 链接：https://zhuanlan.zhihu.com/p/34929725 