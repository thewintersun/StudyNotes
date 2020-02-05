## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

论文地址：https://arxiv.org/abs/1703.10593

作者：Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

机构：Berkeley AI Research (BAIR) laboratory, UC Berkeley

发表：ICCV2017

项目地址：https://junyanz.github.io/CycleGAN/

代码地址：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

理解文章：https://hardikbansal.github.io/CycleGANBlog/



### 摘要

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. ==We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples.==  自动将某一类图片转换成另外一类图片。

Our goal is to learn a mapping $G: X→Y$  such that the distribution of images from $G(X)$  is indistinguishable from the distribution $Y$ using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping $F: Y→X$  and introduce a cycle consistency loss to push $F(G(X))≈X$ (and vice versa). 

Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer （风格迁移）, object transfiguration（目标变换）, season transfer（季节变换）, photo enhancement（照片增强）, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

### 介绍

![1579159993263](D:\Notes\raw_images\1579159993263.png)

图1: 给定任意两个无序的图像集合X和Y，我们的算法学习自动“翻译”一个图像从一个到另一个，反之亦然: (左)来自Flickr的莫奈画作和风景照片; (中)来自ImageNet的斑马和马; (右)Flickr上的夏天和冬天的优美照片。(下)示例应用程序: 使用名家的绘画作品集合，我们的方法学会了把自然的照片渲染成各自的风格。

这篇文章介绍了一种方法：捕获一组图片的特殊特征，并转换到另外一组图片上，而不需要对齐训练数据集。

这个问题可以扩散地描述成：图像翻译。将一个场景的图片转换成目标场景，灰度图转为彩色， 图像to语义标签， 边缘图to照片等。

直接采用生成对抗的方式做两个场景的映射，会出现collapse mode的问题。所以作者添加新的结构到目标中：

> These issues call for adding more structure to our objective. Therefore, we exploit the property that translation should be “cycle consistent” （闭环的）, in the sense that if we translate, e.g., a sentence from English to French, and then translate it back from French to English, we should arrive back at the original sentence [3]. Mathematically, if we have a translator G : X -> Y and another translator F : Y -> X, then G and F should be inverses of each other, and both mappings should be bijections （双射）.

### 相关工作

**GAN：** Generative Adversarial Networks (GANs) [16, 63] have achieved impressive results in ==image generation==（图像生成） [6,39], ==image editing== （图像编辑）[66], and ==representation learning==（表征学习） [39, 43, 37]. Recent methods adopt the same idea for ==conditional image generation==（条件图像生成） applications, such as ==text2image==（文字转图片） [41], ==image inpainting==（图像修复） [38], and ==future prediction==（未来预测） [36], as well as to other domains like videos [54] and 3D data [57]. 

**Image-to-Image Translation：**Our approach builds on the “pix2pix” framework of Isola et al. [22], which uses a ==conditional generative adversarial network== [16] to learn a mapping from input to output images. Similar ideas have been applied to various tasks such as ==generating photographs from sketches== [44] or
==from attribute and semantic layouts== [25]. However, unlike the above prior work, we learn the mapping without paired training examples.

### CycleGAN的原理

我们之前已经说过，CycleGAN的原理可以概述为：**将一类图片转换成另一类图片**。也就是说，现在有两个样本空间，X和Y，我们希望把X空间中的样本转换成Y空间中的样本。

因此，实际的目标就是学习从X到Y的映射。我们设这个映射为F。它就对应着GAN中的**生成器**，F 可以将X中的图片x转换为Y中的图片F(x)。对于生成的图片，我们还需要GAN中的**判别器**来判别它是否为真实图片，由此构成对抗生成网络。设这个判别器为 $D_Y$ 。这样的话，根据这里的**生成器**和**判别器**，我们就可以构造一个GAN损失，表达式为：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B%5Ctext%7BGAN%7D%7D%28F%2CD_%7BY%7D%2CX%2CY%29+%3D+E_%7By+%5Csim+p_%7B%5Ctext%7Bdata%7D%7D%28y%29%7D%5Clbrack%5Clog+D_%7BY%7D%28y%29%5Crbrack+%2B+E_%7Bx+%5Csim+p_%7B%5Ctext%7Bdata%7D%7D%28x%29%7D%5Clbrack%5Clog%281+-+D_%7BY%7D%28F%28x%29%29%29%5Crbrack)

这个损失实际上和原始的GAN损失是一模一样的。但单纯的使用这一个损失是无法进行训练的。原因在于，==映射F完全可以将所有x都映射为Y空间中的同一张图片，使损失无效化。== 对此，作者又提出了所谓的“循环一致性损失”（Cycle Consistency Loss）。

我们再假设一个映射G，它可以将Y空间中的图片y转换为X中的图片G(y)。CycleGAN同时学习F和G两个映射，并要求 ![[公式]](https://www.zhihu.com/equation?tex=F%28G%28y%29%29+%5Capprox+y) ，以及 ![[公式]](https://www.zhihu.com/equation?tex=G%28F%28x%29%29+%5Capprox+x) 。**也就是说，将X的图片转换到Y空间后，应该还可以转换回来。这样就杜绝模型把所有X的图片都转换为Y空间中的同一张图片了。**

根据 ![[公式]](https://www.zhihu.com/equation?tex=F%28G%28y%29%29+%5Capprox+y) 和 ![[公式]](https://www.zhihu.com/equation?tex=G%28F%28x%29%29+%5Capprox+x) ，循环一致性损失就定义为：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B%5Ctext%7Bcyc%7D%7D%28F%2CG%2CX%2CY%29+%3D+E_%7Bx+%5Csim+p_%7B%5Ctext%7Bdata%7D%7D%28x%29%7D%5Clbrack%7C%7CG%28F%28x%29%29+-+x%7C%7C_%7B1%7D%5Crbrack+%2B+E_%7By+%5Csim+p_%7B%5Ctext%7Bdata%7D%7D%28y%29%7D%5Clbrack%7C%7CF%28G%28y%29%29+-+y%7C%7C_%7B1%7D%5Crbrack)

同时，我们为G也引入一个判别器 ![[公式]](https://www.zhihu.com/equation?tex=D_%7BX%7D) ，由此可以同样定义一个GAN的损失 ![[公式]](https://www.zhihu.com/equation?tex=L_%7B%5Ctext%7BGAN%7D%7D%28G%2CD_%7BX%7D%2CX%2CY%29) ，最终的损失就由三部分组成：

![[公式]](https://www.zhihu.com/equation?tex=L+%3D+L_%7B%5Ctext%7BGAN%7D%7D%28F%2CD_%7BY%7D%2CX%2CY%29+%2B+L_%7B%5Ctext%7BGAN%7D%7D%28G%2CD_%7BX%7D%2CX%2CY%29+%2B+%5Clambda+L_%7B%5Ctext%7Bcyc%7D%7D%28F%2CG%2CX%2CY%29)



![1579167670159](D:\Notes\raw_images\1579167670159.png)

Figure 3: (a) Our model contains two mapping functions $G : X \to  Y$ and $F : Y \to X$ , and associated adversarial discriminators $D_Y$ and $D_X$.  $D_Y$ encourages G to translate X into outputs indistinguishable from domain Y , and vice versa for $D_X$ and F. To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started: (b) forward cycle-consistency loss: $ x \to G(x) \to F(G(x)) \approx x$, and (c) backward cycle-consistency loss: $y \to F(y) \to G(F(y)) \approx y$

![1579163465780](D:\Notes\raw_images\1579163465780.png)

**CycleGAN与DCGAN的对比**

为了进一步搞清楚CycleGAN的原理，我们可以拿它和其他几个GAN模型，如DCGAN、pix2pix模型进行对比。

先来看下DCGAN，它的整体框架和最原始的那篇GAN是一模一样的，在这个框架下，输入是一个噪声z，输出是一张图片（如下图），**因此，我们实际只能随机生成图片，没有办法控制输出图片的样子**，更不用说像CycleGAN一样做图片变换了。

![img](https://pic1.zhimg.com/80/v2-b9a95d2bc4ff516dcdc665fa609c9c0c_hd.png)

**CycleGAN与pix2pix模型的对比**

pix2pix也可以做图像变换，它和CycleGAN的区别在于，pix2pix模型必须要求**成对数据**（paired data），而CycleGAN利用**非成对数据** 也能进行训练(unpaired data)。

比如，我们希望训练一个将白天的照片转换为夜晚的模型。如果使用pix2pix模型，**那么我们必须在搜集大量地点在白天和夜晚的两张对应图片，而使用CycleGAN只需同时搜集白天的图片和夜晚的图片，不必满足对应关系。**因此CycleGAN的用途要比pix2pix更广泛，利用CycleGAN就可以做出更多有趣的应用。

### CycleGAN的实现

**网络结构**

We adopt the architecture for our generative networks from Johnson et al. [23] who have shown impressive results for neural style transfer and super-resolution.

> J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In
> ECCV, 2016.

采用Instance Normalization。

For the discriminator networks we use 70x70 PatchGANs [22, 30, 29], which aim to classify whether 70 x 70 overlapping image patches are real or fake.

> P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017.

**训练细节**

First, for LGAN (Equation 1), we replace the negative log likelihood objective by a least-squares loss. 

For a GAN loss  $L_{GAN}(G, D, X, Y )$ , we train the G to minimize：

 $E_{x \sim p_{data}(x)}[(D(G(x)) - 1)^2]$
and train the $D$ to minimize：

 $E_{y \sim p_{data}(y)}[(D(y) - 1)^2]+E_{x \sim p_{data}(x)}[D(G(x)) ^2]$ . 

Second, to reduce model oscillation（振荡） [15], we follow Shrivastava et al.’s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators.

### 实验结果

在Cityscapes 上的 $label \leftrightarrow photos$ 任务：

![1579233066584](D:\Notes\raw_images\1579233066584.png)

在GoogleMap 上的 $photos \leftrightarrow maps$ 任务：

![1579233533079](D:\Notes\raw_images\1579233533079.png)

![1579233834533](D:\Notes\raw_images\1579233834533.png)

**消融学习：**

![1579233924582](D:\Notes\raw_images\1579233924582.png)

![1579234136414](D:\Notes\raw_images\1579234136414.png)

Figure 7: Different variants of our method for mapping $labels \leftrightarrow photos$ trained on cityscapes. From left to right: input, cycle-consistency loss alone, adversarial loss alone, GAN + forward cycle-consistency loss $(F(G(x)) \approx x)$, GAN + backward cycle-consistency loss $(G(F(y)) \approx y)$ , CycleGAN (our full method), and ground truth. 

Both ==Cycle alone and GAN +backward fail to produce images similar to the target domain==. ==GAN alone and GAN + forward suffer from mode collapse==, producing identical label maps regardless of the input photo.

### CycleGAN的应用

集体风格迁移， 对象转换，季节转换，Painting转换为Photo, Photo 增强（比如背景虚化等）。 

### 失败的案例

![1579243415862](D:\Notes\raw_images\1579243415862.png)

![1579243439843](D:\Notes\raw_images\1579243439843.png)

Figure 17: Typical failure cases of our method. Left: in the task of $dog \to cat transfiguration$, CycleGAN can only make minimal changes to the input. Right: CycleGAN also fails in this $horse \to zebra$ example as our model has not seen images of horseback riding during training.  

> On translation tasks that involve color and texture changes, like many of those reported above, the method often succeeds. We have also explored ==tasks that require geometric changes, with little success==. This failure might be caused by our generator architectures which are tailored for good performance on the appearance changes.
>
> Some failure cases are caused by the distribution characteristics of the training datasets.
>
> We also observe a lingering gap between the results achievable with paired training data and those achieved by our unpaired method.

作者发现的三个问题：

1. CycleGAN在纹理和颜色的转换上做得很好，但在几何变换上效果不佳。
2. 因为数据不均衡问题导致一些转换失败，比如马转换为斑马，因为数据中不常出现人，而导致人也被一起转换。
3. 在标签转换的效果上，还是远差于paired training data 的方法，比如PixeltoPixel。