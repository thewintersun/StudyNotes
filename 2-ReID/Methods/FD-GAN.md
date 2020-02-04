### FD-GAN: Pose-guided Feature Distilling GAN for Robust Person Re-identification

论文地址：https://arxiv.org/abs/1810.02936v2

作者：Yixiao Ge, Zhuowan Li, Haiyu Zhao, Guojun Yin, Shuai Yi, Xiaogang Wang, Hongsheng Li

机构：商汤，CUHK

代码地址： https://github.com/yxgeee/FD-GAN

原文链接：https://blog.csdn.net/melody96313/article/details/83757072



#### 摘要

a Feature Distilling Generative Adversarial Network (FD-GAN) is proposed for learning identity-related and pose-unrelated representations. It is a novel framework ==based on a Siamese structure== with ==multiple novel discriminators== on human poses and identities. In addition to the discriminators, a novel ==same-pose loss== is also integrated, which requires appearance of a same person's generated images to be similar. 

After learning pose-unrelated person features with pose guidance, ==no auxiliary pose information and additional computational cost is required during testing==. Our proposed FD-GAN achieves state-of-the-art performance on three person reID datasets, which demonstrates that the effectiveness and robust feature distilling capability of the proposed FD-GAN.

这篇文章的大致思想是：用GAN来对行人的特征进行蒸馏，只保留和身份信息有关的特征，而去除了人体姿态这些冗余的特征信息。而在推断的时候，只需用到encoder提取的鲁棒特征就可以了，没有增加额外的计算量。

即：为了应对多种多样的姿态变化  --> 学习到 ==和身份信息有关，而和姿态信息无关的== 特征表达。

#### 介绍

![1567497961036](C:\Users\j00496872\Desktop\Notes\raw_images\1567497961036.png)

Figure 2: The Siamese structure of the proposed FD-GAN.  孪生网络结构

- Robust identity-related and pose-unrelated features are learned by the image encoder E with a verification loss。 编码器E用来学习身份相关的，而姿态无关的特征，使用一个Verification Loss。
- and the auxiliary task of generating fake images to fool identity and pose discriminators. 每个分支都会有一个生成器和两个判别器分别对姿态和身份进行判别。
- A novel same-pose loss term is introduced to further encourage learning identity-related and pose-unrelated visual features.



**数据库：**Market-1501, CUHK03 ,  DukeMTMC-reID

**网络输入：** 每个分支都是行人图片和目标姿势图，要注意两个分支共享一个姿势图，这样才有后面的 same-pose loss 

**框架：**==孪生网络的结构==也就是两个分支，每个分支都会有一个encoder,一个generator和两个判别器：分别对姿态和身份进行判别。两个分支共享一个身份确认分类器。

With the adversarial losses, identity-irrelevant information, such as pose and background appearance, in the input image is mitigated from the visual features by the image encoder.

![1567500624241](C:\Users\j00496872\Desktop\Notes\raw_images\1567500624241.png)

Figure 3: Network structures of (a) the generator G and the image encoder E, (b) the verification
classifier V , (c) the identity discriminator $D_{id}$, (d) the pose discriminator $D_{pd}$.

##### a. 图像编码器和图像生成器

1. 行人图片编码器：是基于ResNet-50来训练的，输出的向量是2048维的。
2. 目标姿势图：是一个18通道的map，每个通道对应一个关键点的位置，并以高斯分布的热力图所表示的。它的编码器是一个五层的神经网络（ Convolution-BN-ReLU ），得到一个128维的向量。
3. 噪声：从标准高斯分布中采样另外的256维噪声矢量。

视觉特征，目标姿势特征和噪声特征concatenated并输入五层（ Convolution-BN-dropout-ReLU ）上采样块，得到生成的人物图像.

**b. 身份识别分类器**

将两个分支的行人编码器输出的特征表达进行对比来判断是否为同一个人，具体操作，二者特征图相减，平方后，经过一个BN和FC层再加上sigmoid，就可以得到一个二分类的结果。

takes visual features of two person images as inputs and feeds them through element-wise subtraction, element-wise square, a batch normalization layer, a fully-connected layer, and finally a sigmoid non-linearity function to output the probability that the input image pair belongs to the same person. This classifier is trained with binary cross-entropy loss.

![1567501776164](C:\Users\j00496872\Desktop\Notes\raw_images\1567501776164.png)

##### c. 身份判别器

身份判别器：保持行人编码器提取的身份特征。

原始图片和生成图片都通过一个ResNet，得到特征向量之后，进行和身份识别分类器一样结构的网络，得到最后二分类的结果。但要注意的是，这个ResNet和之前行人编码器E的ResNet并不共享权值，也就是完全两个网络,这里的ResNet更多是要区分真假图片, 而编码器的ResNet是为了得到Pose无关的身份信息，两个网络目的不同。

![1567502047333](C:\Users\j00496872\Desktop\Notes\raw_images\1567502047333.png)

##### d. 姿态判别器

姿态判别器：去除行人编码器里的姿态特征。

采用了 PatchGAN 的结构，直接将生成的行人图片和姿势图拼接在一起，然后进入一个四层的convolution-ReLU网络和sigmoid, 最后输出的是一个置信度的图，值越高说明相应位置的匹配度越高。

> Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. CVPR (2017)

![1567502814423](C:\Users\j00496872\Desktop\Notes\raw_images\1567502814423.png)

深入思考：如果包含冗余的姿态信息，那在生成行人的假图片时，可能姿势就和开始要求的有差别，因为它会对最后生成的行人姿态产生一定干扰，这样的话姿态判别这里的loss就会高。于是在慢慢训练之后，伴随着这个loss的下降，encoder被影响的会慢慢忽略对姿态信息的提取。

However, we observe that the pose discriminator $D_{pd}$ might overfit the poses, i.e., ==$D_{pd}$ might
remember the correspondences between specific poses and person appearances==, because each image’s
pose is generally unique. For instance, if we use a blue-top person’s pose as the target pose, the
generated image of a red-top person might end up having blue top. To solve this problem, we propose
an ==online pose map augmentation== scheme. During training, ==for each pose landmark, its 1-channel
Gaussian-like heat-map is obtained with a random Gaussian bandwidth in some specific range==. In
this way, we can create many pose maps for the same pose and mitigate the pose overfitting problem.

**Reconstruction loss**

生成图片和真实图片之间的相似度Loss

The responsibility of G is not only confusing the discriminators, but also generating images that are similar to the ground-truth images. However, ==the discriminators alone cannot guarantee generating human-perceivable images.== Therefore, a reconstruction loss is introduced to minimize the L1 differences between the generated image yk and its corresponding real image y0k

![1567503235463](C:\Users\j00496872\Desktop\Notes\raw_images\1567503235463.png)

**Same-pose loss**

两个分支生成图片之间的，应该越接近越好，如果是同一个行人的。

We input the same person’s two different images and the same target pose to both branches of our Siamese network, if the conditioning visual features in the two branches are truly only identity-related, then the two generated images should be similar in appearance.

![1567503376576](C:\Users\j00496872\Desktop\Notes\raw_images\1567503376576.png)

**最后Loss的组成**

![1567503411960](C:\Users\j00496872\Desktop\Notes\raw_images\1567503411960.png)



#### 训练过程

1）先训练行人encoder和身份识别分类器，只用到分类器的交叉熵。

2）固定行人encoder和身份识别分类器的网络参数，将判别器和生成器加入（二者是轮流训练的），进行训练。

3）最后将所有网络参数微调。

每一个batch有128对图片对，其中，32对是正样本。

All images are resized to 256*128. The Gaussian bandwidth for obtaining pose landmark heat-map is uniformly sampled in [4， 6].



####  实验结果

![1567502317708](C:\Users\j00496872\Desktop\Notes\raw_images\1567502317708.png) 

影响效果的关键点：

（1）引入孪生网络结构，使得多了一个same pose loss

（2）两个ResNet-的权重是不共享的。

从结果可以看出，==没有判别器，效果会差很多==。

![1567502465615](C:\Users\j00496872\Desktop\Notes\raw_images\1567502465615.png)