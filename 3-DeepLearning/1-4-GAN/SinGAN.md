###  SinGAN: Learning a Generative Model from a Single Natural Image

论文地址：https://arxiv.org/pdf/1905.01164.pdf

作者：Tamar Rott Shaham

机构：Google Research

代码：https://github.com/tamarott/SinGAN 



#### 摘要

We introduce SinGAN, an ==unconditional generative model== that can be learned from a single natural image. Our model is trained to capture the internal distribution of patches within the image, and is then able to generate high quality, diverse samples that carry the same visual content as the image. SinGAN contains a pyramid of fully convolutional GANs, each responsible for learning the patch distribution at a different scale of the image. ==This allows generating new samples of arbitrary size and aspect ratio, that have significant variability, yet maintain both the global structure and the fine textures of the training image.==  In contrast to previous single image GAN schemes, our approach is not limited to texture images, and is not conditional (i.e. it generates samples from noise). User studies confirm that the generated samples are commonly confused to be real images. We illustrate the utility of SinGAN in a wide range of image manipulation tasks.



#### **研究背景**

生成对抗网络（GAN）在对视觉数据的高维分布建模方面取得了巨大飞跃。特别是用类别特定数据集（如人脸、卧室）进行训练时，非条件GAN在生成逼真的、高质量样本方面已取得显著成功。但建模具有多个类别、高度多样化的数据集（如ImageNet）的分布仍然是一项重大挑战，并且通常需要根据另一种输入信号来调节生成或为特定任务训练模型。

本文将GAN带入了一个新领域--从单幅自然图像中学习非条件生成模型。对单幅自然图像中的图像内部分布进行建模已被公认为是许多计算机视觉任务的有用先验，单幅自然图像通常具有足够的内部统计信息，可以使网络学习到一个强大的生成模型。作者提出了一种具有简单统一架构的模型SinGAN，能够处理包含复杂结构和纹理的普通自然图像，而不必依赖于具有同一类别图像的数据集。这是通过金字塔结构的全卷积GAN实现的，每个GAN负责捕捉不同尺度的图像分布。经过训练后，SinGAN可以以任意尺寸生成各种高质量的图像样本，这些样本在语义上类似于训练图像，但包含新的目标和结构，如图1所示。并且多种图像处理任务都可以应用SinGAN，如图像绘制、编辑、融合，超分辨率重建和动画化。

![img](https://image.jiqizhixin.com/uploads/editor/b0ce9ce6-5951-4c2a-a7f9-5e43779e2484/640.png)

图 1 从单幅训练样本学习的图像生成模型。本文提出SinGAN--一种在单幅自然图像上训练的新型非条件生成模型。SinGAN使用多尺度对抗训练方案来跨多个尺度学习图像内部统计信息，可以将其用于生成新的逼真图像样本，该样本在生成新目标和结构的同时保留原始图像块分布。

SinGAN在图像处理中的应用

> 绘画转真实图像， 2. 图像编辑，3.图像融合，4.图像超分辨，5. 动画制作。

![1573095546910](D:\Notes\raw_images\1573095546910.png)

#### **相关工作**

**1.单幅图像深度模型。**最近的一些工作提出在单个样本上训练一个“过拟合”的深度模型，它们都是为特定任务设计的，如超分辨率重建、纹理扩展等。Shocher等提出的InGAN是第一个基于内部GAN的单幅自然图像训练模型，其生成的样本取决于输入图像（即将图像映射到图像），并不能绘制随机样本。而本文框架是纯粹生成式的（即将噪声映射到图像样本），因此适合许多不同的图像处理任务。目前非条件单幅图像GAN模型仅对有纹理的图像进行研究，当在没有纹理的图像上训练这些模型时，不会生成有意义的样本，而本文方法不限于纹理，可以处理一般的自然图像，如图1。

**2.用于图像处理的生成模型。**在许多不同的图像处理任务中，基于GAN的方法已经证明了对抗学习的巨大优势，包括交互式图像编辑，草图合成图像和其他图像到图像翻译任务。但是，所有这些方法都是在特定类别的数据集上进行训练的，通常需要另外的输入信号调整生成。本文不关注如何获取同一类图像间的共同特征，而是考虑使用不同的训练数据源--单幅自然图像的多个尺度上所有重叠的图像块。作者表明，可以从这样的数据中学习强大的生成模型，并将其用于许多图像处理任务中。

#### **方法**

本文目标是学习一个非条件的生成模型，该模型可以捕捉单个训练图像x的内部统计信息。此任务在概念上与常规GAN设置相似，不同之处在于，此处训练样本是单幅图像不同尺度下的采样图像，而不是数据集中的整个图像样本。

模型选择处理更一般的自然图像，赋予模型生成纹理外的其他功能。为了捕捉图像中目标形状和排列位置这样的全局属性（如天空在顶部，地面在底部），以及精细的细节和纹理信息，SinGAN包含具有层级结构的patch-GANs（马尔可夫判别器），其中每个判别器负责捕捉x不同尺度的分布，如图2所示。虽然在GAN中已经探索使用了类似的多尺度结构，但本文还是第一个为从单幅图像进行内部学习而探索的网络结构。

![img](https://image.jiqizhixin.com/uploads/editor/c611834a-ff60-4f9d-8649-cb852af7086f/640.png)

![img](https://image.jiqizhixin.com/uploads/editor/81a6c975-0422-4341-903c-0353bc32e33e/640.png)

**1.多尺度结构**

![img](https://image.jiqizhixin.com/uploads/editor/85d2453f-8d54-4ae1-9671-6cbf26bf97d6/640.png)

**2.训练过程**

![img](https://image.jiqizhixin.com/uploads/editor/88e93568-ec58-41c3-88fc-c0d8360b866f/640.png)

#### **实验结果**

作者在图像场景跨度很大的数据集上对SinGAN进行了定性和定量的测试，定性生成的图像如图1和图4所示。SinGAN很好地保留目标的全局结构和较好的纹理信息，如图1中的山、图4中的热气球或金字塔。此外，模型很真实地合成了反射和阴影。

![img](https://image.jiqizhixin.com/uploads/editor/22ba0174-fe3a-4fab-b869-ef3c72ea992e/640.png)图 4 随机生成的图像样本在训练时使用较少数目的尺度，则最粗尺度的有效感受野会更小，从而只能捕获精细纹理。随着尺度数量的增加，出现了更大的支撑结构，并且更好地保留全局目标的排列（位置关系）。测试时可以选择开始生成的尺度，SinGAN的多尺度结构可以控制样本间差异的总量。从最粗尺度开始生成会导致整体结构变化很大，在某些具有较大的显著目标的情况下，可能会生成不真实的样本。当从较细的尺度开始，可以保持整体结构完整，同时仅会改变更精细的图像特征。

为了量化生成图像的真实性以及它们捕捉训练图像内部统计信息的程度，作者使用两个度量：AMT真假用户调研和FID的单幅图像版本。AMT测试结果发现，SinGAN可以生成很真实的样本，人类判别的混淆率较高。利用单幅图像FID量化SinGAN捕捉x内部统计信息的能力的结果如表1所示。从N-1尺度开始生成的SFID评价值比从N尺度开始生成低，这与用户调研一致。作者还报告了SIFID与假图像混淆率之间的相关性，两者之间存在显著的负相关性，这意味着较小的SIFID通常表示较大混淆率。

​																		表 1 两种模式的SIFD值

![img](https://image.jiqizhixin.com/uploads/editor/db81880c-3978-4a73-b96c-8ffaaf17d78e/640.png)



#### **结论**

本文介绍了一种可以从单幅自然图像中学习的新型非条件生成框架--SinGAN。证明了其不仅可以生成纹理，还具有为复杂自然图像生成各种逼真样本的能力。与外部训练的生成方法相比，内部学习在语义多样性方面具有固有的限制。例如，如果训练图像只包含一条狗，SinGAN不会生成不同犬种的样本。不过，作者通过实验证明，SinGAN可以为多种图像处理任务提供非常强大的工具。