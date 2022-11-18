## EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

论文地址: https://arxiv.org/abs/1905.11946

作者：Mingxing Tan, Quoc V. Le

机构：Google Research, Brain Team, Mountain View, CA.

发表：ICML2019

代码：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

第三方实现的PyTorch代码：https://github.com/lukemelas/EfficientNet-PyTorch

文章地址：https://zhuanlan.zhihu.com/p/70369784



### 摘要

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. 

卷积神经网络通常都是先在固定资源预算下开发设计，然后如果资源有多余的话再将模型结构放大以便获得更好的精度。在本篇论文中，我们系统地研究了模型缩放并且仔细验证了**网络深度、宽度和分辨率之间的平衡可以导致更好的性能表现**。基于这样的观察，我们提出了一种新的缩放方法——使用一个简单高效的复合系数来完成对深度/宽度/分辨率所有维度的统一缩放。我们在MobileNets和ResNet上展示了这种缩放方法的高效性。

为了进一步研究，我们使用神经架构搜索设计了一个baseline网络，并且将模型放大获得一系列模型，我们称之为EfficientNets，它的精度和效率比之前所有的卷积网络都好。尤其是我们的EfficientNet-B7在ImageNet上获得了最先进的 84.4% 的top-1精度 和 97.1% 的top-5精度，同时比之前最好的卷积网络大小缩小了8.4倍、速度提高了6.1倍。我们的EfficientNets也可以很好的迁移，并且实现了最先进的精度——CIFAR-100（91.7%）、Flowers（98.8%）、其他3个迁移学习数据集。



### 介绍



为了获得更好的精度，放大卷积神经网络是一种广泛的方法。举个例子，ResNet可以通过使用更多层从ResNet-18放大到ResNet-200；目前为止，GPipe通过将baseline模型放大四倍在ImageNet数据集上获得了84.3%的top-1精度，然而，放大CNN的过程从来没有很好的理解过，**目前通用的几种方法是放大CNN的深度、宽度和分辨率，在之前都是单独放大这三个维度中的一个，尽管任意放大两个或者三个维度也是可能的，但是任意缩放需要繁琐的人工调参同时可能产生的是一个次优的精度和效率。**

在本篇论文中，我们想要研究和重新思考放大CNN的过程，尤其地，我们调查了一个中心问题：**是否存在一个原则性的放大CNN的方法实现更好的精度和效率**？我们的实验研究表明了平衡深度、宽度和分辨率这三个维度是至关重要的，令人惊讶的是这样的平衡可以通过简单的使用一组常量比率来缩放每一个维度，基于这个观察，我们提出了==一个简单高效的复合缩放方法，不像传统实践中任意缩放这些因子，我们的方法使用一组固定的缩放系数统一缩放网络深度、宽度和分辨率==。举个例子，如果想使用 $2^N$ 倍的计算资源，我们可以简单的对网络深度扩大 $\alpha^N$倍、宽度扩大 $\beta^N$、图像尺寸扩大 $\gamma^N$ 倍，这里的 $\alpha \ , \beta \ , \gamma$ 都是由原来的小模型上做微小的网格搜索决定的常量系数，Figure 2展示了我们的缩放方法和传统方法之间的区别。

![1592382475876](D:\Notes\raw_images\1592382475876.png)

直观来讲，==如果图像尺寸变大，复合的缩放方法会很有意义，因为当图像尺寸变大意味着网络需要更多层来增加接受野，同时需要更多的通道来捕捉更大图像上更多细粒度的模式信息==。事实上，以前的理论（Raghu等，2017; Lu等，2018）和实证结果（Zagoruyko＆Komodakis，2016）两者表明网络之间存在一定的关系宽度和深度，但据我们所知，我们是首先要凭经验量化三者之间的关系网络宽度，深度和分辨率的尺寸。

我们在已经存在的MobileNets和ResNets上展示了我们的缩放方法可以工作得很好，值得注意的是，==模型缩放的高效性严重地依赖于baseline网络==，为了进一步研究，我们使用网络结构搜索发展了一种新的baseline网络，然后将它缩放来获得一系列模型，称之为EfficientNets。Figure 1总结了ImageNet的表现，EfficientNets由于其他CNN网络，尤其地，我们的EfficientNet-B7超越了之前最好的GPipe的精度，但是参数量少了8.4倍、速度快了6.1倍。相比于广泛使用的ResNet-50，我们的EfficientNets-B4在相近的FLOPS下将top-1精度从76.3%提高到了82.6%。除了ImageNet，EfficientNets在其他数据集上表现也很好，在8个广泛应用的数据集中的5个实现了最先进的精度，然而比存在的卷积网络的参数量减少了21倍。

![1592383104899](D:\Notes\raw_images\1592383104899.png)



### 相关工作

**ConvNet精度**：自从AlexNet赢得2012的ImageNet竞赛，ConvNets随着变得更大获得了更高的精度，同时GoogleNet使用6.8M的参数获得了74.8%的top-1精度，SENet使145M参数获得了82.7%的top-1精度。最近，GPipe进一步提高了精度——使用了557M的参数获得了84.3%的top-1验证精度：它如此的大以至于需要将网络划分使用特定的并行化手段训练，然后将每一部分传递到一个不同的加速器。然而这些模型主要是为ImageNet设计，但是将其迁移到其他应用上时效果也很好。

**ConvNet效率**：深度网络的参数过多，模型压缩是一种通用的方法减小模型大小——平衡精度和效率。当移动网络变得无处不在时，我们也通常要手动设计高效的ConvNets，比如SqueezeNets、MobileNets、ShuffleNets。最近，神经网络结构搜索在设计高效的ConvNets变得越来越流行，并且通过广泛搜索网络宽度、深度、卷积核类型和大小得到了比手动设计的模型更高的精度。然而，目前还不清楚怎么将这些技术应用到更大的模型中，通常要更大的设计空间和更贵的微调成本，在本篇论文中，我们旨在研究效率设计更大规模的ConvNets，为了实现这个目标，我们采用模型缩放。

**模型缩放**：有很多的方法因不同的资源限制对ConvNet进行缩放：ResNet可以通过调整深度（缩小到ResNet-18，放大到ResNet-200），WideResNet和MobileNets可以通过对宽度（#channels）缩放。公认的是更大的输入图像尺寸有利于提高精度，同时需要更多的计算量。尽管之前的研究展示了网络深度和宽度对ConvNets的表现力很重要，它仍然是一个开放的问题来高效缩放ConvNet获得更好的效率和精度。我们的工作系统地研究了ConvNet对网络深度、宽度和分辨率这三个维度进行了缩放。



### 复合模型缩放

在本节中，我们将缩放问题公式化，研究了不同的方法并且提出了我们新的缩放方法。

#### 3.1. Problem Formulation

卷积层 i 可以用公式 $Y_i =F_i(X_i)$ 定义，$F_i$ 代表卷积操作， $Y_i$ 表示输出张量， $X_i$ 表示输入张量， $<H_i, W_i, C_i >$ 表示输入张量的形状， $H_i, W_i$ 是feature map的尺寸， $C_i$ 是feature map的输入通道数。卷积N可以用一系列组成层表示：$N=F_k \bigodot \cdots \bigodot F_2 \bigodot F_1=\bigodot_{j=1\dots k} F_j(X_1) $  。实践中，ConvNet层经常被划分为多个stages，并且每个stage的所有层共享相同的结构：举个例子，ResNet有5个stages，每个stage的所有层有相同的卷积类型（除了第一层有一个下采样），因此，我们可以将ConvNet定义为：

![1592383881742](D:\Notes\raw_images\1592383881742.png)

N是分类网络，X表示输入， $F_i$ 是基础网络层， $L_i$ 表示在第 $i$ 个stage中基础网络层 $F_i$ 的重复次数，总共有$s$个 stage。Figure 2(a)展示了一种代表性的ConvNet，其空间维度，即feature map的尺寸逐层递减，但是通道维度是逐渐增加的，举个例子，从最开始的输入维度<224,224,3>变化到最终的输出维度<7,7,512>。

不像规则的ConvNets设计，大多聚焦于发现最好的基础网络层 $F_i$ ，然后再利用模型缩放在不改变预先在baseline网络定义好的 $F_i$ 的前提下扩展网络深度 $L_i$  、宽度 $C_i$ 、分辨率  $<H_i, W_i>$  。通过固定 $F_i$  ，模型缩放简化了资源约束条件，但是它仍然有一个比较大的搜索空间  $<L_i, C_i, H_i, W_i>$  ，为了更进一步减小设计空间，我们限制所有层都统一以一个常量比例缩放，我们的目标是在给定资源预算下最大化模型精度，可以定义为如下优化问题：

![1592384106889](D:\Notes\raw_images\1592384106889.png)

这里的w,d,r是缩放网络分别对宽度、深度、分辨率的缩放系数， $\hat F_i , \hat L_i, \hat H_i, \hat W_i, \hat C_i $ 是预先在baseline网络中定义好的参数。

#### 3.2. Scaling Dimensions

**深度（d）**：缩放网络深度在许多ConvNets都有使用，直觉上更深的网络可以捕获到更丰富和更复杂的特征，在新任务上也可以泛化的更好。然而，更深的网络由于梯度消失问题（这里我更倾向于说成是网络退化问题）也更难训练。尽管有一些技术，例如跨层连接、批量归一化等可以有效减缓训练问题，但是深层网络的精度回报减弱了：举个例子，ResNet-1000和ResNet-101具有类似的精度，即使它的层数更多。Figure 3（中间的图）展示了我们在使用不同的深度系数d缩放网络的研究结果，更近一步的表明了精度回报的减弱问题。

**宽度（w）**：缩放网络宽度也是一种常用的手段，正如之前讨论过的，更宽的网络可以捕捉到更细粒度的特征从而易于训练。然而，非常宽而又很浅的网络在捕捉高层次特征时有困难，我们的实验结果Figure 3（左）表明了当网络宽度随着w变大时，精度很快就饱和了。

**Resolution（r）**：使用更高分辨率的输入图像，ConvNets可能可以捕捉到更细粒度的模式。从最早的 224x224，现在有些ConvNets为了获得更高的精度选择使用 229x229 或者 331x331。目前，GPipe使用 480x480 的分辨率获得了最先进的ImageNet精度，更好的精度比如 600x600 也被广泛使用在目标检测网络中。Figure 3（右）展示了缩放网络分辨率对精度的影响，同样可以看到在非常高的分辨率时网络精度回报会减弱。

![1592384299880](D:\Notes\raw_images\1592384299880.png)

由此，我们得到 **Observation 1：对网络深度、宽度和分辨率中的任何温度进行缩放都可以提高精度，但是当模型足够大时，这种放大的收益会减弱。**

#### 3.3. Compound Scaling

我们经验上可以观察到不同缩放维度之间是不独立的，直观上来讲，对于分辨率更高的图像，我们应该增加网络深度，因为需要更大的感受野来帮助捕获更多像素点的类似特征，同时也应该增加网络宽度来获得更细粒度的特征。这些直觉指导着我们去协调平衡不同缩放维度而不是传统的单个缩放维度。

为了验证我们的直觉，我们在不同的网络深度和分辨率下比较了宽度变化的影响，正如Figure 4中所示，如果我们在 d=1.0 和 r=1.0 时仅仅缩放网络宽度，精度很快就饱和了。但是在d=2.0 和 r=2.0时在相同的FLOPS下宽度缩放就可以获得更好的精度。这些结果导致我们得到了第二个观察结果。

**Observation 2：为了追去更好的精度和效率，在缩放时平衡网络所有维度至关重要。**

事实上，之前的一些工作已经开始在追去任意缩放网络深度和宽度，但是他们仍然需要复杂的人工微调。在本篇论文中，我们提出了一个新的复合缩放方法——使用一个复合系数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 统一缩放网络宽度、深度和分辨率：

![1592384542869](D:\Notes\raw_images\1592384542869.png)

这里的 $\alpha \ , \beta \ , \gamma$ 都是由一个很小范围的网络搜索得到的常量，直观上来讲， $\phi$ 是一个特定的系数，可以控制用于资源的使用量， $\alpha \ , \beta \ , \gamma$ 决定了具体是如何分配资源的。值得注意的是，常规卷积op的计算量是和 $d, w^2, r^2$ 成正比的，加倍深度会使得FLOPS加倍，但是加倍宽度和分辨率会使得FLOPS加4倍。由于卷积ops经常在CNN中占据了大部分计算量，使用等式（3）缩放ConvNet将会使得整体计算量近似增加 $(\alpha\cdot\beta^2\cdot\gamma^2 )^\phi $ 倍。在本篇论文中，我们对任意 $\phi$ 增加了约束 $ \alpha\cdot\beta^2\cdot\gamma^2 \approx 2$ ，整体的计算量近似增加了 $2^{\phi}$ 倍。

![1592384500395](D:\Notes\raw_images\1592384500395.png)



### EfficientNet 网络结构

因为模型缩放不改变baseline网络中的 $\hat F_i$ ，所以有一个好的baseline网络是很重要的，我们使用之前的ConvNets对我们的缩放方法做了评估，但是为了更好验证我们的缩放方法的高效性，我们也提出了一种基于移动应用的baseline模型——EfficientNet。

受到MnasNet的启发，我们也开发了一种多目标的神经网络结构搜索同时优化精度和FLOPS，我们的搜索空间和MnasNet相同，同时我们的优化目标为 $ ACC(m)×[F LOP S(m)/T]^w$ ，这里的 $ACC(m)$ 和 $FLOPS(m)$分别是模型m的精度和计算量， $T$ 是目标计算量， $w=-0.07$ 是一个超参数用来权衡精度和FLOPS。不像MnasNet中的优化目标，==这里优化的是FLOPS而不是延迟==，因为我们没有说是要在特定的硬件平台上做加速。我们的搜索方法得到了一个高效的网络，我们称之为EfficientNet-B0，因为我们使用的搜索空间和MnasNet相似，所以得到的网络结构也很相似，不过我们的EfficientNet-B0稍微大了点，因为我们的FLOPS预算也比MnasNet中大（400M）。table 1展示了EfficientNet-B0的结构，它的==主要构建块就是移动倒置瓶颈MBConv==，其网络结构如下：

![1592385691120](D:\Notes\raw_images\1592385691120.png)

![1592385165839](D:\Notes\raw_images\1592385165839.png)

然后以EfficientNet-B0为baseline模型，我们将我们的复合缩放方法应用到它上面，分为两步：

- **STEP 1**：我们首先固定 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi+%3D1) ，假设有相比于原来多了2倍的资源，我们基于等式（2）和（3）先做了一个小范围的搜索，最后发现对于EfficientNet-B0来说最后的值为 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha+%3D+1.2%2C%5Cbeta+%3D+1.1%2C%5Cgamma+%3D+1.15) ，在 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha+%5Ccdot+%5Cbeta+%5E%7B2%7D+%5Ccdot+%5Cgamma+%5E%7B2%7D%5Capprox+2) 的约束下；
- **STEP 2**：接着我们固定 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%2C%5Cbeta%2C%5Cgamma) 作为约束，然后利用不同取值的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 对baseline网络做放大，来获得Efficient-B1到B7；

值得注意的是，直接在一个大模型上搜索得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%2C%5Cbeta%2C%5Cgamma) 可能会导致一个更好的表现，但是搜索成本会很高，我们的方法解决了这个问题——STEP 1时只在小的baseline网络中做了一次搜索得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%2C%5Cbeta%2C%5Cgamma) ，然后对于其他的模型都使用的是相同的 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%2C%5Cbeta%2C%5Cgamma) ，只是通过 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 来调整模型的大小和FLOPS。



### 实验结果

为了验证我们的方法，我们首先将缩放方法应用到MobileNets和ResNets中，Table 3展示了以不同的方法缩放的ImageNet结果。与其他单一维度的缩放方法相比，我们的复合缩放方法精度提高了。

![1592385966568](D:\Notes\raw_images\1592385966568.png)

我们采用了和MnasNet相似的设置对我们的EfficientNet模型进行训练：

- RMSProp优化器，decay为0.9，momentum为0.9；
- batch norm momentum为0.99；
- weight decay为1e-5；
- 初始学习率为0.256，并且每2.4个epoches衰减0.97；
- 同时使用了swish activation，固定的增强技术，随机深度（drop connect ratio 为0.2），同时更大的模型需要更强的正则化，所以我们线性地增加dropout比率从EfficientNet-B0的0.2 到 EfficientNet-B7的0.5；

![1592386074268](D:\Notes\raw_images\1592386074268.png)

![1592386148168](D:\Notes\raw_images\1592386148168.png)

![1592386117311](D:\Notes\raw_images\1592386117311.png)

![1592386195598](D:\Notes\raw_images\1592386195598.png)

![1592386218059](D:\Notes\raw_images\1592386218059.png)



### 讨论

下面是使用了我们的复合缩放方法后精度的提升效果：

![1592385275111](D:\Notes\raw_images\1592385275111.png)

为了更近一步的理解为什么我们的复合缩放方法就比其他的方法效果好，Figure 7 比较了使用不同方法的模型得到的class activation map的效果图，所有的这些模型都是从baseline模型缩放得到的，它们的统计特性如图 Table 7。图片是随机从验证集中得到的，正如在figure中可见，复合缩放得到的模型倾向聚焦于与更多目标细节相关的区域，而其他的模型要么缺乏目标细节，要么不能捕捉到图片中所有的目标。

![1592385258845](D:\Notes\raw_images\1592385258845.png)