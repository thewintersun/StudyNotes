## Once for All: Train One Network and Specialize it for Efficient Deployment

[Once For All: Train One Network and Specialize It for Efficient Deployment](https://arxiv.org/pdf/1908.09791.pdf)
Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
*International Conference on Learning Representations (ICLR), 2020.*[[paper](https://arxiv.org/pdf/1908.09791.pdf)] [[code](https://github.com/mit-han-lab/once-for-all)]

代码地址：https://github.com/mit-han-lab/once-for-all

We address the challenging problem of efficient deep learning model deployment, where the goal is to ==design neural network architectures that can fit different hardware platform constraints==. Most of the traditional approaches either manually design or use Neural Architecture Search (NAS) to find a specialized neural network and train it from scratch for each case, which is computationally expensive and unscalable. Our key idea is to ==decouple model training from architecture search to save the cost==. To this end, we propose to train a once-for-all network (OFA) that ==supports diverse architectural settings (depth, width, kernel size, and resolution)==. Given a deployment scenario, we can then quickly get a specialized sub-network by selecting from the OFA network without additional training. To prevent interference between many sub-networks during training, we also propose a novel ==progressive shrinking algorithm==, which can train a surprisingly large number of sub-networks simultaneously. 

Extensive experiments on various hardware platforms (CPU, GPU, mCPU, mGPU, FPGA accelerator) show that OFA consistently outperforms SOTA NAS methods (up to 4.0% ImageNet top1 accuracy improvement over MobileNetV3) while reducing orders of magnitude GPU hours and emission. In particular, OFA achieves a new SOTA 80.0% ImageNet top1 accuracy under the mobile setting ( < 600M FLOPs). Code and pre-trained models are released at [this https URL](https://github.com/mit-han-lab/once-for-all)

**Train once, specialize for many deployment scenarios**

![img](https://camo.githubusercontent.com/ca2bc707dda0be2a37edaaa5bdd1cc85da0c3f3e/68747470733a2f2f68616e6c61622e6d69742e6564752f66696c65732f4f6e6365466f72416c6c2f666967757265732f6f766572766965775f6e65772e706e67)

1、本文的主要思想是使用大量的GPU资源首先训练一个整体网络，再使用渐进收缩算法逐步训练子网络，之后根据不同的硬件要求从这个训练好的网络中选择子网络进行应用。所以在某种程度上，本文方法和One-Shot类方法的思想基本相同。

2、本文的主要贡献：（1）提出了渐进收缩算法来防止子网络训练耦合；（2）训练一次以后，针对不同的硬件平台不需要再进行网络架构搜索，直接根据要求查表即可获得满足要求的子网络（即文中所述的将专用平台的深度学习部署成本从O(N)降到了O(1)，这其实和Single Path One-Shot非常类似，后者的不足在于搜索出来满足条件的网络架构后还需要从头训练，而本文不需要）

![img](https://pic4.zhimg.com/v2-e2109b11dd0cf2ca6dd6132bf00daef7_b.jpg)

#### **Method**

1、问题定义：本文对搜索目标建模为如下形式。W是网络，arch是网络配置，C是选择方案（val依然是从训练集选择一部分数据作为验证集）

![img](https://pic3.zhimg.com/v2-7d5008e20d325d6ff1aef105251e2c1e_b.jpg)

2、本文定义的网络包含多个stage，每个stage包含一系列分辨率相同的block。训练时并不是从头训练整个网络，而是先训练包含最大尺寸（宽度，深度和卷积核尺寸）的网络，然后渐进缩小各种尺寸进行微调，流程图如下（文中只有一小段在具体谈这个问题，没有讲的特别清楚，进一步理解需要看源码）：

![img](https://pic4.zhimg.com/v2-53aa5e8cdb97c3a42379f96b325adacb_b.jpg)

3、使用Once-for-all网络进行专用模型部署：OFA网络可以和任意的搜索算法相结合，然后取其中的一些子网络，计算出它们的精度和延迟表格，根据不同的硬件要求选择合适的网络。（测量精度时测量一定步幅的一系列分辨率下的精度，如果该分辨率没有测，就直接使用如下公式进行线性近似）

![img](https://pic2.zhimg.com/v2-72183c51cdf3e2c09ce249269cb8cbed_b.png)

#### **Experiments**

1、使用和ProxylessNAS相同的树状结构的PyramidNet作为搜索空间。在ImageNet上子网络的表现结果：

![img](https://pic2.zhimg.com/v2-04cd6869e5e1ae72b65ac4ba5eaa5c1d_b.jpg)

2、在三星Note8上进行对比：

![img](https://pic3.zhimg.com/v2-3e852d686ff3756206b0f251dad514b6_b.jpg)

4、在不同的硬件平台（移动设备，CPU和GPU）上的专用部署结果对比：

![img](https://pic2.zhimg.com/v2-2c0f6f675eb8ea10c753b9fa5672cb3d_b.jpg)

#### **Thinkings**

1、文章的具体训练过程没有讲清楚，也没有开源模型和训练代码。

2、总体的思路感觉并不是很创新，只是说针对不同的硬件平台可以直接搜网络进行部署，但是初期的训练成本仍然很高（1,200 GPU hours on V100 GPUs），并且搜索时需要进行大量的前向传播，文中写道也需要大约200 GPU hours。Single Path One-Shot训练足够久的话可能也会达到这种效果（一次训练，之后搜索网络架构不需要从头训练）。

3、本文的主要贡献在于提出Progressive Shrinking，文中写道该方法可以防止子网络之间相互干扰（是否可以理解为防止各子网络的训练耦合）。