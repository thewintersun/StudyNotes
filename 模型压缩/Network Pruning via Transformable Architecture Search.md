## Network Pruning via Transformable Architecture Search

论文：  https://papers.nips.cc/paper/6813-runtime-neural-pruning.pdf

论文地址：https://arxiv.org/abs/1905.09717

作者： Xuanyi Dong, Yi Yang

机构：悉尼科技大学，百度

代码地址：https://github.com/D-X-Y/AutoDL-Projects



### 摘要

网络修剪可减少过度参数化的网络的计算成本，而不会影响性能。现有的修剪算法会预先定义修剪网络的宽度和深度，然后将参数从未修剪的网络传输到修剪的网络。为了突破修剪网络的结构限制，我们提出应用神经架构搜索（NAS）直接搜索具有灵活的通道大小和层大小的网络。

通过最小化修剪网络的损失来学习通道/层的数量。修剪后的网络的特征图是==K个特征图==片段的集合（由不同大小的K个网络生成），这些片段是根据概率分布进行采样的。损耗不仅可以反向传播到网络权重，还可以反向传播到参数化分布，以显式调整通道/层的大小。

另外，我们应用逐通道插值（填充）以使具有不同通道大小的特征图在聚合过程中保持对齐。

每个分布中最大概率的size用作修剪网络的宽度和深度，修剪网络的参数是通过知识迁移（例如知识蒸馏）从原始网络中获知的。

与传统的网络修剪算法相比，在CIFAR-10，CIFAR-100和ImageNet上进行的实验证明了我们的网络修剪新观点的有效性。进行了各种搜索和知识转移方法以显示这两个组件的有效性。   



### 介绍

深度卷积神经网络（CNN）变得越来越广泛，以在不同的应用程序上实现高性能。尽管它们取得了巨大的成功，但将它们部署到资源受限的设备（如移动设备和无人机）上是不切实际的。解决此问题的直接解决方案是使用网络修剪减少过度参数化的CNN的计算成本[29、12、13、20、18]。

> [29] Y. LeCun, J. S. Denker, and S. A. Solla. Optimal brain damage. In The Conference on Neural Information Processing Systems (NeurIPS), pages 598–605, 1990.
>
> [12] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations (ICLR), 2015.
>
> [13] S. Han, J. Pool, J. Tran, and W. Dally. Learning both weights and connections for efficient neural network.In The Conference on Neural Information Processing Systems (NeurIPS), pages 1135–1143, 2015.
>
> [20] Y. He, X. Zhang, and J. Sun. Channel pruning for accelerating very deep neural networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1389–1397, 2017.
>
> [18] Y. He, G. Kang, X. Dong, Y. Fu, and Y. Yang. ==Soft filter pruning== for accelerating deep convolutional neural networks. In International Joint Conference on Artificial Intelligence (IJCAI), pages 2234–2240, 2018.

如图1(a)所示，用于网络修剪的典型pipeline是通过删除冗余过滤器，然后基于原始网络微调修剪网络来实现的。基于滤波器重要性的不同技巧被使用，例如==滤波器的L2范数==[30]，==重构误差==[20]和==可学习的缩放因子==[32]。最后，研究人员对修剪后的网络应用了各种==微调策略==[30，18]，以有效地传递未修剪网络的参数并最大化修剪后的网络的性能。

> [30] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf. Pruning filters for efficient convnets. In
> International Conference on Learning Representations (ICLR), 2017.
>
> [32] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang. Learning efficient convolutional networks through ==network slimming==. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2736–2744, 2017.

![1583842315577](D:\Notes\raw_images\1583842315577.png)

传统的网络修剪方法在保持准确性的同时，对网络压缩产生了有效的影响。他们的网络结构是直观设计的，例如，在每一层中修剪30％的滤波器[30、18]，预测稀疏率[15]或利用正则化[2]。修剪后的网络的精度受手工制作的结构或结构规则的限制。为了克服这一限制，我们应用神经体系结构搜索（NAS）将体系结构的设计转变为学习过程，并提出了一种新的网络修剪范式，如图1（b）所示。

 现有的NAS方法优化了网络拓扑，而本文的重点是自动网络规模。为了满足要求并公平地比较以前的修剪策略，我们==提出了一种称为可转换体系结构搜索（TAS）的新NAS方案==。**TAS旨在搜索最佳网络规模而不是拓扑，通过最小化计算成本进行正则化（浮点操作FLOPs）。然后，通过知识迁移来学习搜索/修剪网络的参数**[21、44、46]。

==TAS是一种可微分的搜索算法，可以有效而高效地搜索网络的宽度和深度==。具体而言，信道/层的不同候选以可学习的概率被附加。通过反向传播修剪后的网络产生的损失来学习概率分布，修剪后的网络的特征图是根据概率分布采样的K个特征图片段（不同大小的网络输出）的集合。这些不同通道大小的特征图借助通道方式插值进行汇总。每个分布中大小的最大概率用作修剪网络的宽度和深度。 

在实验中，我们表明，通过知识蒸馏（KD）传递参数的搜索架构优于CIFAR-10，CIFAR-100和ImageNet上以前的最新修剪方法。我们还对传统的人工修剪方法[30，18]和随机架构搜索方法[31]生成的架构测试了不同的知识迁移方法。对不同架构的相同改进证明了知识迁移的普遍性。

> [31] H. Liu, K. Simonyan, and Y. Yang. Darts: Differentiable architecture search. In International Conference on Learning Representations (ICLR), 2019.

TAS剪枝过程包括三个步骤：

（1）通过标准分类训练程序训练未修剪的大型网络。

（2）通过建议的TAS搜索小型网络的深度和宽度。

（3）通过简单的KD方法将知识从未修剪的大型网络转移到搜索的小型网络

### 相关工作

网络修剪[29，33]是压缩和加速CNN的有效技术，因此允许我们在存储和计算资源有限的硬件设备上部署有效的网络。已经提出了多种技术，例如==低秩分解==[47]，==权重修剪==[14、29、13、12]，==通道修剪==[18、33]，==动态计算==[9、7]和==量化==[23， 1]。它们有两种模式：非结构化修剪[29、9、7、12]和结构化修剪[30、20、18、33]。

**非结构化修剪方法**通常会 强制卷积权重[29、14]或特征图[7、9]稀疏。非结构化修剪的先驱 LeCun等[29]和Hassibi等[14]研究了使用二阶导数信息来修剪浅层CNN的权重。在深度网络于2012年诞生后[28]，Han等人[12，13，11]提出了一系列==基于L2正则化==获得高度压缩的深度CNN的工作。经过这一发展，许多研究人员探索了不同的正则化技术来提高稀疏度，同时又保持了准确性，例如==L0正则化==[35]和==输出灵敏度==[41]。由于这些非结构化方法使大型网络稀疏而不是改变网络的整体结构，因此它们==需要针对依赖项的专用设计[11]和特定的硬件==来加快推理过程。

**结构化修剪方法**[30、20、18、33]的目标是对卷积过滤器或所有层进行修剪，因此可以轻松开发和应用修剪后的网络。该领域的早期工作[2，42]利用==组Lasso来实现深度网络的结构化稀疏性==。之后，李等人[30]提出了典型的三阶段修剪范例（训练大型网络，修剪，再训练）。这些修剪算法将具有较小范数的过滤器视为不重要，并且倾向于修剪它们，但是这种假设在深层非线性网络中不成立[43]。因此，许多研究人员专注于信息过滤器的更好标准。例如，刘等[32]利用L1正则化；Ye等[43]对ISTA施加了惩罚(applied a ISTA penalty)；He等[19]利用了基于几何中位数的标准。与以前的修剪pipeline相反，我们的方法==允许显式优化通道/层的数量==，从而使学习到的结构具有高性能和低成本。

> [33] Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell. Rethinking the value of network pruning. In
> International Conference on Learning Representations (ICLR), 2018.

除了信息过滤器的标准，网络结构的重要性在[33]中被提出。通过自动确定每一层的修剪和压缩率，某些方法可以隐式地找到特定于数据的架构[42、2、15]。相比之下，我们使用NAS明确发现了该架构。先前的大多数NAS算法[48、8、31、40]会自动发现神经网络的拓扑结构，而==我们专注于搜索神经网络的深度和宽度==。基于强化学习（RL）的[48，3]方法或基于进化算法的[40]方法可以搜索具有灵活宽度和深度的网络，==但是它们需要大量的计算资源，因此无法直接用于大规模目标数据集==。可微分的方法[8，31，4]显着降低了计算成本，但它们通常==假定不同搜索候选集的通道数相同==。 TAS是一种可微分的NAS方法，它能够有效地搜索具有灵活宽度和深度的迁移网络。

> [48] B. Zoph and Q. V. Le. Neural architecture search with reinforcement learning. In International Conference on Learning Representations (ICLR), 2017.
>
> [8]  Dong and Y. Yang. Searching for a robust neural architecture in four gpu hours. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1761–1770, 2019.
>
> [31] H. Liu, K. Simonyan, and Y. Yang. Darts: Differentiable architecture search. In International Conference on Learning Representations (ICLR), 2019.
>
> [3] H. Cai, T. Chen, W. Zhang, Y. Yu, and J. Wang. Efficient architecture search by network transformation. In AAAI Conference on Artificial Intelligence (AAAI), pages 2787–2794, 2018.
>
> [40] E. Real, A. Aggarwal, Y. Huang, and Q. V. Le. Regularized evolution for image classifier architecture
> search. In AAAI Conference on Artificial Intelligence (AAAI), 2019.
>
> [4] H. Cai, L. Zhu, and S. Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations (ICLR), 2019.

网络迁移(Network transformation) [5，10，3]也研究了网络的深度和宽度。 Chen等[5]手动拓宽和加深网络，并建议使用Net2Net初始化更大的网络。 Ariel等[10]提出了一种启发式策略，通过在缩小和扩展之间交替来找到合适的网络宽度。蔡等[3]利用RL代理来增加CNN的深度和宽度，而我们的TAS是一种可微分的方法，不仅可以扩大CNN，而且可以缩小CNN。

在网络修剪相关的文献中，知识迁移已被证明是有效的。网络的参数可以从预训练的初始化[30，18]中迁移。Minnehan等[37]通过逐块重构损失来迁移未压缩网络的知识。在本文中，我们应用了一种简单的KD方法[21]来进行知识转移，从而使架构搜索具有良好的性能。

> [21] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. In The Conference on Neural Information Processing Systems Workshop (NeurIPS-W), 2014.

### 方法

我们的修剪方法包括三个步骤：（1）通过标准分类训练程序训练未修剪的大型网络。 （2）通过TAS搜索小型网络的深度和宽度。（3）通过简单的KD方法将知识从未修剪的大型网络迁移到搜索得到的小型网络[21]。 下面将介绍背景知识，TAS的详细信息，并说明知识迁移过程。

#### 迁移架构搜索

网络通道修剪旨在减少网络每一层中的通道数量。 给定输入图像，网络会将其作为输入，并在每个目标类别上产生概率。 假设X和O是第l个卷积层的输入和输出特征张量（我们以3×3卷积为例），该层计算过程如下：
$$
O_j=\sum_{k=1}^{c_{in}}{X_k,:,: \ast W_{j,k},:,:}  \qquad  where  \; 1\leq j \leq c_{out} \tag{1}
$$
其中$W\in R^{c_{out \times c_{in} \times 3 \times 3} }$表示卷积核权重，$c_{in}$为输入通道，$c_{out}$为输出通道。 $Wj,k,:,:$ 对应第k个输入通道和第j个输出通道。∗表示卷积运算。 通道修剪方法可以减少cout的数量，因此，也减少了下一层的cin。

**搜索宽度** 我们使用参数$\alpha \in R^{|C|}$来表示一层中可能的通道数分布，其中，max(C)≤cout 。 选择通道数量的第j个候选的概率可以表示为：
$$
p_j=\frac{exp(\alpha_j)}{\sum_{k=1}^{|C|}{exp(\alpha_k)}} \qquad where\; 1\leq j \leq |C| \tag{2}
$$
但是，**上述过程中的采样操作是不可微的**，这阻止了我们将梯度从pj反向传播到αj。 受[8]的启发，我们应用Gumbel-Softmax [26，36]来软化采样过程以优化α：

> 可以参考《[Gumbel-Softmax的采样技巧](https://blog.csdn.net/weixin_40255337/article/details/83303702)》
>
> 可以参考《[Gumbel-Softmax 对离散变量再参数化](https://zhuanlan.zhihu.com/p/50065712)》

![img](http://dongzebo.com/images/20191029210453_0.png)

其中U(0,1)表示0和1之间的均匀分布。τ是softmax温度参数。 当τ→0时，$\hat{p}=[\hat{p}_1,…,\hat{p}_j,…]$ 变为one-shot，并且基于p的Gumbel-softmax分布与类别分布相同。当τ→∞，Gumbel-softmax分布在C上变为均匀分布。







我们方法中的特征图定义为具有不同大小的原始特征图片段的加权和，其中权重为 $\hat{p}$。 通过逐通道插值（CWI）对齐具有不同大小的特征图，以便计算加权和。 为了减少内存成本，我们选择索引 $I\subseteq[|C|]$ 为的小子集进行聚合，而不是使用所有候选集。此外，权重根据所选size的概率重新归一化，公式如下：

![1584092195124](D:\Notes\raw_images\1584092195124.png)

其中τ^pτp^表示由^pp^参数化的多项式概率分布。CWI是将不同尺寸的特征图对齐的一般操作。它可以通过多种方式实现，例如空间迁移网络（ spatial transformer network）的3D变体[25]或自适应池操作[16]。在本文中，我们选择3D自适应平均池化操作[16]作为CWI2CWI2，因为它没有带来额外的参数并且可以忽略不计的额外成本。我们在CWI之前使用批规范化[24]来规范化不同的片段。图2以|I|=2|I|=2为例阐述了上述过程。



![img](http://dongzebo.com/images/201910301502_0.png)



###   实验结果

![1583842539455](D:\Notes\raw_images\1583842539455.png)



![1583842462447](D:\Notes\raw_images\1583842462447.png)