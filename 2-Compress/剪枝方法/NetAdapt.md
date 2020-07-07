## NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications

论文地址：https://arxiv.org/abs/1804.03230

作者：Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, Hartwig Adam

机构：MIT, Google

发表：ECCV 2018

项目地址：https://web.mit.edu/netadapt/

代码地址：https://github.com/denru01/netadapt



### 摘要

This work proposes an algorithm, called NetAdapt, that automatically adapts a pre-trained deep neural network to a mobile platform given a resource budget. While many existing algorithms simplify networks based on the number of MACs or weights, optimizing those indirect metrics may not necessarily reduce the direct metrics, such as latency and energy consumption. ==To solve this problem, NetAdapt incorporates direct metrics into its adaptation algorithm. These direct metrics are evaluated using empirical measurements, so that detailed knowledge of the platform and toolchain is not required.== NetAdapt automatically and progressively simplifies a pre-trained network until the resource budget is met while maximizing the accuracy. Experiment results show that NetAdapt achieves better accuracy versus latency trade-offs on both mobile CPU and mobile GPU, compared with the state-of-the-art automated network simplification algorithms. For image classification on the ImageNet dataset, NetAdapt achieves up to a 1.7X speedup in measured inference latency with equal or higher accuracy on MobileNets (V1&V2).

NetAdapt作用：用户可以自动简化一个预训练的网络以使其达到硬件资源限制，同时最大化精确度。
NetAdapt简介：将 direct metrics（延时，能量，内存占用等等， 等等，或者是这些指标的结合）并入自适应算法，direct metrics 用empirical measurements （实证测量）分析，这样就不用对特殊平台的细节进行了解了（当然将来的改进可以对平台细节进行了解）。在每次迭代中，NetAdapt会差生很多network proposal并将他们在目标平台上测量，以测量结果指导NetAdapt产生下一批network proposal。

### 介绍

![1592465768527](D:\Notes\raw_images_2\1592465768527.png)

Fig. 1. NetAdapt automatically adapts a pretrained network to a mobile platform given a resource budget. This algorithm is guided by the direct metrics for resource consumption. At each iteration, NetAdapt generates many network proposals and measures the proposals on the target platform. The measurements are used to guide NetAdapt to generate the next set of network proposals at the next iteration.

问题分析：

![1592469555874](D:\Notes\raw_images_2\1592469555874.png)

问题分解：

![1592469578177](D:\Notes\raw_images_2\1592469578177.png)

解决方法：

![1592469616929](D:\Notes\raw_images_2\1592469616929.png)

伪代码图示：

![1592471497458](D:\Notes\raw_images_2\1592471497458.png)

Fig. 2. This gure visualizes the algorithm flow of NetAdapt. At each iteration, NetAdapt decreases the resource consumption by simplifying (i.e., removing lters from) one layer. In order to maximize accuracy, it tries to simplify each layer individually and picks the simplied network that has the highest accuracy. Once the target budget is met, the chosen network is then fine-tuned again until convergence.

比如（ResNet50)有50个Conv或FC，这里K就是50，每次迭代会生成50个子网络，50个子网络都只剪裁当前的K层，其他层不变。50个子网络都会进行ShortFineTune，然后在plateform 衡量后选择ACC最高的网络，再进行下一轮剪裁。

**choose number of filters：**尽可能选择卷积核多的，又满足目前资源限制的network，注意一层的卷积核被移除，也会影响其下一个层的通道数。
**choose which filters：**保留L2范数最大的N个卷积核，N是由前一步确定的要保留的卷积核个数。
**short-/long term fine-tine：**network-wise end-to-end fine-tuning。

#### 如何选择Filter的个数：快速资源消耗预测。

NetAdapt uses empirical measurements to determine the number of filters to keep in a layer given the resource constraint. We solve this problem by building ==layer-wise look-up tables== with pre-measured
resource consumption of each layer.

NetAdapt使用==经验度量==来确定给定资源约束的层中保留的过滤器的数量。我们通过==构建分层查找表==来解决这个问题，并预先测量每一层的资源消耗。

作者先给出了一个图，说明预测的准确性，预测和实际的Latency保持正比。

![1592472748508](D:\Notes\raw_images_2\1592472748508.png)

查表的使用方法：

![1592472891151](D:\Notes\raw_images_2\1592472891151.png)



### 实验结果

![1592478110552](D:\Notes\raw_images_2\1592478110552.png)

![1592478078972](D:\Notes\raw_images_2\1592478078972.png)



**缺点分析：**

1. 计算量大，对于大模型可能难以分析使用。
2. 基于建表查找的方式，不够准确。
3. 迭代次数多，裁剪时间可能过长。

