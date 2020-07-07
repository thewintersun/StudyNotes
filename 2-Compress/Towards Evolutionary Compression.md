## Towards Evolutional Compression

论文地址：https://arxiv.org/abs/1707.08005

作者：Yunhe Wang, Chang Xu, Jiayan Qiu, Chao Xu, Dacheng Tao

机构：北京大学

发表：KDD 2018



总结: 

1. 会生成大量的模型，最后种子的选择问题，Flops和ACC的权衡。

2. 依赖验证集来进行种子选择。

3. 进化次数，每次的量等超参需要人为确定。

4. Flops为理论推理，即（conv, bn, linear等操作数），与硬件无关。

5. 训练过程中需要：generation * population * epoches 次数的training，计算量很大。

   

依赖软件包：

项目地址：https://github.com/DEAP/deap

文档地址: https://deap.readthedocs.io/en/master/

DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. It seeks to make algorithms explicit and data structures transparent. It works in perfect harmony with parallelization mechanism such as multiprocessing and SCOOP. The following documentation presents the key concepts and many features to build your own evolutions.



### 摘要

Compressing convolutional neural networks (CNNs) is essential for transferring the success of CNNs to a wide variety of applications to mobile devices. In contrast to directly recognizing subtle weights or filters as redundant in a given CNN, this paper presents an evolutionary method to automatically eliminate redundant convolution filters. We represent each compressed network as a binary individual of specific fitness. Then, the population is upgraded at each evolutionary iteration using genetic operations. As a result, an extremely compact CNN is generated using the fittest individual. In this approach, either large or small convolution filters can be redundant, and filters in the compressed network are more distinct. In addition, since the number of filters in each convolutional layer is reduced, the number of filter channels and the size of feature maps are also decreased, naturally improving both the compression and speed-up ratios. Experiments on benchmark deep CNN models suggest the superiority of the proposed algorithm over the state-of-the-art compression methods.



![1583842873855](D:\Notes\raw_images\1583842873855.png)

![1583842951211](D:\Notes\raw_images\1583842951211.png)

### 实验结果

![1583843062551](D:\Notes\raw_images\1583843062551.png)

