## Channel Pruning via Automatic Structure Search

论文地址：https://arxiv.org/abs/2001.08565

作者：Mingbao Lin, Rongrong Ji, Yuxin Zhang, Baochang Zhang, Yongjian Wu, Yonghong Tian

机构：厦门大学，腾讯优图实验室

代码：https://github.com/lmbxmu/ABCPruner



相似文章：MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning

https://arxiv.org/abs/1903.10258v2

https://zhuanlan.zhihu.com/p/64422341



总结：作者根据论文（Rethinking the value of network pruning )认为，剪枝的目的主要在结构搜索。而AMC这样的方法很耗资源，所以作者做了两个改进：

1. 为了解决深度网络中被剪枝结构的大组合问题，我们首先提出将保留通道限制在特定空间内(10%, 20% ... 100% ) 的剪枝结构的组合进行缩小，从而使被剪枝结构的组合显著减少。
2. 然后，将最优剪枝结构的搜索表述为一个优化问题，并集成ABC算法自动求解，以减少人为干扰。

### 摘要

In this paper, we propose a new channel pruning method based on ==artificial bee colony algorithm== (ABC), dubbed as ABCPruner, which aims to efficiently ==find optimal pruned structure, i.e., channel number in each layer, rather than selecting "important" channels as previous works did==.  

To solve the intractably huge combinations of pruned structure for deep networks, we first propose to shrink the combinations where the preserved channels are limited to a specific space, thus the combinations of pruned structure can be significantly reduced. And then, we formulate the search of optimal pruned structure as an optimization problem and integrate the ABC algorithm to solve it in an automatic manner to lessen human interference. 

ABCPruner has been demonstrated to be more effective, which also enables the fine-tuning to be conducted efficiently in an end-to-end manner. 

Experiments on CIFAR-10 show that ABCPruner reduces 73.68% of FLOPs and 88.68% of parameters with even 0.06% accuracy improvement for VGGNet-16. 

On ILSVRC-2012, it achieves a reduction of 62.87% FLOPs and removes 60.01% of parameters with negligible accuracy cost for ResNet-152.



人工蜂群算法的介绍

​    人工蜂群算法(Artificial Bee Colony, ABC)是由Karaboga于2005年提出的一种新颖的基于群智能的全局优化算法，其直观背景来源于蜂群的采蜜行为，蜜蜂根据各自的分工进行不同的活动，并实现蜂群信息的共享和交流，从而找到问题的最优解。人工蜂群算法属于==群智能算法==的一种。



Channel pruning via artificial bee colony (ABC) in an automatic manner：![img](https://raw.githubusercontent.com/zyxxmu/Images/master/ABCPruner/ABCPruner_framework.png)

(a) A structure set is initialized first, elements of which represent the preserved channel number. 

(b) The filters of the pre-trained model are randomly assigned to each structure. We train it for given epochs to measure its fitness.

(c) Then, the ABC algorithm is introduced to update the structure set and the fitness is recalculated through (b). (b) and (c) will continue for some cycles. 

(d) The optimal pruned structure with best fitness is picked up, and the trained weights are reserved as a warm-up for fine-tuning the pruned network. (Best viewed with zooming in)



### 实验结果

![1583829352510](D:\Notes\raw_images\1583829352510.png)



![1583829390027](D:\Notes\raw_images\1583829390027.png)

