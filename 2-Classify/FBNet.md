## FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search

论文地址： https://arxiv.org/abs/1812.03443 

作者：Bichen Wu,  and Kurt Keutzer

机构： UC Berkeley, Princeton University，FaceBook

参考文章： https://blog.csdn.net/soulmeetliang/article/details/94763582

发表：CVPR2019



#### 摘要

为Mobile设备设计一个精准且高效的ConvNets是很困难的，因为模型搜索空间很大，因此先前的neural architecture search (NAS) 是非常昂贵的。

ConvNets架构依赖两个因素，输入大小和目标设备。

之前的架构搜索方法有两个缺点：

- 一个一个Case By Case分离训练模型；
- FLOPs指标不够可靠。

新方法DNAS (differentiable network architecture search) ：

https://openreview.net/pdf?id=BJ-MRKkwG

- DNAS使用==基于梯度的方法==，不同于之前NAS流派的基于RL的方法。
- DNAS==优化的是结构分布==，不同于之前是直接寻找的结构。
- DNAS的==Loss由交叉熵和延时两部分组成==。
- 使用速查表去计算延时, ==延时是硬件相关的==, 之前用FLOPs度量速度。
- 延时==关于层结构是可微的==。

 FBNets, a family of models discovered by DNAS surpass state-of-the-art models both designed manually and generated automatically.



#### 介绍

![1567237839123](D:\Notes\raw_images\1567237839123.png)

Figure 1. Differentiable neural architecture search (DNAS) for ConvNet design. 

- DNAS explores a layer-wise space that each layer of a ConvNet can choose a different block. 
- The search space is represented by a stochastic super net. （通过一个超级网络表达搜索空间 ）
- The search process trains the stochastic super net using SGD to optimize the architecture distribution. （基于梯度的优化方法，比如SGD直接训练结构分布）
- Optimal architectures are sampled from the trained distribution. 
- The latency of each operator is measured on target devices and used to compute the loss for the super net.（训练stochastic super net 的 loss 由改善准确率的交叉熵loss和减少目标设备时延的延时Loss组成）

##### 搜素空间

we ==construct a layer-wise search space with a fixed macro-architecture==, and each layer can choose a different block.

固定的宏观架构，构造了一个layer wise的搜索空间，定义了层数和每层的输入输出维度。

![1567238684954](D:\Notes\raw_images\1567238684954.png)

![1567238730016](D:\Notes\raw_images\1567238730016.png)

​					==每层都可以选不同的Block==

![1567238876561](D:\Notes\raw_images\1567238876561.png)

In summary, our overall search space contains 22 layers and each layer can choose from 9 candidate blocks from Table 2, so it contains $9^{22} \approx 10^{21} $ possible architectures.

latency lookup table model (延时速查表模型)

![1567239729915](D:\Notes\raw_images\1567239729915.png)

过记录几百个在搜索空间中用到的操作的延时，我们能够简单地估算出整个搜索空间中全部 $10^{21}$个结构的运行时间。

##### Loss函数

![1567239343225](D:\Notes\raw_images\1567239343225.png)

##### 搜索算法

![img](D:\Notes\raw_images\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NvdWxtZWV0bGlhbmc=,size_16,color_FFFFFF,t_70)

#### 实验结果

![1567239879071](D:\Notes\raw_images\1567239879071.png)

![1567240004182](D:\Notes\raw_images\1567240004182.png)