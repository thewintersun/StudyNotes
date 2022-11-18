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

- DNAS使用==基于梯度的方法==，不同于之前NAS流派的基于RL的方法。
- DNAS==优化的是结构分布==，不同于之前是直接寻找的结构。
- DNAS的==Loss由交叉熵和延时两部分组成==。
- 使用速查表去计算延时, ==延时是硬件相关的==, 之前用FLOPs度量速度。
- 延时==关于层结构是可微的==。

 FBNets, a family of models discovered by DNAS surpass state-of-the-art models both designed manually and generated automatically.

FBNets-B在ImageNet上top-1准确率为74.1%，295M FLOPs，在三星S8上23.1ms延迟，比MobileNetv2小2.4倍，快1.5倍。

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

In summary, ==our overall search space contains 22 layers and each layer can choose from 9 candidate blocks from Table 2, so it contains $9^{22} \approx 10^{21} $ possible architectures==.

latency lookup table model (延时速查表模型)

![1567239729915](D:\Notes\raw_images\1567239729915.png)

过记录几百个在搜索空间中用到的操作的延时，我们能够简单地估算出整个搜索空间中全部 $10^{21}$个结构的运行时间。

##### Loss函数

![1567239343225](D:\Notes\raw_images\1567239343225.png)

##### 搜索算法

During the inference of the super net, only one candidate block is sampled and executed with the
sampling probability of:

![1594111275288](D:\Notes\raw_images\1594111275288.png)

$\theta_l$ contains parameters that determine the sampling probability of each block at layer-l. Equivalently, the output of layer-l can be expressed as:

![1594111396269](D:\Notes\raw_images\1594111396269.png)

where $m_{l,i}$ is a random variable in ${0 - 1}$ and is evaluated to 1 if block $b_{l,i}$  is sampled. 否则为0，

![b_{l,i}(x_l)](https://math.jianshu.com/math?formula=b_%7Bl%2Ci%7D(x_l))表示第![l](https://math.jianshu.com/math?formula=l)层的第![i](https://math.jianshu.com/math?formula=i)个Block的输出（给定输入特征图![x_l](https://math.jianshu.com/math?formula=x_l)），每层独立采样，因此结构可以定义为：
$$
P_{\theta}(a)=\prod_{l}P_{\theta_{l}}(b_l=b_{l,i}^{(a)})
$$
![\theta](https://math.jianshu.com/math?formula=%5Ctheta)是由![\theta_{l,i}](https://math.jianshu.com/math?formula=%5Ctheta_%7Bl%2Ci%7D)组成的向量（第![l](https://math.jianshu.com/math?formula=l)层的第![i](https://math.jianshu.com/math?formula=i)个Block），![b_{l,i}^{(a)}](https://math.jianshu.com/math?formula=b_%7Bl%2Ci%7D%5E%7B(a)%7D)表示在采样的结构![a](https://math.jianshu.com/math?formula=a)中，第![l](https://math.jianshu.com/math?formula=l)层选择第![i](https://math.jianshu.com/math?formula=i)个Block。重写优化问题的表达形式：

![1594115216871](D:\Notes\raw_images\1594115216871.png)

最小化损失在结构分布上的期望

其中的损失函数对于权重![w_a](https://math.jianshu.com/math?formula=w_a)明显可微，可以采用SGD解决，但是该损失对于抽样参数![\theta](https://math.jianshu.com/math?formula=%5Ctheta)并不是直接可微的，因为梯度不能由变量![m_{l,i} ](https://math.jianshu.com/math?formula=m_%7Bl%2Ci%7D%20)传递至![\theta_{l,i}](https://math.jianshu.com/math?formula=%5Ctheta_%7Bl%2Ci%7D)，为了避开这个问题，将离散随机变量![m_{l,i}](https://math.jianshu.com/math?formula=m_%7Bl%2Ci%7D)扩展为：

![1594115238397](D:\Notes\raw_images\1594115238397.png)

其中，$g_{l,i} \sim  Gumbel(0,1) $, 是一个服从Gumbel分布的随机噪音。Gumbel Softmax函数由温度参数 $\tau$ 控制，当其接近0时，![m_{l,i}](https://math.jianshu.com/math?formula=m_%7Bl%2Ci%7D)近似服从
$$
P_{\theta}(a)=\prod_{l}P_{\theta_{l}}(b_l=b_{l,i}^{(a)})
$$
所决定的离散类别抽样，当温度参数变大时，$m_{l,i}$ 变成一个连续随机变量，无论其怎么变化，![m_{l,i}](https://math.jianshu.com/math?formula=m_%7Bl%2Ci%7D)对![\theta_{l,i} ](https://math.jianshu.com/math?formula=%5Ctheta_%7Bl%2Ci%7D%20)都是直接可微的。对于延时损失，使用速查表进行效率估计，因此：
$$
LAT(a)=\sum_{l}LAT(b_{l}^{(a)} )
$$
写作
$$
LAT(a)=\sum_{l}\sum_{i}m_{l,i}LAT(b_{l,i} )
$$
每个操作![LAT(b_{l,i})](https://math.jianshu.com/math?formula=LAT(b_%7Bl%2Ci%7D))的延时是常数，因此整体延时对![m_{l,i},\theta_{l,i}](https://math.jianshu.com/math?formula=m_%7Bl%2Ci%7D%2C%5Ctheta_%7Bl%2Ci%7D)都是可微的，因此损失函数对于权重![w_a](https://math.jianshu.com/math?formula=w_a)和结构分布参数![\theta](https://math.jianshu.com/math?formula=%5Ctheta)都是可微的，因此可以使用SGD解决。

搜索过程等价于去训练随机 supernet。训练过程中，计算：损失对于权重的偏导数，去训练supernet中每个操作f 的权重，这部分与传统的卷积神经网络没有分别，操作搜索训练完城后，不同的操作使得整体网络的精度和速度不同，因此计算损失对于网络结构分布的偏导数，去更新每个操作的抽样概率 $P_{\theta}$。这一步会使得网络整体的精度和速度更高，因为抑制了低精度和低速度的选择，supernet训练完成之后，可以从结构分布 $P_{\theta}$中抽样获得最优的结构。

#### 实验结果

![1567239879071](D:\Notes\raw_images\1567239879071.png)

![1567240004182](D:\Notes\raw_images\1567240004182.png)