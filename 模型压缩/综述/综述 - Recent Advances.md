## Recent Advances in Efficient Computation of Deep Convolutional Neural Networks

论文地址：https://arxiv.org/abs/1802.00939

作者：Jian Cheng, Peisong Wang, Gang Li, Qinghao Hu, Hanqing Lu

机构：中科院



### 摘要

In this paper, we provide a comprehensive survey of recent advances in network acceleration, compression and accelerator design from both algorithm and hardware points of view. 

Specifically, we provide a thorough analysis of each of the following topics: network pruning, low-rank approximation, network quantization, teacher-student networks, compact network design and hardware accelerators. Finally, we will introduce and discuss a few possible future directions.

 

### 网络剪枝

网络剪枝基于这样的假设,在深网络中的许多参数都是不重要或不必要的, 修剪方法用来删除不重要的参数。通过这种方法,修剪方法可以显著扩展参数的稀疏性。剪枝后参数的高度稀疏, 对深神经网络有两种好处。一方面, 修剪后的稀疏参数需要较少的磁盘存储, 因为参数可以存储在压缩稀疏行格式(compressed sparse row, CSR)或压缩稀疏列(compressed sparse column, CSC)格式中。另一方面,省略了这些修剪参数的计算, 从而降低了深网络的计算复杂度。 根据修剪的粒度, 修剪方法可以分为五组: 细粒修剪、向量级修剪、内核修剪、组级修剪 和 filter级修剪, fine-grained pruning, vector-level pruning, kernel-level pruning, group-level pruning and filter-level pruning. 

![1583152039623](D:\Notes\raw_images\1583152039623.png)

#### fine-grained pruning

细粒度的修剪方法或香草修剪方法以一种unstructered的方式移除参数, 在卷积内核中任何不重要的参数都可以修剪.

![1583152075954](D:\Notes\raw_images\1583152075954.png)





### 低秩逼近



#### 二元分解



#### 三元分解



#### 四元分解



### 网络量化

标量和向量量化

固定点量化



### 老师学生网络



### 紧凑网络设计



### 硬件加速



### 未来趋势讨论

