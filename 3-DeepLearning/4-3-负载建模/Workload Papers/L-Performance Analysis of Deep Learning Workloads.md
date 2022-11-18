## Performance Analysis of deep learning workloads on leading-edge Systems

论文地址：https://arxiv.org/abs/1905.08764

作者：Yihui Ren, Shinjae Yoo, Adolfy Hoisie

机构：Computational Science Initiative Brookhaven National Laboratory



### 摘要

本文研究了用于机器学习计算的领先前沿系统的性能, 包括 NVIDIA DGX-2、Amazon Web Services(AWS)P3、IBM Power System 加速计算服务器AC922 和一个消费者级Exxact TensorEX TS4 GPU服务器。

从计算机视觉和自然语言处理领域的代表性深入学习工作负载是分析的重点。性能分析与许多重要维度一起执行。考虑了通信互连和大、高吞吐量深度学习模型的性能。

研究了独立和在云中的系统的不同潜在使用模型。

分析了各种优化的深度学习模型和系统配置的效果。

Index Terms-Deep learning, 高性能计算, 基准测试, 性能分析, 计算机架构, 并发计算, DGX-2, GPU





### IV. PERFORMANCE ANALYSIS

This section details the performance analysis of DL workloads using the four systems (already described) under consideration.

本节详细介绍使用正在考虑的四个系统(已经描述)对DL工作负载进行性能分析。

The all-important communication performance is first presented. Given the different workload characteristics, the analysis is done separately for large-scale and high-throughput models. Performance details for an increasingly popular code NCCL supports collective communication primitives, such as all-reduce, all-gather, reduce-scatter, reduce, and broadcast.

==首先介绍了最重要的通信性能==。考虑到不同的工作负载特征，对大规模和高吞吐量模型分别进行分析。对于日益流行的代码，NCCL支持的通信原语集合，如all-reduce, all-gather, reduce-scatter, reduce, and broadcast。

#### A. Communication Performance

NCCL支持集合通信原语，如all-reduce、all-gather、reduce-scatter、reduce和broadcast。

As the most relevant communication kernels occurring in the benchmarks considered, ==all-reduce and broadcast== are examined for performance using NVIDIA’s NCCL-tests code. 

Results are presented normalized to the ”bus bandwidth,” a concept described by NVIDIA in the NCCL-tests. Bus bandwidth is obtained by applying a normalization divider of the measured bandwidth (“message size”/time) different for each communication kernel to reflect its communication complexity and topological mapping to the network. 

Because the bus bandwidth reflects how optimally the hardware is used, it provides a consistent and normalized way to compare the results with the theoretical peak bandwidth, including across different communication primitives.

由于==总线带宽==反映了硬件的最优性,它提供了一种一致的、标准化的方法来将结果与理论峰值带宽进行比较, 包括跨不同的通信原语。

In this work, ==data size varies from 1 MB to 1 GB==, which covers the communication needs for synchronizing model parameters. Each data point is ==averaged over 500 iterations==, except for the case of 16 GPUs using two AWS P3s, which is averaged over 50 iterations due to the slow inter-node Ethernet connection. Figure 2 illustrates the results:

![image-20200908164826520](D:\Notes\raw_images\image-20200908164826520.png)

![image-20200908164912843](D:\Notes\raw_images\image-20200908164912843.png)

#### B. Performance of Deep Learning Workloads

Computation performance is measured in terms of the model training throughput: the average number of training samples, or instances, the system can process per second.

计算性能是根据==模型训练吞吐量==来度量的:系统每秒可以处理的训练样本或实例的平均数量。

![image-20200909105421579](D:\Notes\raw_images\image-20200909105421579.png)

#### **C. Performance of Mixed-Precision Training**

Mixed-precision training [31] retains most if not all neural network predictive performance, yet offers significant computational speedup and reduces the memory footprint.

混合精度训练[31]保留大多数,如果不是所有的神经网络预测性能,还提供显著的计算速度和减少内存足迹。

The NVIDIA Turing GPU architecture, such as V100 and RTX 2080 Ti, provides dedicated hardware acceleration called “tensor cores” [32] for this purpose. 

NVIDIA Turing GPU架构,如V100和RTX 2080 Ti,为这个目的提供了专用的硬件加速称为“张量核心”[32]。

The tensor core provides high-throughput fused multiply-add (FMA) operations for mixed-precision matrices (inputs in half precision, outputs in either half or single precision). The other advantage of using mixed-precision is the smaller memory footprint, therefore less communication overhead for synchronizing the model replicas.

张量核心为混合精度矩阵==提供 high-throughput fused multiply-add (FMA) 操作(输入半精度,输出一半或单精度)==。使用混合精度的另一个优点是==内存空间较小, 因此通信开销较少,以同步模型副本==。

Figure 8a shows the performance of ResNet50 on DGX-2 when using mixed-precision (FP16) for batch size (bsz) 128 and 256, comparing it to the performance when using single precision (FP32) for the same model in Figure 8b.

图8a显示,在使用混合精度(FP16)用于批量大小(bsz)128和256时,在DGX-2上的性能(FP16),将其与在图8b中相同模型的单个精度(FP32)中进行比较。

![image-20200909110139675](D:\Notes\raw_images\image-20200909110139675.png)

#### D. Comparing the PyTorch On-node Data Parallel with Distributed Data Parallel

model = torch.nn.DataParallel(model)  VS torch.utils.data.DataLoader

The communication pattern of on-node data parallel differs from the distributed data parallel. In it, one GPU maintains a master copy of the model parameters. At every iteration, it broadcasts the parameters to the other GPUs in the configuration. At the end of every iteration, the parameters are “all-reduced” back to the master GPU, which updates the model parameters. Therefore, for each iteration, two global communications (broadcast and reduce) are issued. 

单节点数据并行的通信模式不同于分布式数据并行。在其中,一个GPU维护模型参数的主副本。在每次迭代中,它将参数传递给配置中的另一个gpu。在每次迭代结束时,参数都是“all-reduced”回到主GPU,它更新模型参数。因此,对于每个迭代,发布了两个全球通信(broadcast and reduce)。

To emulate the common practice of most PyTorch models, we use the default PyTorch data loader for on-node data parallel experiments (torch.utils.data.DataLoader), which supports multi-worker and pinned memory but not asynchronous data loading. 

为了模拟大多数PyTorch模型的常见做法,我们使用默认的PyTorch数据加载器进行节点数据并行实验(torch.utils.data.DataLoader), 它支持multi-worker和固定内存,而不是异步数据加载。

PyTorch’s on-node data parallel design maximizes its usefulness but targets small parallel GPU configurations, such as those common in workstations.

PyTorch的单节点数据并行设计最大限度地实现了它的有用性, 但目标是小的并行GPU配置, 比如工作站中常见的配置。

![image-20200909111030315](D:\Notes\raw_images\image-20200909111030315.png)

