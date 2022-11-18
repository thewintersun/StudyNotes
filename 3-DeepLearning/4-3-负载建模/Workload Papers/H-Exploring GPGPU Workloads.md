## Exploring GPGPU Workloads: Characterization Methodology, Analysis and Microarchitecture Evaluation Implications

论文地址：https://ieeexplore.ieee.org/document/5649549

机构：Intelligent Design of Efficient Architecture Lab (IDEAL)

时间：**Date Added to IEEE Xplore:** 03 December 2010

发表： [IEEE International Symposium on Workload Characterization (IISWC'10)](https://ieeexplore.ieee.org/xpl/conhome/5644749/proceeding)



### 摘要

Abstract—  In this research, we propose a set of microarchitecture agnostic GPGPU workload characteristics to represent them in a microarchitecture independent space.

Correlated ==dimensionality reduction process== and ==clustering analysis== are used to understand these workloads. In addition, we propose a set of evaluation metrics to accurately evaluate the GPGPU design space. With growing number of GPGPU workloads, this approach of analysis provides meaningful, accurate and thorough simulation for a proposed GPU architecture design choice. Architects also benefit by choosing a set of workloads to stress their intended functional block of the GPU microarchitecture. We present a diversity analysis of GPU benchmark suites such as Nvidia CUDA SDK, Parboil and Rodinia.

研究了相关的降维过程和聚类分析,以了解这些工作负载。此外,我们还==提出了一套评估指标,以准确评价GPGPU设计空间==。随着GPGPU工作负载的增加, 这种分析方法为==提出的GPU架构设计选择提供了有意义、准确和彻底的仿真==。架构师还可以通过选择一组工作负载来强调他们的GPU微架构的功能块。我们对Nvidia CUDA SDK、parfields和Rodinia等GPU基准套件进行了多样性分析。

Our results show that with a large number of diverse kernels, workloads such as ==Similarity Score==, ==Parallel Reduction==, and ==Scan of Large Arrays== show diverse characteristics in different workload spaces. We have also explored diversity in different workload subspaces (e.g. memory coalescing and branch divergence). Similarity Score, Scan of Large Arrays, MUMmerGPU, Hybrid Sort, and Nearest Neighbor workloads exhibit relatively large variation in branch divergence characteristics compared to others. Memory coalescing behavior is diverse in Scan of Large Arrays, K-Means, Similarity Score and Parallel Reduction.

我们的研究结果表明,在不同的工作负载空间中,有大量的不同的内核,工作量如相似度、并行还原和大阵列的扫描显示了不同的特征。我们还探索了不同工作负载子空间的多样性(例如内存合并和分支散度)。相似性分数、大阵列扫描、MUMmerGPU、混合排序和最近的邻居工作负载比其他工作表现出相对较大的分支散度特性变化。记忆合并行为在大阵列、k -均值、相似度和并行还原等方面是多样化的。



### 介绍

为了实现上述目标，我们提出了==一套GPU微架构无关的GPGPU工作负载特性==，以准确捕捉工作负载行为，并使用广泛的指标来评估表征的有效性。

1. 我们提出了一套GPGPU工作负载表征指标。使用38×6设计点，我们表明这些指标独立于底层GPU微架构。这些指标将允许GPGPU研究人员评估新兴GPU微架构的性能，而不管它们的微架构是否有所改进。
2. 利用提出的GPGPU工作负载指标，我们研究了现有==GPGPU内核之间的相似性==，并观察到它们经常强调相同的瓶颈。我们证明消除冗余可以显著节省仿真时间。
3. 我们提供了==基于不同工作负载子空间的工作负载分类==，如发散特性、内核特性、内存合并等。我们根据它们的重要性对不同的工作负载特征进行分类。我们还表明，可用工作负载空间在==分支散度特征==方面差异最大，而在==线程批处理级合并==行为方面差异最小。

![1605173778505](D:\Notes\raw_images\1605173778505.png)

![1605173799310](D:\Notes\raw_images\1605173799310.png)

![1605173833816](D:\Notes\raw_images\1605173833816.png)

### EXPERIMENTAL METHODOLOGY

In this study, we used ==GPGPU-Sim== [6], a cycle accurate ==PTX-ISA simulator==.

不同的Config

![1605180058046](D:\Notes\raw_images\1605180058046.png)

采用的Benchmark见论文

采用的评价：

![1605180179470](D:\Notes\raw_images\1605180179470.png)

Activity factor [24] is defined as the average number of active threads at a given time during the execution phase. Several branch divergence related characteristics of the benchmark change this parameter. For example, the absence of branch divergence produces an activity factor of 100%. 

活动因子[24]定义为执行阶段==某一给定时间的活动线程的平均数量==。基准的几个分支发散相关的特性改变了这个参数。例如，分支差异的缺失产生了100%的活动因子。

SIMD parallelism [24] captures the scalability of a workload. Higher value for SIMD parallelism indicates that the workload performance will improve on a GPU that have higher SIMD width. 

SIMD并行性[24]捕获==工作负载的可伸缩性==。SIMD并行度值越高，表明在SIMD宽度越大的GPU上工作负载性能越好。

DRAM efficiency [6] describes how frequently memory accesses are requested during kernel execution. If a benchmark has large number of shared memory accesses or the benchmark has ALU operations properly balanced in between memory operations, then the metric will show higher value.

DRAM效率[6]描述了在==内核计算期间请求内存访问的频率==。它还捕获了在整个内核执行期间执行DRAM内存传输所花费的时间。如果一个基准测试有大量的共享内存访问，或者基准测试在内存操作之间有适当平衡的ALU操作，那么这个指标将显示更高的值。

