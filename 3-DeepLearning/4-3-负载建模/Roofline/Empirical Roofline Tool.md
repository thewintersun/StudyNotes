## Empirical Roofline Tool (ERT)

网址：https://crd.lbl.gov/departments/computer-science/par/research/roofline/software/ert/

代码库: https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/

使用手册：https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/Empirical_Roofline_Tool-1.1.0/ERT_Users_Manual.pdf



### 介绍

经验屋顶线工具ERT自动为给定计算机生成屋顶线数据。这包括==内存层次结构各个级别的最大带宽==和==最大Gflop速率==。该数据是使用各种“micro-kernels”获得的。

ERT带有一组用于许多计算机/体系结构的配置文件。这些配置文件可以适合您的本地环境，并且需要更好地测量计算机的Roofline参数。

经验Roofline工具（ERT）凭经验确定为屋顶线模型生成机器特征所需的机器特征（CPU或GPU加速）。这些特性是存储器层次结构中每个级别的带宽和最大Gflop速率。需要此工具有以下几个原因：

- 估计屋顶线分析所需的机器特性非常耗时，而且非常困难，即使不是不可能。
- 即使可以估算出机器特性，也只是理论上的最大值，可能没有代码可以实现这些最大值。
- ==理论上的最大值无法指导开发人员确定什么代码可实现最高性能，需要哪种类型的并行性，使用哪些编译器，使用哪些选项以及如何最佳地运行代码==。

以下是在NERSC的Cori（单核68核Knights Landing manycore处理器）上运行ERT的示例。注意，ERT始终将存储器的第一级标记为L1，将最后一级标记为DRAM。这样，在带有numactl-m1的四边形中，ERT将MCDRAM存储器标记为“ DRAM”。应该记住，KNL MCDRAM带宽取决于阵列大小，群集模式（象限quadrant，SNC2，SNC4）以及MCDRAM是高速缓存还是内存。

已观察到更大的问题大小可获得更高的带宽（超过450GB / s），而 -qopt-streaming-stores =never 发现在四级缓存模式下是有益的。

![ert cori](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600464-ert-cori.png)

同样，我们可以为CUDA编译ERT，并在ORNL的SummitDev（4个NVIDIA P100 GPU）上运行它。由于ERT使用读-修改-写 内核，因此名义上将其缓存在GPU L2上（L1不可见）。但是，ERT将它看到的第一个缓存（GPU L2）标记为L1。此外，与KNL一样，它将检测到的最后一级内存（P100上的HBM）标记为DRAM。这样，借助4个GPU，每个SummitDev节点的总持续HBM带宽总计接近2TB / s。

![ert summitdev 4gpus](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600464-ert-summitdev-4gpus.png)

