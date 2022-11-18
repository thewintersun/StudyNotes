## Roofline Performance Model

网址：https://crd.lbl.gov/departments/computer-science/par/research/roofline/

Roofline论文列表：https://crd.lbl.gov/departments/computer-science/par/research/roofline/publications/

Roofline是塞缪尔·威廉姆斯（Samuel Williams）创建的一种直观直观的性能模型，用于限制在多核，多核或加速器处理器体系结构上运行的各种数值方法和运算的性能。通过简单地将局部性，带宽和不同的并行化范式组合成一个性能指标，该模型可以用来评估获得的性能质量，而不仅仅是简单地使用==峰值估算值==。可以检查最终的Roofline数字，以确定实施和固有性能限制。

- Introduction

  Roofline模型围绕应用程序数据局部性，数据带宽和计算吞吐量之间的相互作用进行定位。这些主题中的每一个都会在此处进一步完善和讨论。算术强度Roofline模型背后的核心参数是算术强度。

- Software

  我们已经创建，协作和利用了许多影响Roofline建模的工具。从广义上讲，这些工具实现了机器表征，应用程序检测，缓存模拟和瓶颈识别的子集。此外，这些工具可以牺牲性能开销来获得准确性，从而使用户可以从粗略的高级表征入手，并进行关键的详细分析。

  - [Empirical Roofline Tool (ERT)](https://crd.lbl.gov/departments/computer-science/par/research/roofline/software/ert/)
  - [Roofline Visualizer](https://crd.lbl.gov/departments/computer-science/par/research/roofline/software/roofline-visualizer/)
  - [Intel Advisor](https://docs.nersc.gov/programming/performance-debugging-tools/advisor/)
  - [NVIDIA NVProf / NSight](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/)
  - [LIKWID](https://docs.nersc.gov/programming/performance-debugging-tools/likwid/)
  - [SDE/VTune](https://docs.nersc.gov/performance/arithmetic_intensity/)

### Introduction

Roofline模型围绕==应用程序数据局部性==，数据带宽和计算吞吐量之间的相互作用进行定位。这些主题中的每一个都会在此处进一步完善和讨论。

#### Arithmetic Intensity

Roofline模型背后的核心参数是算术强度。算术强度是总浮点运算与总数据移动量（字节）的比率。 BLAS-1矢量-矢量增量（x [i] + = y [i]）具有非常低的算术强度，为0.0417（N FLOPS/24N字节），并且与向量大小无关。相反，FFT执行5\*N\*logN触发器以进行N点双复数变换。如果在写分配高速缓存体系结构上不合适，则转换将至少移动48N字节。这样，FFT的算术强度为0.104 * logN，并且随着数据大小的增长而缓慢增长。不幸的是，缓存容量会将FFT运算强度限制为2 flops per byte。最后，BLAS3和N-Body粒子-粒子方法将具有非常快的算术强度增长。

![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflineai.png)

#### Roofline Model

最基本的Roofline模型可用于将浮点性能限制为机器峰值性能，机器峰值带宽和算术强度的函数。
可以通过将性能界限（GFlop / s）绘制为算术强度的函数来可视化Roofline模型。生成的曲线可以视为存在内核或应用程序性能的性能范围。

#### Effects of NUMA on Memory Bandwidth

现代SMP和GPU加速系统将呈现不统一的内存访问。根据数据的位置和线程的位置，内存带宽可能会发生巨大变化。下面的示例重点介绍了OpenMP中数据的初次接触分配对带宽的影响。实际上，对于几乎所有可达到的算术强度，所得的较低带宽都会降低性能。


![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflinebw.png)

#### Effects of Cache Behavior on Arithmetic Intensity

Roofline模型需要估计总体数据移动。在基于缓存的体系结构上，3C的缓存模型强调了一个事实，即不仅仅是强制数据移动。缓存容量和conflict misses可以增加数据移动并降低算术强度。同样，多余的缓存写分配可能导致数据移动增加一倍。向量初始化操作x [i] = 0.0要求每写入一条高速缓存行分配一次写分配和一次写回。写分配是多余的，因为该高速缓存行的所有元素都将被覆盖。不幸的是，硬件流预取器的存在使很难量化超出实际发生的强制数据移动量。  

![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflinecache.png)

####  Instruction-Level Parallelism and Performance

现代处理器体系结构已深入流水线化。尽管这可以提高频率和峰值性能，但确实会增加各种指令的等待时间。为了避免管道中出现bubbles并获得最佳性能，程序员和编译器必须进行协作，并确保按顺序发出独立的指令（指令级并行性）。缺少ILP可能会降低足够计算密集型内核的性能。相反，在内存密集型操作中，缺少ILP可能不会影响性能。下面的示例在一个人为的求和示例中强调了这一点，其中部分sums被构造并在循环结束时求和。

![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflineilp.png)

#### Data-Level Parallelism and Performance

数据级并行性（矢量化，SIMD化等）已成为使性能和能源效率最大化的一种非常有吸引力的方法。不幸的是，可获得的性能在很大程度上取决于编译器或程序员利用这些指令的能力。对于高算术强度，缺乏有效的SIMD化会降低性能。但是，对于低算术强度，对性能的影响可以忽略不计。

![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflinedlp.png)

#### Run Time vs Arithmetic Intensity

可以将模型理解为运行时间与算术强度之间的关系，而不是将Roofline视作性能是算术强度的函数。为此，必须使用数据集大小将所有元素的性能（每个元素）转换为运行时间。在下面的示例中，我们显示运行时间与多项式的次数无关，直到达到机器平衡为止。超过这一点，求解时间应随多项式的次数线性增加。研究人员应将此视为机会。可以更改算法（例如，移至高阶方法并获得更好的错误）而不会影响运行时间。




![img](https://crd.lbl.gov/assets/Uploads/FTG/Projects/Roofline/_resampled/ResizedImage600300-rooflinetime.png)



