## Modeling GPU-CPU Workloads and Systems 

论文地址：https://dl.acm.org/doi/10.1145/1735688.1735696

发表：Publication:GPGPU-3: Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing UnitsMarch 2010 Pages 31–42https://doi.org/10.1145/1735688.1735696

机构：Georgia Institute of Technology Atlanta, GA

### 摘要

We believe that a comprehensive analysis and modeling framework is necessary to ease application development and automate program optimization on heterogeneous platforms.

我们相信，一个全面的分析和建模框架对于简化异构平台上的应用程序开发和自动化程序优化是必要的。

本文报告了==4个gpu==和==3个cpu==的==25个CUDA应用程序的经验评估==, 利用Ocelot动态编译器基础设施,可以在任何目标上执行相同的CUDA应用程序。 使用instrumentation和统计分析的组合，我们为每个应用程序记录==37个不同的指标==，并使用它们在异构处理器上推导==程序行为和性能之间==的关系。

然后，这些关系被输入一个建模框架，该框架试图预测在不同处理器上类似类别（classes ）的应用程序的性能。最重要的是，本研究确定了程序特性之间的几个非直观的关系，并证明了使用内核执行前可用的指标来==精确模拟CUDA内核性能==是可能的。

### 介绍

**Ocelot Framework**

本文提出的分析和模型==利用了Ocelot框架==来检测数据并行应用程序并在异构平台上执行它们。==Ocelot框架[13]是一个仿真和编译基础设施==，实现CUDA运行时API 

(1) emulates executing kernels, (2) translates kernels to the CPU ISA, or (3) emits the kernel representation to the CUDA driver for execution on attached GPUs.

(1)模拟正在执行的内核，(2)将内核转换为CPU ISA，或(3)将内核表示发送给CUDA驱动程序，以便在附加的gpu上执行。

Consequently, in addition to enabling detailed and comparative workload characterizations, the infrastructures enables transparently portability of PTX kernels across CPUs and NVIDIA GPUs.

Ocelot的结果是与特定的GPU实现解耦的，除了内存效率度量，并提供了对应用行为的洞察。

https://docs.microsoft.com/en-us/dotnet/architecture/microservices/multi-container-microservice-net-applications/implement-api-gateways-with-ocelot

https://github.com/ThreeMammals/Ocelot

**PTX**

NVIDIA’s Parallel Thread eXecution (PTX) [16] is a virtual instruction set architecture with explicit data-parallel execution semantics that are well-suited to NVIDIA’s GPUs.

NVIDIA的并行线程执行 (PTX)[16]是一个具有显式数据并行执行语义的==虚拟指令集架构==，非常适合NVIDIA的GPUs

PTX 由一组RISC-like的指令组成，这些指令用于明确类型的算术计算、装载和存储到一组地址空间、并行和同步指令以及内置变量。

Functions implemented in PTX, known as kernels are intended to be executed by a large grid of threads arranged hierarchically into ==a grid of cooperative thread arrays== (CTAs).

在PTX中实现的函数，也就是所谓的内核，是要由一个大的线程网格执行的，这个线程网格分层排列成一个协作线程阵列(CTAs)网格。

![1605083942888](D:\Notes\raw_images\1605083942888.png)

Threads within a CTA are assumed to be executing on a single processor and may synchronize at programmer-inserted barriers within the kernel.

CTA中的线程均在单个处理器上执行，并且可以在内核中程序员插入的barrier上同步。

PTX执行模型允许cta的==序列化执行==，以避免状态的压倒性大爆炸。

**Ocelot Infrastructure**

The Ocelot compiler infrastructure strives to decouple CUDA applications from GPUs by wrapping CUDA Runtime API [17], ==parsing kernels stored as bytecode== within the application into an internal representation, executing these kernels on devices present, and maintaining a complete list of CUDA memory allocations that store state on the GPU.

Ocelot编译器基础设施, 力求解耦CUDA GPU的应用程序, 通过包装CUDA运行时API[17],解析内核作为==字节码==存储在应用程序中为一个内部表示, 在设备上执行这些内核, 维护完整列表, CUDA GPU内存分配存储状态。

通过将CUDA应用程序与CUDA驱动程序解耦，Ocelot提供了一个模拟GPU、收集性能指标、将内核转换为GPU以外的架构、检测内核以及优化内核以便在GPU上执行的框架。

### TRANSLATION

**LLVM**
The Low Level Virtual Machine (LLVM) [14] is a maturing compiler infrastructure that maintains a strongly-typed program representation of that program throughout its lifetime.

LLVM[14]是一个成熟的编译器基础设施,在其一生中维护该程序的一种严格类型的程序表示。

这是内核的LLVM表示，可以由一个主机线程执行CTA中的每个线程，并正确地执行内核。

![1605085441280](D:\Notes\raw_images\1605085441280.png)

**Execution Model Translation**

在多核cpu上编译PTX内核需要首先将内核转换为所需的指令集,然后将内核的执行语义从PTX的线程层次结构转换为单个主机线程。

Ocelot可以通过在CTA中启动一个内核级别的线程,并依靠os级支持障碍和多线程来提供并发性和上下文切换, 可以使用LLVM翻译来执行PTX内核。

The multicore execution model translation technique described here ==assumes warp size is equal to 1 thread==, so applications must be ==recompiled with synchronization points== following those statements expected to be executed simultaneously on architectures with a larger warp size.

在这里描述的多核执行模型翻译技术假设,经过的翻译技术等于1个线程,因此必须重新编译应用程序,在这些语句遵循这些语句后,期望在具有更大的经转尺寸的架构上同时执行。

**CTA Runtime Support**

当应用程序启动内核时，多线程运行时层除了每个线程的上下文数据结构外，还会启动尽可能多的工作线程，因为每个线程都有可用的硬件线程。这个上下文由共享内存块、用于寄存器溢出的本地内存块和特殊寄存器组成。然后工作线程遍历内核网格的块，每个块作为CTA执行。执行模型允许CTAs的任意排序和到并发工作线程的任意映射。

### Metrics and Statistics

14个Metrics

**Activity Factor.** Any given instruction is executed by all threads in a warp. However, individual threads can be predicated off via explicit predicate registers or as a result of branch divergence. Activity factor is the fraction of ==threads active averaged over all dynamic instructions==.

活动的因素。任何给定的指令都是由所有线程一起执行的。但是，单个线程可以通过显式谓词寄存器或由于分支发散而被断言。活动因子是==所有动态指令中活动线程的平均比例==。

**Branch Divergence.** When a warp reaches a branch instruction, all threads may branch or fall through, or the warp may diverge in which the warp is split with some threads falling through and other threads branching. ==Branch Divergence is the fraction of branches that result in divergence averaged over all dynamic branch instructions==.

分支散度。当经经线到达分支指令时,所有的线程都可以分叉或向下,或者warp可能会分叉,在其中,warp 被分裂,一些线程被落,其他的线程也会分支。

**Instruction Counts.** These metric ==count the number of dynamic instructions== binned according to the functional unit that would execute them on a hypothetical GPU. The functional units considered here include integer arithmetic, floating-point arithmetic, logical operations, control-flow, off-chip loads and stores, parallelism and synchronizations, special and transcendental, and data type conversions.

指令数。这些度量数的动态指令的数量，根据功能单元，将执行它们在一个假设的GPU。这里考虑的功能单元包括整型算术、浮点算术、逻辑操作、控制流、芯片外负载和存储、并行性和同步、特殊和超越以及数据类型转换。

**Inter-thread Data Flow.** The PTX execution model includes synchronization instructions and shared data storage accessible by threads of the same CTA. Interthread data flow measures ==the fraction of loads from shared memory such that the data loaded was computed by another thread within the CTA==. This is a measure of producer-consumer relationships among threads.

Inter-thread数据流。PTX执行模型包括同步指令和共享数据存储，可由相同CTA的线程访问。线程间数据流==测量来自共享内存的负载的比例==，这样加载的数据由CTA中的另一个线程计算。==这是线程之间生产者-消费者关系的度量==。

**Memory Intensity.** Memory intensity ==computes the fraction of instructions resulting in communication to offchip memory==. These may be explicit loads or stores to global or local memory, or they may be texture sampling instructions. This metric does not model the texture caches which are present in most GPUs and counts texture samples as loads to global memory.

记忆的强度。内存强度计算导致通信到芯片存储器的指令的分数。这些可能是显式加载或存储到全局或本地内存，或者它们可能是纹理采样指令。这个度量不建模texture缓存，这是存在于大多数gpu和计数texture 样本作为加载到全局内存。

**Memory Efficiency.** Loads and stores to global memory may reference arbitrary locations. However, if threads of the same warp access locations in the same block of memory, the operation may be completed in a single memory transaction; otherwise, transactions are serialized. This metric expresses the minimum number of transactions needed to satisfy every dynamic load or store divided by the actual number of transactions, computed according to the memory coalescing protocol defined in [17] §5.1.2.1. This is a measure of spatial locality.

记忆效率。加载和存储到全局内存可以引用任意位置。但是，如果具有相同warp的线程位于同一内存块中，则该操作可以在单个内存事务中完成; 否则，事务被序列化。这个指标表示==满足每个动态负载或存储所需的最小事务数除以实际事务数==，根据[17]§5.1.2.1中定义的内存合并协议计算。这是==空间局部性的度量==。

**Memory Extent.** This metric uses pointer analysis to compute the working set of kernels as the number and layout of all reachable pages in all memory spaces. It represents the total amount of memory that is accessible to a kernel immediately before it is executed.

记忆程度。这个度量使用指针分析来计算内核的工作集，作为所有内存空间中所有可达页面的数量和布局。==它表示内核在执行之前可访问的内存总量==。

**Context Switch Points.** CTAs may synchronize threads at the start and end of kernels as well as within sections of code with uniform control flow, typically to ensure shared memory is consistent when sharing data. Each synchronization requires a context switch point inserted by Ocelot during translation for execution on multicore as described in[6].

上下文切换点。CTAs 可以在内核的开始和结束以及使用统一控制流的代码段内同步线程，通常是为了确保共享内存在共享数据时是一致的。如[6]所述，每次同步都需要Ocelot在转换期间插入一个上下文切换点，以便在多核上执行。

**Live Registers.** Unlike CPUs, GPUs are equipped with large register files that may store tens of live values per thread. Consequently, executing CTAs on a multicore x86 CPU requires spilling values at context switches. This metric expresses the average number of spilled values.

Live Registers。与cpu不同，gpu配备了大的寄存器文件，可以为每个线程存储数十个活动值。因此，在多核x86 CPU上执行CTAs需要在上下文切换时溢出值。这个度量==表示溢出值的平均数量==。

**Machine Parameters**. GPUs and CPUs considered here are characterized by clock rate, number of concurrent threads, number of cores, off-chip bandwidth, number of memory controllers, instruction issue width, L2 cache capacity, whether they are capable of executing out-of-order, and the maximum number of threads within a warp.

机器参数。这里考虑的gpu和cpu的特征是==时钟速率、并发线程数量、核心数量、芯片外带宽、内存控制器数量、指令发出宽度、L2缓存容量、它们是否能够执行无序，以及一个warp内的最大线程数量==。

**Registers per Thread.** The large register files of GPUs may be partitioned into threads at runtime according to the number of threads per CTA. Larger numbers of threads increases the ability to hide latencies but reduces the number of registers available per thread. On CPUs, these may be spilled to local memory. This metric expresses the average number of registers allocated per thread.

每个线程Registers 。根据CTA的线程数,在运行时,gpu的大寄存器文件可以被分区到线程中。越来越多的线程增加了隐藏延迟的能力,但减少了每线程可用的寄存器数量。在cpu上, 这些可能会溢出到本地内存中。这个度量==表示每个线程分配的平均寄存器数量==。

**Kernel Count.** The number of times an application launches a kernel indicates the number of global barriers across all CTAs required.

内核数。应用程序==启动内核的次数表示需要跨越所有CTAs 的全局障碍的数量==。

**Parallelism Scalability.** This metric determines the maximum amount of SIMD and MIMD parallelism[13] available in a particular application averaged across all kernels.

并行性可伸缩性。这个指标确定了特定应用程序中==在所有内核中可用的SIMD和MIMD并行度==[13]的最大值。

**DMA Transfer Size.** CUDA applications explicitly copy buffers of data to and from GPU memory before kernels may be called incurring a latency and bandwidth constrained transfer via the PCI Express bus of the given platform.We measure both==the number of DMAs== and the ==total amount of data transferred==.

DMA传输的大小。CUDA应用程序明确拷贝缓冲区的数据到和从GPU内存之前，内核被called 可能导致了一个延迟和带宽受限的传输, 通过给定平台的PCI Express总线。我们测量DMAs的数量和传输的数据总量。

### **BenchMark**

![1605162220838](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1605162220838.png)



![1605162262045](D:\Notes\raw_images\1605162262045.png)

**Metrics List (37个)  :**

在内核执行之前，可以通过静态分析（static  analysis）来确定一些数量，例如每个内核的静态指令计数，在内核启动之前启动的DMA操作的数量/大小，以及由保守指针分析确定的工作集大小的上界。

其他的可能是在运行时通过在内核中插入（instrumentation）检测并在执行时记录平均值来确定的;这些包括SIMD和MIMD并行性度量。

最后，一些指标—通常是动态指令计数—可能只能通过PTX仿真（emulation）执行内核完成并分析结果指令跟踪来确定。

| Metric           | Units        | Description         | Collection   | method    |
| ---------------- | ------------ | ------------------- | ------------ | --------- |
| Extent_of_Memory | bytes        | Size of working set | static       | analysis  |
| Context_switches| switch points| Number of thread context switch points |static |analysis|
|Live_Registers | registers| Number registers spilled at context switch points| static| analysis|
|Registers_Per_Thread | registers | Number of registers per thread |static |analysis|
|DMAs |transfers| ==Number of transfers between GPU memory== |static |analysis|
|Static_Integer_arithmetic |instructions| Number of integer arithmetic instructions |static| analysis|
|Static_Integer_logical| instructions| Number of logical instructions |static |analysis|
|Static_Integer_comparison |instructions| Number of integer compare instructions |static |analysis|
|Static_Memory_offchip |instructions| Number of off-chip memory transfer instructions| static |analysis|
|Static_Memory_onchip |instructions| Number of on-chip memory transfer instructions |static |analysis|
|Static_Control |instructions| Number of control-flow instructions| static |analysis|
|Static_Parallelism |instructions |Number of parallelism instructions |static |analysis|
|Dynamic_Integer_arithmetic |instructions| Number of executed integer arithmetic s | instructions|emulation|
|Dynamic_Integer_logical |instructions| Number of executed integer logical| instructions |emulation|
|Dynamic_Memory_offchip |instructions| Number ==executed off-chip memory transfer== |instructions |emulation|
|Dynamic_Memory_onchip |instructions| Number of executed on-chip memory transer |instructions |emulation|
|Dynamic_Integer_comparison |instructions| Number of executed integer comparison |instructions| emulation|
|Static_Float_single |instructions| Single-precision floating point arithmetic |static |analysis|
|Static_Float_comparison |instructions| Single-precision floating point compare |static |analysis|
|Static_Special |instructions| Special function instructions| static |analysis|
|Memory_Efficiency |percentage |Memory efficiency| metric |instrumentation|
|Memory_Sharing |percentage| ==Inter-thread data flow== |metric| instrumentation|
|Activity_Factor |percentage| Activity factor |metric| instrumentation|
|MIMD |speedup| MIMD Parallelism |metric |instrumentation|
|SIMD |speedup| SIMD Parallelism |metric |instrumentation|
|Dynamic_Float_single |instructions| Number of single-precision arithmetic |instructions |emulation|
|Dynamic_Float_comparison |instructions| Number of singl-precision comparison| instructions |emulation|
|DMA_Size| bytes| Avg DMA transfer size| static |analysis|
|Dynamic_Control |instructions| Number of executed control-flow |instructions| emulation|
|Dynamic_Parallelism |instructions| Number of executed parallelism |instructions| emulation|
|Dynamic_Special |instructions| Number of executed spcial |instructions |emulation|
|Static_Float_double |instructions |Number of double precision floating point |instructions| static |
|Memory_Intensity |instructions| Memory Intensity |metric |instrumentation|
|Dynamic_Float_double| instructions| Number of executed double-precision floating point |instructions |emulation |
| Dynamic_Other    | instructions | Other               | instructions | emulation |



![1605163173764](D:\Notes\raw_images\1605163173764.png)

### RESULTS

**Machine Principal Components**

**PC0: Single Core Performance.** The variables that contribute strongly to the first principal component are
shown in the left of Figure 3. Note that all of these metrics, clock frequency, issue width, cache size, etc correspond to the performance of a single processor core. Additionally, note that threads-per-core and warp size are negatively correlated with clock frequency, issue width, and out of order, highlighting the differences between GPU and CPU design philosophies.

此外，请注意，每核线程和warp大小与时钟频率、问题宽度和无序呈负相关，突出了GPU和CPU设计理念之间的差异。

![1605172253684](D:\Notes\raw_images\1605172253684.png)

**PC1: Core and Memory Controller Count.** The second PC illustrates that ==the core count is correlated with the memory controller count and memory bandwidth per channel==, indicating that multi-core CPUs and GPUs are designed such that the off-chip bandwidth scales with the number of cores.

第二PC演示了核数与每通道的内存控制器数和内存带宽相关，这表明多核cpu和gpu的设计使片外带宽随核数的变化而变化。

![1605167679286](D:\Notes\raw_images\1605167679286.png)

**Application Components**

**PC0: MIMD Parallelism**

**PC1: Problem Size.** The second component is composed most significantly of average dynamic integer, floating point, and memory instruction counts which collectively describe the number of instructions executed in each kernel.

**PC2: Data Dependencies.** We believe that the second principal component exposed the most significant and nonobvious relationship in this study. It indicates that data dependencies are likely to be propagated throughout all levels of the programming model; if there is a large degree of data sharing between instructions, then there is likely to be a large degree of data sharing among threads in each CTA and among all CTAs in a program.

PC2:数据依赖项。我们认为,第二个主要组成部分暴露了这项研究中最显著和不明显的关系。它表明,在编程模型的所有级别中都可能传播数据依赖项;如果指令之间有很大程度的数据共享,那么在每个CTA中,在所有的CTA中,在一个程序中,都可能有很大程度的数据共享。

**PC3: Memory Intensity.** The next principal component is composed almost entirely of metrics that are associated with the memory behavior of a program.

**PC4: Control Flow Uniformity/SIMD Parallelism.**

