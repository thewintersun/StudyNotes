## Kernel Profiling Guide

网址：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

Regular Application Execution

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/regular-application-process.png)

Profiled Application Execution

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/profiled-process.png)

## 2. Metric Collection

性能指标的收集是NVIDIA Nsight Compute的关键特性。由于有大量的度量指标, 所以通常使用一些更容易的预先定义的集合工具来收集一个常用的子集。用户可以自由地根据需要调整收集哪些指标, 但重要的是要记住==与数据收集相关的开销==。

### 2.1. Sets and Sections

NVIDIA Nsight Compute使用Section Sets(short sets)在一个非常高的级别上决定要收集的度量值的数量。每个集合包括一个或多个部分，每个部分指定几个逻辑关联的指标。例如，一个部分可能只包括高级SM和内存利用率指标，而另一个部分可能包括与内存单元或HW调度器相关的指标。

### 2.2. Sections and Rules

每个模块是一个section文件:

![1611809646607](D:\Notes\raw_images\1611809646607.png)

| Identifier and Filename                      | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| ComputeWorkloadAnalysis (计算负载分析)       | Streaming Multiprocessors (SM) 计算资源的详细分析，包括实现的每个时钟指令(instructions per clock, IPC) 和 每个可用管道的利用率。具有很高利用率的管道可能会限制整体性能。 |
| InstructionStats                 (指令统计)  | Statistics of the executed low-level assembly instructions (SASS). 指令组合观察了==执行指令的类型和频率==。==A narrow mix of instruction types意味着依赖于少数指令管道，而其他指令管道仍未使用==。==使用多个管道可以隐藏延迟并支持并行执行==。 |
| LaunchStats                       (启动统计) | 用于启动内核的配置摘要。==启动配置定义了内核网格的大小、将网格划分为块以及执行内核所需的GPU资源==。选择有效的启动配置可以最大化设备利用率。 |
| MemoryWorkloadAnalysis  (内存负载分析)       | GPU内存资源的详细分析。当充分利用涉及的硬件单元(Mem Busy)、耗尽这些单元之间的可用通信带宽 (Max Bandwidth) 或达到 issuing内存指令的最大吞吐量(Mem Pipes Busy)时，内存可能会成为总体内核性能的限制因素。根据限制因素，内存图和表允许识别内存系统中的确切瓶颈。 |
| Nvlink                                       | NVLink利用率的高级摘要。它显示了总的接收和传输(发送)内存，以及总的链路峰值利用率。 |
| Occupancy                         (占用率)   | ==占用率是每个多处理器的active warps数与可能的最大active warps数的比率==。理解占用率的另一种方式是the percentage of the hardware's ability to process warps that is actively in use。==更高的占用率并不总是会带来更高的性能，但是，低占用率总是会降低隐藏延迟的能力==，从而导致总体性能下降。在执行过程中，==理论占用率和实际占用率之间的巨大差异通常表明工作负载高度不平衡==。 |
| SchedulerStats                  (调度统计)   | Summary of the activity of the schedulers issuing instructions.  活动摘要的调度程序发出指令。==每个调度器都维护一个可以发出指令的warps池==。池中warps的上限（Theoretical Warps）受到启动配置的限制。在每个周期中，==每个调度器检查池中已分配的warps (Active Warps)==的状态。未停止的活跃warps(Eligible Warps) 准备好发出下一条指令。==从Eligible Warps集合中，调度程序选择一个Warp，从中发出一个或多个指令（Issued Warp）==。在没有Eligible Warp的周期中，==跳过 issue slot，不发出指令==。有许多跳过的issue slot表明延迟隐藏较差。 |
| SourceCounters                (源计数器)     | 源指标，包括分支效率和sampled warp stall 原因。在内核运行时定期采样数据度量。它们显示了warps什么时候停了，什么时候不能scheduled。有关所有warps stall原因的描述，请参阅文档。Only focus on stalls if the schedulers fail to issue every cycle. |
| SpeedOfLight                     (概况)      | GPU计算资源和内存资源的利用率概述。对于每个单元，Speed Of Light (SOL)报告了相对于理论最大值的利用率的实现百分比。在Volta+ GPUs上，它将SOL SM和SOL Memory的故障breakdown报告给每个单独的sub-metric，以清楚地识别最高的贡献者。 |
| WarpStateStats           (Warp状态统计)      | 分析所有warps在内核执行过程中spent周期的状态。==Warp状态描述了warp能够或不能发出下一条指令==。每条指令的warp周期定义了两个连续指令之间的延迟。==这个值越高，隐藏这个延迟所需的Warp并行性就越大==。对于每种Warp状态，图表显示了每个发出指令在该状态中花费的平均周期数。Stalls并不总是影响整体性能，也不是完全可以避免的。如果调度程序不能发出每个周期，只关注Stalls暂停原因。 |

### 2.3. Kernel Replay

根据如何收集内核启动的指标, 内核可能需要重新Reply一个或多个次, 因为不是所有的指标都可以在单个pass中收集。为了解决这个问题,在NVIDIA Nsight Compute 中要求一个特定的内核实例的所有指标都被分组到一个或多个passes中。对于第一个传递, 可以通过内核访问所有的GPU内存。在第一次通过之后, 由内核编写的内存子集确定。在每个传递(除了第一个)之前, 这个子集在其原始位置恢复, 使内核在每个重播传递中访问相同的内存内容。

Regular Application Execution

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/replay-regular-execution.png)

通过内核重播执行。将保存所有内存，并在重播过程之间恢复内核写入的内存。

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/replay-kernel.png)

### 2.4. Application Replay

根据为内核启动收集的指标，内核可能需要重放一次或多次，因为不是所有指标都可以在一次传递中收集到。例如，GPU可以同时收集的来自硬件(HW)性能计数器的指标数量是有限的。此外，基于补丁的软件(SW)性能计数器对内核运行时有很大的影响，并且会影响HW计数器的结果。

对于正确的匹配和结合从单个内核启动的多个应用程序重放传递收集的性能计数器, 应用程序需要对其内核活动和它们的分配到GPUs、上下文、流和潜在的NVTX ranges。通常情况下, 这也意味着应用程序需要对其总体执行进行确定。

应用程序重放的好处是，==内核访问的内存不需要通过工具保存和恢复，因为每个内核启动在应用程序进程的生命周期中只执行一次==。除了避免内存保存和恢复开销外，应用程序重放还允许禁用缓存控制。如果应用程序在特定内核启动之前使用其他GPU活动来将缓存设置为某种预期状态，那么这一点特别有用。

此外,应用重播还可以支持在执行过程中对主机有相互依赖的内核。内核回放, 这类内核通常在被破坏时挂起, 因为宿主的必要响应在所有情况下都丢失了, 但第一次通过。相反, 应用程序Replay确保每个传递程序执行程序执行的正确行为。与内核重播相比,通过应用程序重放收集的多个传递意味着应用程序的所有host-side活动都是重复的。如果应用程序需要重要的时间, 例如设置或file-system访问,那么开销就会相应地增加。

Regular Application Execution

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/replay-regular-execution.png)

通过应用程序重放执行。没有内存被保存或还原，但是运行应用程序本身的成本是重复的。

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/replay-application.png)

Kernel matching during application replay using the grid strategy.

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/replay-application-kernel-matching.png)

### 2.5. Profile Series

内核的性能高度依赖于所使用的启动参数。对启动参数的微小更改可能会对内核的运行时行为产生显著影响。然而，通过手工测试大量的组合来确定内核的最佳参数集可能是一个乏味的过程。

为了使这个工作流程更快、更方便，概要文件系列==提供了使用不断变化的参数自动多次概要文件单个内核的能力==。需要修改的参数和需要测试的值可以独立启用和配置。对于所选参数值的每个组合，都会收集一个唯一的配置文件结果。并在对一系列结果的描述中跟踪修改后的参数值。通过比较一个剖面序列的结果，可以看出核函数在参数变化时的行为，从而快速识别出最优的参数集。

### 2.6. Overhead

与大多数度量一样,使用NVIDIA Nsight Compute CLI 收集性能数据在应用程序上遇到了一些运行时开销。开销确实取决于许多不同的因素:

- Number and type of collected metrics
- The collected section set
- Number of collected sections
- Number of profiled kernels
- GPU Architecture

## 3. Metrics Guide

### 3.1. Hardware Model

![1611811175674](D:\Notes\raw_images\1611811175674.png)

![img](https://cinwell.files.wordpress.com/2013/09/gpu-hier.png)

GPU由许多流式多处理器（SM）组成，每个SM通常具有8到32的SIMT宽度（Fermi系列具有SIMT宽度为32的16个SM，而AMD的ATI 5870 Evergreen有SIMT宽度为16的20个SM） 。每个SM都与一个专用的L1数据高速缓存，只读纹理和恒定高速缓存以及一个低延迟的共享内存（暂存器）相关联。每个MC都与共享L2缓存的一部分相关联，以便更快地访问缓存的数据。 MC和L2均在芯片上。

![1611811273028](D:\Notes\raw_images\1611811273028.png)

### Compute Model

所有的NVIDIA GPUs 都被设计成支持通用的异构并行编程模型，通常称为Compute。该模型将GPU从传统图形管道中解耦出来，并将其公开(exposes)为通用并行多处理器。异构计算模型指存在主机(host) 和 设备(device)，即CPU和GPU。在高层视图中，主机(CPU)管理自身和设备之间的资源，并将工作发送到 设备(device) 以并行执行。

计算模型的中心是==网格（Grid）、块（Block）、线程层次结构（Thread hierarchy）==, 它定义了在GPU上组织计算工作的方式。从上到下的层次结构如下:

- A Grid is a 1D, 2D or 3D array of thread blocks.
- A Block is a 1D, 2D or 3D array of threads, also known as a Cooperative Thread Array (CTA).
- A Thread is a single thread which runs on one of the GPU's SM units.

网格，块，线程层次结构的目的是在一组线程，即协作线程数组（Cooperative Thread Array，CTA），中公开局部性的概念。在CUDA中，CTA被称为线程块（Thread Blocks）。该体系结构可以==通过在单个CTA中的线程之间提供快速共享的内存和屏障来利用这种局域性==。启动网格时，该==体系结构可确保CTA中的所有线程将在同一SM上同时运行==。有关网格和块的信息，请参见“启动统计信息”部分。

每个SM上适合的CTA数量取决于CTA所需的物理资源。这些资源限制器包括线程和寄存器的数量，共享内存的利用率以及硬件障碍。==每个SM的CTA数量称为CTA占用率==，这些物理资源限制了该占用率。  

每个CTA都可以安排在任何可用的SM上，但不能保证其执行顺序。==因此，CTA必须完全独立，这意味着一个CTA不可能等待另一个CTA的结果==。由于CTA是独立的，因此主机（CPU）可以启动一个大型网格，该网格无法一次全部适用于硬件，但是任何GPU仍然可以运行它并产生正确的结果。

==CTA进一步分为称为Warps的32个线程组==。如果CTA中的线程数不能除以32，则最后一个Warp将包含剩余的线程数。

可以==在给定GPU上同时运行的CTA总数称为Wave==。因此，Wave的大小与GPU可用SM的数量成比例，但也与内核的占用率成比例。

![img](https://cinwell.files.wordpress.com/2013/09/thread-hier.png)

### Streaming Multiprocessor （流媒体多处理器）

![1611811338491](D:\Notes\raw_images\1611811338491.png)

流媒体多处理器 (Streaming Multiprocessor，SM) 是GPU的核心处理单元。SM对大量的工作负载进行了优化, 包括通用计算、深度学习、射线跟踪、照明（lighting ）和遮阳（shading）。==SM的设计目的是同时执行多个CTAs。CTAs可以来自不同的启动的网格==。

SM实现了一个称为单指令多线程（Single Instruction Multiple Threads，SIMT）的执行模型，==该模型允许单个线程具有唯一的控制流，同时仍作为Warp的一部分执行==。 Turing SM继承了Volta SM的独立线程调度模型。 SM维护每个线程的执行状态，包括程序计数器（PC）和调用堆栈。独立的线程调度允许GPU执行任何线程的执行，以更好地利用执行资源，或者允许线程等待另一个线程在同一Warp中可能产生的数据。收集Source Counters源计数器部分使您可以检查Source Page上的指令执行和predication details以及Sampling information采样信息。

每个SM被划分为四个处理块, 称为SM子分区。SM子分区是SM的主要处理元素。==每个子分区包含以下单元==: 

- Warp Scheduler
- Register File
- Execution Units/Pipelines/Cores
  - Integer Execution units
  - Floating Point Execution units
  - Memory Load/Store units
  - Special Function unit
  - Tensor Cores

在四个SM分区的SM中共享:

- Unified L1 Data Cache / Shared Memory
- Texture units
- RT Cores, if available

A warp 被分配给子分区，并且从启动到完成驻留在子分区中。映射到子分区时，warp 称为活动 (active )的或常驻(resident ) 的。子分区管理一个固定大小的warps池。在Volta架构中，池的大小是16 warps。在图灵架构中，池的大小是8个warps。激活(Active) 的warp可以处于合格(eligible) 状态，如果warp准备发出指令。这需要warp有一个解码(decoded )的指令，所有的输入依赖项都被解析，并且函数单元是可用的。关于活动的、合格的和发出（issuing ）的warp的统计信息可以通过Scheduler Statistics部分收集。

A warp is  

- an instruction fetch,
- a memory dependency (result of memory instruction),
- an execution dependency (result of previous instruction), or
- a synchronization barrier.

![1611811420940](D:\Notes\raw_images\1611811420940.png)

See    

编译器控制下的最重要的资源是内核使用的寄存器的数量。每个子分区有一组32位寄存器, 由HW在固定大小块中分配。启动统计部分显示了内核的register用法。

### Memory

**全局内存：**==是一个49位虚拟地址空间，它映射到设备上的物理内存==，固定的系统内存或对等内存（ peer memory）。==全局内存对GPU中的所有线程可见。通过SM L1和GPU L2访问全局内存==。

**本地内存：**==是执行线程的专用存储，在该线程之外不可见==。它用于线程本地数据，例如线程堆栈和寄存器溢出。 AGU单元将本地内存地址转换为全局虚拟地址。本地内存与全局内存具有相同的延迟。全局内存和本地内存之间的一个区别是，==本地内存的排列方式使得连续32位字可以通过连续线程ID进行访问==。因此，==只要warp中的所有线程都访问相同的相对地址（例如，数组变量中的相同索引，结构变量中的相同成员等），访问就会完全合并==。

**共享内存: **位于芯片上，因此==它比本地或全局内存有更高的带宽和更低的延迟==。==共享内存可以跨计算CTA共享==。试图通过共享内存跨线程共享数据的计算CTA==必须在store和load之间使用同步操作==(例如__syncthreads())，以确保任何一个线程写入的数据对CTA中的其他线程可见。类似地，需要通过全局内存共享数据的线程必须使用更重量级的全局内存barrier。

共享内存具有32个存储体，这些存储体经过组织，因此==连续的32位字映射到可以同时访问的连续存储体==。因此，可以同时处理由32个地址组成的32位存储器中的任何32位存储器读取或写入请求，这些请求位于单个存储器的带宽的32倍之内。

对于Warp的共享内存请求不会在访问相同32位字中的任何地址的两个线程之间产生存储体冲突（即使两个地址都位于同一存储体中）。==当多个线程进行相同的读取访问时，一个线程接收数据，然后将其广播到其他线程==。当多个线程写到同一位置时，只有一个线程成功写入；否则，写操作失败。哪个线程成功是不确定的。

### Caches

所有GPU单元都通过2级缓存（也称为L2）与主内存进行通信。 L2缓存位于片上内存客户端和framebuffer之间。 L2在物理地址空间中工作。除了提供缓存功能外，==L2还包括执行压缩和全局原子atomics的硬件==。

L1级数据高速缓存（L1）在处理全局，本地，共享，纹理和表面内存的读写操作以及归约和原子操作中起着关键作用。在Volta和Turing架构上，==每个TPC有两个L1缓存，每个SM都有一个==。有关L1如何适合纹理管道的更多信息，请参见TEX单元说明。

还要注意，尽管本节经常使用名称“ L1”，但是应该理解，L1数据高速缓存，共享数据和纹理数据高速缓存是相同的。

L1接收来自两个单元的请求：SM和TEX。 L1从SM接收全局和本地存储请求，并从TEX接收texture和surface请求。这些操作访问全局内存空间中的内存，L1通过二级缓存L2发送该内存。

高速缓存的命中率和未命中率以及数据传输在“内存工作量分析”部分中报告。

### Texture/Surface

TEX单元执行Texture获取和过滤。除了普通的texture存储器访问，TEX还负责将texture读取请求转换为结果所需的寻址，LOD，换行，过滤和格式转换操作。

TEX通过其输入接口从SM接收两种一般的请求类别：texture请求和surface加载/存储操作。texture和surface内存空间驻留在设备内存中，并缓存在L1中。texture和surface存储器被分配为 block-linear surfaces（例如2D，2D阵列，3D）。这样的surfaces提供了数据的高速缓存友好布局，从而2D表面上的相邻点在内存中的位置也彼此靠近，从而提高了访问位置。在访问内存之前，通过TEX单元对surfaces访问进行边界检查，该内存可用于实现不同的texture包裹模式。

L1缓存针对2D空间局部性进行了优化，因此读取2D空间中紧密靠近的texture或surface地址的同一Warp线程将获得最佳性能。 L1高速缓存还设计用于以恒定延迟进行流式获取。高速缓存命中可以减少DRAM带宽需求，但不能减少获取延迟。通过texture或surface存储器读取设备存储器具有一些优势，可以使其成为从全局或常量存储器中读取存储器的有利替代方案。

### 3.2. Metrics Structure

### Metrics Overview

对于为使用SM 7.0及更高版本的GPU收集的指标，NVIDIA Nsight Compute使用高级指标计算系统，旨在帮助您确定发生了什么（计数和指标），以及程序达到峰值GPU性能的距离（以百分比表示的吞吐量） 。每个计数器在数据库中都有关联的峰值速率，以允许按百分比计算其吞吐量。

吞吐量指标返回其组成计数器的最大百分比值。这些成分已经过精心选择，以代表GPU管线中控制峰值性能的部分。虽然所有计数器都可以转换为峰值百分比，但并非所有计数器都适合峰值性能分析。不合适的计数器的示例包括活动的合格子集和工作负载驻留计数器。使用吞吐量指标可确保进行有意义且可行的分析。

每个计数器都有两种类型的峰值速率：突发和持续(burst and sustained)。==突发速率是单个时钟周期内可报告的最大速率==。对于“典型”操作，==持续速率是在无限长的测量周期内可达到的最大速率==。对于许多柜台来说，爆发等于持续。==由于不能超过突发速率，因此突发速率的百分比将始终小于100％==。在边缘情况下，持续发生率的百分比有时可能会超过100％。

### Metrics Entities

在NVIDIA Nsight Compute中，所有性能计数器均称为指标，可以将它们进一步划分为具有特定属性的组。对于通过PerfWorks测量库收集的度量，**存在四种类型的度量实体：**

指标：这些是计算量。每个指标都有以下内置的子指标：

| Name                             | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `.peak_burst`                    | the peak burst rate （单个时钟周期内）                       |
| `.peak_sustained`                | the peak sustained rate （在无限长的测量周期内）             |
| `.per_cycle_active`              | the number of operations per unit active cycle               |
| `.per_cycle_elapsed`             | the number of operations per unit elapsed cycle              |
| `.per_cycle_in_region`           | the number of operations per user-specified range cycle      |
| `.per_cycle_in_frame`            | the number of operations per user-specified frame cycle      |
| `.per_second`                    | the number of operations per second                          |
| `.pct_of_peak_burst_active`      | % of peak burst rate achieved during unit active cycles      |
| `.pct_of_peak_burst_elapsed`     | % of peak burst rate achieved during unit elapsed cycles     |
| `.pct_of_peak_burst_region`      | % of peak burst rate achieved over a user-specified range time |
| `.pct_of_peak_burst_frame`       | % of peak burst rate achieved over a user-specified frame time |
| `.pct_of_peak_sustained_active`  | % of peak sustained rate achieved during unit active cycles  |
| `.pct_of_peak_sustained_elapsed` | % of peak sustained rate achieved during unit elapsed cycles |
| `.pct_of_peak_sustained_region`  | % of peak sustained rate achieved over a user-specified range time |
| `.pct_of_peak_sustained_frame`   | % of peak sustained rate achieved over a user-specified frame time |

计数器：可以是来自GPU的原始计数器，也可以是计算出的计数器值。每个计数器下面都有4个子指标，也称为：

比率：每个计数器下面都有2个子指标：

|          | 说明                                 |
| -------- | ------------------------------------ |
| `.pct`   | The value expressed as a percentage. |
| `.ratio` | The value expressed as a ratio.      |

吞吐量：一系列百分比指标，指示GPU的一部分与峰值速率的接近程度。每个吞吐量都有以下子指标：  

| Name                             | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `.pct_of_peak_burst_active`      | % of peak burst rate achieved during unit active cycles      |
| `.pct_of_peak_burst_elapsed`     | % of peak burst rate achieved during unit elapsed cycles     |
| `.pct_of_peak_burst_region`      | % of peak burst rate achieved over a user-specified "range" time |
| `.pct_of_peak_burst_frame`       | % of peak burst rate achieved over a user-specified "frame" time |
| `.pct_of_peak_sustained_active`  | % of peak sustained rate achieved during unit active cycles  |
| `.pct_of_peak_sustained_elapsed` | % of peak sustained rate achieved during unit elapsed cycles |
| `.pct_of_peak_sustained_region`  | % of peak sustained rate achieved over a user-specified "range" time |
| `.pct_of_peak_sustained_frame`   | % of peak sustained rate achieved over a user-specified "frame" time |

除了PerfWorks指标外，

- 设备属性：device\_\_attribute_ * 指标表示CUDA设备属性。收集它们不需要额外的内核重播，因为可从CUDA驱动程序获得每个CUDA设备的值。
- 启动指标：launch__* 指标是在每次内核启动时收集的，不需要额外的重播。它们可以作为内核启动参数的一部分（例如网格大小，块大小等）使用，也可以使用CUDA占用率计算器进行计算。

### Metrics Examples

```python
## non-metric names -- *not* directly evaluable
sm__inst_executed                   # counter
smsp__average_warp_latency          # ratio
sm__throughput                      # throughput

## a counter's four sub-metrics -- all evaluable
sm__inst_executed.sum               # metric
sm__inst_executed.avg               # metric
sm__inst_executed.min               # metric
sm__inst_executed.max               # metric

## all names below are metrics -- all evaluable
l1tex__data_bank_conflicts_pipe_lsu.sum
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_burst
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_active
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_elapsed
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_region
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_frame
l1tex__data_bank_conflicts_pipe_lsu.sum.per_second
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_burst_active
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_burst_elapsed
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_burst_region
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_burst_frame
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_active
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_region
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_frame           
```

### 指标命名约定

计数器和指标通常遵循命名方案：

- Unit-Level Counter : `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)`
- Interface Counter : `unit__(subunit?)_(pipestage?)_(interface)_quantity_(qualifiers?)`
- Unit Metric : `(counter_name).(rollup_metric)`
- Sub-Metric : `(counter_name).(rollup_metric).(submetric)`

where

- unit:  GPU的逻辑或物理单位
- subunit:  要测量计数器的单元中的子单元。有时，这是一种管道模式。
- pipestage: 测量计数器的子单元内的管道Stage。
- quantity: 正在测量的内容。通常匹配尺寸单位。
- qualifiers: 应用于计数器的任何其他predicates或过滤器。通常，不合格的计数器可以细分为几个合格的子组件。
- interface:  格式为“ sender2receiver”，其中“ sender”是源单位，而“ receiver”是目的单位。
- rollup_metric: 求和，平均值，最小值，最大值。
- submetric: refer to section Metrics Entities

### Cycle Metrics

Counters using the term  

- `unit__cycles_elapsed` : 一个范围内的cycles数。cycles的DimUnit特定于设备的时钟域。 
- `unit__cycles_active` : 设备正在处理数据的cycles数。
- `unit__cycles_stalled` : 由于其输出接口被阻塞，设备无法处理新数据的cycles数。
- `unit__cycles_idle` : 设备空闲的cycles数。

接口级cycle计数器通常（并非总是）具有以下变体形式：

- `unit__(interface)_active` : 数据从源单元传输到目标单元的cycles。
- `unit__(interface)_stalled` : 源单元拥有数据但目标单元无法接受数据的cycles。

### 3.3. Metrics Decoder

以下说明了在NVIDIA Nsight Compute SM 7.0及以上指标名称中找到的术语。

### Units

| Name    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| `dram`  | Device (main) memory 设备（主）内存，GPU全局和本地内存驻留在其中。 |
| `fbpa`  | FrameBuffer分区是位于2级缓存（LTC）和DRAM之间的内存控制器。 FBPA的数量因GPU而异。 |
| `fe`    | Frontend unit 前端单元负责驱动程序发送的总体工作负载。FE还促进了许多同步操作。 |
| `gpc`   | General Processing Cluster 通用处理群集包含TPC形式的SM，Texture和L1。它会在芯片上复制多次。 |
| `gpu`   | The entire Graphics Processing Unit. 整个图形处理单元。      |
| `gr`    | Graphics Engine 图形引擎负责所有2D和3D图形，计算工作以及同步图形复制工作。 |
| `idc`   | InDexed Constant Cache InDexed常量缓存是SM的一个子单元，负责缓存用寄存器索引的常量。 |
| `l1tex` | The Level 1 (L1)/Texture Cache 级别1（L1）/Texture 缓存位于GPC内。它可以用作定向映射的共享内存和/或在其缓存部分中存储全局，局部和纹理数据。 |
| `lts`   | A Level 2 (L2) Cache Slice 二级（L2）缓存片是二级缓存的子分区。 |
| `sm`    | Streaming Multiprocessor 流多处理器将内核的执行作为32个线程的组（称为Warp）进行处理。Warp进一步分为协作线程阵列（CTA），在CUDA中称为块Block。 CTA的所有Warps都在同一SM上执行。 CTAn内线程共享各种资源，例如共享内存。 |
| `smsp`  | SMSP是SM的子分区。                                           |
| `sys`   | 几个单位的逻辑分组                                           |
| `tpc`   | Thread Processing Clusters 线程处理群集是GPC中的单位。它们包含一个或多个SM，Texture 和L1单元，指令缓存（ICC）和索引常量缓存（IDC）。 |

### Subunits

| 子单位            | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| `aperture_device` | Memory interface to ==local device memory (dram)==           |
| `aperture_peer`   | Memory interface to ==remote device memory==                 |
| `aperture_sysmem` | Memory interface to ==system memory==                        |
| `l1`              | Level 1 cache                                                |
| `lsu`             | ==Load/Store unit==                                          |
| `mem_global`      | Global memory                                                |
| `mem_lg`          | Local/Global memory                                          |
| `mem_local`       | Local memory                                                 |
| `mem_shared`      | Shared memory                                                |
| `mem_surface`     | Surface memory                                               |
| `mem_texture`     | Texture memory                                               |
| `mio`             | Memory input/output                                          |
| `mioc`            | Memory input/output control                                  |
| `rf`              | Register file                                                |
| `texin`           | TEXIN                                                        |
| `xbar`            | Crossbar 交叉开关（XBAR）负责将数据包从给定的源设备传输到特定的目标设备。 |

### Pipelines

| 流水线                           | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `ADU`                            | Address Divergence Unit 地址发散单元。 ADU负责分支/跳转的地址差异处理。它还为恒定负载和块级屏障指令提供支持。 |
| `ALU`                            | Arithmetic Logic Unit 算术逻辑单元。 ALU负责执行大多数==位操作和逻辑指令==。它还执行整数指令，==不包括IMAD和IMUL==。在NVIDIA Ampere架构芯片上，ALU管线可实现快速的FP32到FP16转换。 |
| `CBU`                            | Convergence Barrier Unit. 收敛屏障单元。 CBU负责Warp级收敛，屏障Barrier和分支branch指令。 |
| `FMA`                            | 融合乘法相加/累加。 FMA流水线处理大多数FP32算法（FADD，FMUL，FMAD）。它还执行整数乘法运算（IMUL，IMAD）以及整数点积。在GA10x上，FMA是表示FP32和FP16x2峰值性能的逻辑流水线。它由FMAHeavy和FMALite物理管道组成。 |
| `FMA (FP16)`                     | FMA（FP16）表示逻辑FMA流水线内的FP16x2指令执行。它还包含一个快速的FP16至FP32转换器。 |
| `FMALite`                        | FMALite执行FP32算术（FADD，FMUL，FMA）和FP16算术（HADD2，HMUL2，HFMA2）。 |
| `FMAHeavy`                       | FMAHeavy执行FP32算术（FADD，FMUL，FMAD），FP16算术（HADD2，HMUL2，HFMA2）和 整数点积。 |
| `FP16`                           | Half-precision floating-point unit.半精度浮点单元。在Volta，Turing和NVIDIA GA100上，FP16管线执行成对的FP16指令（FP16x2）。它还包含一个快速的FP32至FP16和FP16至FP32转换器。从GA10x芯片开始，此功能是FMA管道的一部分。 |
| `FP64`                           | Double-precision floating-point unit. 双精度浮点单元。 FP64单元负责大多数FP64指令（DADD，DMUL，DMAD等）。 FP64的实现因芯片而异。因此，它的吞吐量可能会显着不同。 |
| `LSU`                            | Load Store Unit. 负载存储单元。 LSU管道向L1TEX单元发出加载，存储，原子和还原指令，以获取全局，本地和共享内存。它还向L1TEX单元发出特殊的寄存器读取（S2R），随机播放和CTA级到达/等待屏障指令。 |
| `Tensor (DP)/Tensor (FP64)`      | 双精度浮点矩阵乘和累加单元。                                 |
| `Tensor (FP)/Tensor (FP16/TF32)` | 混合精度（FP16 / TF32和FP32）浮点矩阵乘和累加单元。          |
| `Tensor (INT)`                   | 整数矩阵乘以累加单元。                                       |
| `TEX`                            | Texture单元。 SM Texture管线将纹理和表面指令转发到L1TEX单元的TEXIN阶段。在FP64或Tensor管线解耦的GPU上，纹理管线也转发这些类型的指令。 |
| `Uniform`                        | Uniform Data Path. 统一数据路径。此标量单元执行所有线程使用相同输入并生成相同输出的指令。 |
| `XU`                             | Transcendental and Data Type Conversion Unit. 超越和数据类型转换单元。 XU管道负责特殊功能，例如sin，cos和倒数平方根。它还负责int到float以及float到int类型的转换。 |

### Quantities

| 数量          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| `Instruction` | 汇编（SASS）指令。每个执行的指令可以生成零个或多个请求。     |
| `Request`     | 进入硬件单元以执行某些动作的命令，例如从某个内存位置加载数据。每个请求访问一个或多个扇区。 |
| `Tag`         | 高速缓存行的唯一键。如果线程地址不是全部都位于单个高速缓存行对齐的区域内，则请求可以查找多个标签。 L1和L2都有128字节的缓存行。标签访问可以归类为命中或未命中。 |
| `Set Access`  | 逻辑上与标签相同。                                           |
| `Sector`      | 高速缓存行或设备内存中对齐的32个字节的内存块。 L1或L2高速缓存行是四个扇区，即128个字节。如果存在标签并且在高速缓存行内存在扇区数据，则将扇区访问分类为命中。标签缺失和标签命中数据缺失均被归类为缺失。 |
| `Wavefronts`  | 在处理阶段结束时为请求生成的唯一“工作包”的数量。波前的所有工作项将并行处理，而不同波前的工作项将在不同的周期进行序列化和处理。每个请求至少产生一个波前。 |



## 7. Roofline Charts

Roofline 图提供了一种非常有用的方法，可以可视化在复杂处理单元（例如GPU）上的性能。本节介绍配置文件报告中显示的Roofline图表。

### 7.1. Overview

内核性能不仅取决于GPU的运行速度。由于内核需要处理数据，因此性能还取决于GPU可以将数据馈送到内核的速率。典型的roofline图将GPU的峰值性能和内存带宽与称为算术强度（工作与内存流量之间的比率）的指标结合在一起，形成一个图表，以更真实地表示已分析内核的性能。一个简单的屋顶线图可能如下所示：

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/roofline-overview.png)

该图表实际上显示了两个不同的屋顶线。但是，可以为每个组件标识以下组件：

- **Vertical Axis** -  垂直轴表示每秒的浮点操作（FLOPS）。对于GPU，此数字可能会变得非常大，因此可以缩放此轴上的数字，以便于阅读（如此处所示）。为了更好地适应该范围，使用对数刻度绘制此轴。
- **Horizontal Axis** - 水平轴表示算术强度，它是计算（以浮点运算/秒表示）与内存流量（以字节/秒表示）之间的比率。结果单位是每个字节的浮点运算。该轴也使用对数刻度显示。
- **Memory Bandwidth Boundary** - 内存带宽边界是屋顶线的倾斜部分。默认情况下，此斜率完全由GPU的内存传输速率确定，但可以根据需要在SpeedOfLight_RooflineChart.section文件中自定义。
- **Peak Performance Boundary** - 峰值性能边界是Roofline 的平坦部分。默认情况下，该值完全由GPU的峰值性能决定，但可以根据需要在SpeedOfLight_RooflineChart.section文件中自定义。
- **Ridge Point** - 脊点是内存带宽边界与峰值性能边界相交的点。这是分析内核性能时的有用参考。
- **Achieved Value** - 达到的值表示配置文件内核的性能。如果使用基线，则Roofline图表还将包含每个基线的已实现值。绘制的实现值点的轮廓颜色可用于确定该点来自哪个基线。

### 7.2. Analysis

Roofline图对于指导特定内核的性能优化工作很有帮助。

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/roofline-analysis.png)

如此处所示，山脊点将车顶线图分为两个区域。倾斜的“内存带宽边界”下蓝色阴影区域是“内存Bound ”区域，而“峰值性能边界”下绿色阴影区域是“计算Bound ”区域。达到的值所在的区域决定了内核性能的当前限制因素。

从达到的值到各个roofline边界的距离（在该图中显示为白色虚线）代表了性能提高的机会。获得的值越接近roofline边界，其性能越好。位于存储器带宽边界上但尚未达到脊点高度的已实现值将表明，只有同时增加算术强度时，总体FLOP/s的任何进一步改善都是可能的。

将基线功能与屋顶线图结合使用，是跟踪许多内核执行过程中优化进度的好方法。

## 8. Memory Chart

内存图表显示了GPU上和下的内存子单元的性能数据的图形逻辑表示。效果数据包括传输大小，命中率，指令或请求数等。

### 8.1. Overview

NVIDIA A100 GPU的内存图表

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/memory-chart-a100.png)

### Logical Units (green)

逻辑单位以绿色显示。

加载共享的全局存储：直接从全局加载到共享内存的指令，而无需中间寄存器文件访问。

### Physical Units (blue)

物理单位显示为蓝色。

- L1/TEX Cache: L1/纹理缓存。基础物理内存在此缓存和用户管理的共享内存之间分配。
- Shared Memory: CUDA的用户管理的共享内存。基础物理内存在此缓存和L1/TEX缓存之间分配。
- L2 Cache: The L2 cache
- L2 Compression:  L2缓存的内存压缩单元
- System Memory: 片外系统（CPU）内存
- Device Memory: 执行内核的CUDA设备的片上设备（GPU）内存
- Peer Memory: 其他CUDA设备的片上设备（GPU）内存

取决于确切的GPU架构，所显示单位的确切集合可能会有所不同，因为并非所有GPU都具有所有单位。

### Links

内核和其他逻辑单元之间的链接表示针对各个单元的已执行指令（Inst）的数量。例如，内核和全局之间的链接表示从全局存储空间加载或存储到全局存储空间的指令。分别显示了使用NVIDIA A100的“加载全局存储共享”范例的说明，因为它们的注册或缓存访问行为可能不同于常规的全局加载或共享存储。

逻辑单元和蓝色物理单元之间的链接表示由于其各自指令而发出的请求数（Req）。例如，从L1/TEX高速缓存到全局的链接显示了由于全局加载指令而生成的请求数。

每个链接的颜色代表相应通信路径的峰值利用率百分比。图表右侧的颜色图例显示了从未使用（0％）到以峰值性能（100％）运行时所应用的颜色渐变。图例左侧的三角形标记与图表中的链接相对应。与仅使用颜色渐变相比，这些标记可为获得的峰值性能提供更准确的值估计。

一个单元通常共享一个公共数据端口，用于传入和传出流量。尽管共享端口的链路可能会在各自的峰值性能以下运行，但是设备的数据端口可能已经达到峰值。图表中，端口利用率在输入和输出链接处的单元内以彩色矩形显示。端口使用与数据链接相同的颜色渐变，并且在图例的左侧也有一个相应的标记。

## 9. Memory Tables

内存表显示了各种内存硬件单元的详细指标，例如共享内存，缓存和设备内存。对于大多数表条目，您可以将鼠标悬停在其上以查看基础指标名称。某些条目是作为其他单元格的派生而生成的，==它们本身未显示度量标准名称==。您可以将鼠标悬停在行标题或列标题上，以查看对该表部分的描述。

### 9.1. Shared Memory

Example Shared Memory table, collected on an RTX 2080 Ti

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/memory-tables-smem.png)

### Columns

| Name             | 介绍                                                         |
| ---------------- | ------------------------------------------------------------ |
| `Instructions`   | 对于每种访问类型，每个warp中所有实际执行的汇编（SASS）指令的总数。不包括断言指令，例如，指令STS将计入Shared Store。 |
| `Requests`       | 共享内存的所有请求总数。在SM 7.0（Volta）和更高版本的体系结构上，每条共享内存指令都恰好生成一个请求。 |
| `Wavefronts`     | 服务请求的共享内存数据所需的wavefronts波前数。wavefronts波前被序列化并在不同的周期进行处理。 |
| `% Peak`         | 峰值利用率百分比。较高的值表示该单元的利用率较高，并且可能显示潜在的瓶颈，因为它不一定表示有效使用。 |
| `Bank Conflicts` | 如果多个线程的请求地址映射到同一内存库中的不同偏移量，则访问将被序列化。硬件根据需要将冲突的内存请求分成尽可能多的独立的无冲突请求，从而将有效带宽减少等于冲突的内存请求数量的因数。 |

### Rows

| Name             | 介绍                         |
| ---------------- | ---------------------------- |
| `(Access Types)` | 共享内存访问操作。           |
| `Total`          | 同一列中所有访问类型的汇总。 |

### 9.2. L1/TEX Cache

Example L1/TEX Cache memory table, collected on an RTX 2080 Ti

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/memory-tables-l1.png)

### Columns

| Name                  | 介绍                                                         |
| --------------------- | ------------------------------------------------------------ |
| `Instructions`        | 对于每种访问类型，每个Warp中所有实际执行的汇编（SASS）指令的总数。不包括断言指令，例如LDG指令将计入全局负载。 |
| `Requests`            | 针对每种指令类型生成的对L1的所有请求的总数。在SM 7.0（Volta）和更新的体系结构上，每条指令仅生成一个LSU通信请求（全局，本地，...）。对于Texture（TEX）流量，可能会生成多个请求。在此示例中，每个65536个全局加载指令恰好生成一个请求。 |
| `Wavefronts`          | 服务请求的内存操作所需的wavefronts波前数。波前被序列化并在不同的周期进行处理。 |
| `Wavefront % Peak`    | 处理wavefronts的单元的峰值利用率百分比。高数量可能意味着处理管道已饱和，并且可能成为瓶颈。 |
| `Sectors`             | 发送到L1的所有L1扇区访问的总数。每个加载或存储请求都访问L1缓存中的一个或多个扇区。原子和约简传递到L2高速缓存。 |
| `Sectors/Req`         | 扇区与L1缓存请求的平均比率。对于经线中相同数量的活动线程，==较小的数字表示更有效的内存访问模式==。对于具有32个活动线程的经线，每个访问大小的最佳比率为：32位：4、64位：8、128位：16。较小的比率表示高速缓存行内某种程度的均匀性或重叠负载。较高的数量可能意味着不分先后的内存访问，并会导致内存流量增加。在此示例中，每个请求的全局负载平均比率为32个扇区，这意味着每个线程需要访问一个不同的扇区。理想情况下，对于具有32个活动线程的扭曲，每个线程访问一个对齐的32位值，比率为4，因为每8个连续线程访问同一扇区。 |
| `Hit Rate`            | L1高速缓存中的扇区命中率（所请求的扇区百分比未丢失）。需要从L2请求未命中的扇区，从而为L2造成扇区未命中。较高的命中率由于较低的访问延迟而暗示了更好的性能，因为该请求可以由L1而不是稍后的阶段服务。不要与标签点击率混淆（未显示）。 |
| `Bytes`               | ==从L1请求的字节总数==。这与扇区数乘以32字节相同，因为L1中的最小访问大小为一个扇区。 |
| `Sector Misses to L2` | L1中未命中并在L2缓存中生成后续请求的扇区总数。在此示例中，全局和本地负载的262144扇区未命中率可以计算为12.5％的未命中率乘以2097152扇区数。 |
| `% Peak to L2`        | L1-to-XBAR接口的峰值利用率百分比，用于发送L2缓存请求。如果此数字很高，则工作量可能由分散的{写，原子，减少}主导，这可能会增加延迟并导致Warp Stall。 |
| `Returns to SM`       | 从L1缓存发送回SM的返回数据包数。请求访问大小越大，返回的数据包数量越多。 |
| `% Peak to SM`        | XBAR到L1返回路径的峰值利用率百分比（将返回值与SM比较）。如果此数字很高，则工作量很可能由分散的读取操作主导，从而导致Warp Stall。提高阅读市场营销或提高L1命中率可以降低这种利用率。 |

### Rows

| Name             | 介绍                                                         |
| ---------------- | ------------------------------------------------------------ |
| `(Access Types)` | 各种访问类型，例如从全局存储器加载或在表面存储器上进行归约操作。 |
| `Loads`          | 同一列中所有加载访问类型的汇总。                             |
| `Stores`         | 同一列中所有Store访问类型的汇总。                            |
| `Total`          | 同一列中所有加载和存储访问类型的汇总。                       |

### 9.3. L2 Cache

Example L2 Cache memory table, collected on an RTX 2080 Ti

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/memory-tables-l2.png)

### Columns

| Name                      | 介绍                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `Requests`                | 对于每种访问类型，对L2缓存的请求总数。这与L1缓存的L2扇区丢失有关。每个请求的目标是一个128字节的缓存行。 |
| `Sectors`                 | 对于每种访问类型，从L2高速缓存请求的扇区总数。每个请求访问一个或多个扇区。 |
| `Sectors/Req`             | 扇区与二级高速缓存请求的平均比率。对于经线中相同数量的活动线程，较小的数字表示更有效的内存访问模式。对于具有32个活动线程的经线，每个访问大小的最佳比率为：32位：4、64位：8、128位：16。较小的比率表示高速缓存行内某种程度的均匀性或重叠负载。较高的数字可能意味着未分配内存访问权限，并会导致内存流量增加。 |
| `% Peak`                  | 高峰持续sectors数的百分比。 L2缓存中的“工作包”是一个扇区。较高的值表示该单元的利用率较高，并且可能显示潜在的瓶颈，因为它不一定表示有效使用。 |
| `Hit Rate`                | L2高速缓存中的命中率（不丢失的请求扇区的百分比）。需要从稍后的阶段请求未命中的扇区，从而促成设备的扇区未命中，系统的扇区未命中或对等的扇区未命中之一。由于较低的访问延迟，较高的命中率意味着更好的性能，因为该请求可以由L2而不是稍后的阶段处理。 |
| `Bytes`                   | ==从L2请求的字节总数==。这与扇区数乘以32字节相同，因为L2中的最小访问大小为一个扇区。 |
| `Throughput`              | L2高速缓存吞吐量达到了每秒字节数。==高值表示该单元的利用率高==。 |
| `Sector Misses to Device` | L2中未命中并在设备内存中生成后续请求的扇区总数。             |
| `Sector Misses to System` | L2中未命中并在系统内存中生成后续请求的扇区总数。             |
| `Sector Misses to Peer`   | L2中未命中并在对等内存中生成后续请求的扇区总数。             |

### Rows

| Name             | 介绍                                                         |
| ---------------- | ------------------------------------------------------------ |
| `(Access Types)` | 各种访问类型，例如源自L1缓存的loads or reductions量。        |
| `L1/TEX Total`   | 源自L1缓存的所有操作的总计。                                 |
| `GPU Total`      | L2缓存的所有客户端上所有操作的总计。与使它们在此表中分开分开无关。 |

### 9.4. Device Memory

Example Device Memory table, collected on an RTX 2080 Ti

![img](https://docs.nvidia.com/nsight-compute/ProfilingGuide/graphics/memory-tables-dram.png)

### Columns

| Name         | 介绍                                                         |
| ------------ | ------------------------------------------------------------ |
| `Sectors`    | 对于每种访问类型，从设备内存请求的扇区总数。                 |
| `% Peak`     | 设备内存利用率峰值的百分比。较高的值表示该单元的利用率较高，并且可能显示潜在的瓶颈，因为它不一定表示有效使用。 |
| `Bytes`      | ==L2缓存和设备内存之间传输的字节总数==。                     |
| `Throughput` | ==设备内存吞吐量==，以每秒字节数为单位。高值表示该单元的利用率高。 |

### Rows

| Name             | 介绍                         |
| ---------------- | ---------------------------- |
| `(Access Types)` | 设备内存加载和存储。         |
| `Total`          | 同一列中所有访问类型的汇总。 |


