## API Statistic

![1603953554101](D:\Notes\raw_images\1603953554101.png)

统计信息：

- Name
- Number of Calls 
- Total Duration
- Average Duration
- Minimun Duration
- Maximun Duration
- Duration Standard Deviation

![1603956344382](D:\Notes\raw_images\1603956344382.png)

## 统计的方面包括

![1603953314585](D:\Notes\raw_images\1603953314585.png)

![1603954675609](D:\Notes\raw_images\1603954675609.png)



### GPU Speed Of Light

GPU的计算和内存资源利用率的高级概述。对于每个单元，光速(SOL)报告了相对于理论最大值的已实现的利用率百分比。图形处理器的计算和内存资源的利用率的==高层概述==以一个顶部图表的形式呈现。

![1603954092304](D:\Notes\raw_images\1603954092304.png)

![1603954113690](D:\Notes\raw_images\1603954113690.png)

![1603954130961](D:\Notes\raw_images\1603954130961.png)

![1603954142316](D:\Notes\raw_images\1603954142316.png)

![1603954158942](D:\Notes\raw_images\1603954158942.png)

![1603954178369](D:\Notes\raw_images\1603954178369.png)

![1610627172721](D:\Notes\raw_images\1610627172721.png)

![1610627192106](D:\Notes\raw_images\1610627192106.png)

![1610627243652](D:\Notes\raw_images\1610627243652.png)

**Bottleneck**

The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. 

内核正在利用设备超过80.0%的可用计算或内存性能。为了进一步改善业绩，很可能需要把利用最多的单位的工作转移到另一个单位。

**Roofline Analysis**

The ratio of peak float (fp32) to double (fp64) performance on this device is 2:1. The kernel achieved  close to 1% of this device's fp32 peak performance and 0% of its fp64 peak performance.

This chart actually shows two different rooflines. However, the following components can be identified for each:

- **Vertical Axis** - The vertical axis represents Floating Point Operations per Second (FLOPS). For GPUs this number can get quite large and so the numbers on this axis can be scaled for easier reading (as shown here). In order to better accommodate the range, this axis is rendered using a logarithmic scale.
- **Horizontal Axis** - The horizontal axis represents Arithmetic Intensity, which is the ratio between Work (expressed in floating point operations per second), and Memory Traffic (expressed in bytes per second). ==The resulting unit is in floating point operations per byte==. This axis is also shown using a logarithmic scale.
- **Memory Bandwidth Boundary** - The memory bandwidth boundary is the *sloped* part of the roofline. By default, this slope is determined entirely by the memory transfer rate of the GPU but can be customized inside the SpeedOfLight_RooflineChart.section file if desired.
- **Peak Performance Boundary** - The peak performance boundary is the *flat* part of the roofline By default, this value is determined entirely by the peak performance of the GPU but can be customized inside the SpeedOfLight_RooflineChart.section file if desired.
- **Ridge Point** - The ridge point is the point at which the memory bandwidth boundary meets the peak performance boundary. This point is a useful reference when analyzing kernel performance.
- **Achieved Value** - The achieved value represents the performance of the profiled kernel. If baselines are being used, the roofline chart will also contain an achieved value for each baseline. The outline color of the plotted achieved value point can be used to determine from which baseline the point came.

### Compute Workload Analysis

流多处理器(SM)的计算资源的详细分析，包括实现的每时钟指令(IPC)和每个可用管道的利用。具有很高利用率的管道可能会限制总体性能。

![1603954243290](D:\Notes\raw_images\1603954243290.png)

![1603954251815](D:\Notes\raw_images\1603954251815.png)

![1610436492747](D:\Notes\raw_images\1610436492747.png)

### Memory Workload Analysis

Detailed analysis of the memory resources of the GPU. 

1. Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (==Mem Busy==), 
2. exhausting the available communication bandwidth between those units (==Max Bandwidth==), 
3. or by reaching the maximum throughput of issuing memory instructions (==Mem Pipes Busy==). 

Detailed chart of the memory units. Detailed tables with data for each memory unit. Deprecated UI elements for backwards compatibility.

详细分析了GPU的内存资源。当充分利用所涉及的硬件单元(Mem繁忙)、耗尽这些单元之间的可用通信带宽(最大带宽)或达到发出内存指令的最大吞吐量(Mem管道繁忙)时，内存可能成为内核总体性能的一个限制因素。内存单元的详细图表。每个内存单元的详细数据表。为了向后兼容而弃用的UI元素。

![1603954270543](D:\Notes\raw_images\1603954270543.png)

![1603954279100](D:\Notes\raw_images\1603954279100.png)

![1603954293691](D:\Notes\raw_images\1603954293691.png)

![1603954305970](D:\Notes\raw_images\1603954305970.png)

|                  |                                                              |
| ---------------- | ------------------------------------------------------------ |
| `Instructions`   | For each access type, the total number of all actually executed assembly (SASS) [instructions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) per warp. Predicated-off instructions are not included.E.g., the instruction STS would be counted towards Shared Store. |
| `Wavefronts`     | Number of wavefronts required to service the requested shared memory data. |
| `% Peak`         | Percentage of peak utilization. Higher values imply a higher utilization of the unit and can show potential bottlenecks, as it does not necessarily indicate efficient usage. |
| `Bank Conflicts` | If multiple threads' requested addresses map to different offsets in the same memory bank, the accesses are serialized. The hardware splits a conflicting memory request into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of colliding memory requests. |

|                  |                                                        |
| ---------------- | ------------------------------------------------------ |
| `(Access Types)` | Shared memory access operations.                       |
| `Total`          | The aggregate for all access types in the same column. |

![1603954322305](D:\Notes\raw_images\1603954322305.png)

|                       |                                                              |
| --------------------- | ------------------------------------------------------------ |
| `Instructions`        | For each access type, the total number of all actually executed assembly (SASS) [instructions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) per warp. Predicated-off instructions are not included.E.g., the instruction LDG would be counted towards Global Loads. |
| `Requests`            | The total number of all [requests](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) to L1, generated for each instruction type. On SM 7.0 (Volta) and newer architectures, each instruction generates exactly one request for LSU traffic (global, local, ...). For texture (TEX) traffic, more than one request may be generated.In the example, each of the 65536 global load instructions generates exactly one request. |
| `Sectors`             | The total number of all L1 [sectors](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) accesses sent to L1. Each load or store request accesses one or more sectors in the L1 cache. Atomics and reductions are passed through to the L2 cache. |
| `Sectors/Req`         | The average ratio of sectors to requests for the L1 cache. For the same number of active threads in a warp, smaller numbers imply a more efficient memory access pattern. For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16. Smaller ratios indicate some degree of uniformity or overlapped loads within a cache line. Higher numbers can imply uncoalesced memory accesses and will result in increased memory traffic.In the example, the average ratio for global loads is 32 sectors per request, which implies that each thread needs to access a different sector. Ideally, for warps with 32 active threads, with each thread accessing a single, aligned 32-bit value, the ratio would be 4, as every 8 consecutive threads access the same sector. |
| `Wavefront % Peak`    | Percentage of peak utilization for the units processing [wavefronts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities). High numbers can imply that the processing pipelines are saturated and can become a bottleneck. |
| `Hit Rate`            | [Sector](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) hit rate (percentage of requested sectors that do not miss) in the L1 cache. Sectors that miss need to be requested from L2, thereby contributing to Sector Misses to L2. Higher hit rates imply better performance due to lower access latencies, as the request can be served by L1 instead of a later stage. Not to be confused with Tag Hit Rate (not shown). |
| `Bytes`               | Total number of bytes requested from L1. This is identical to the number of sectors multiplied by 32 byte, since the minimum access size in L1 is one sector. |
| `Sector Misses to L2` | Total number of sectors that miss in L1 and generate subsequent requests in the [L2 Cache](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables-l2).In this example, the 262144 sector misses for global and local loads can be computed as the miss-rate of 12.5%, multiplied by the number of 2097152 sectors. |
| `% Peak to L2`        | Percentage of peak utilization of the L1-to-XBAR interface, used to send L2 cache requests. If this number is high, the workload is likely dominated by scattered {writes, atomics, reductions}, which can increase the latency and cause [warp stalls](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler__warp-scheduler-states). |
| `Returns to SM`       | Number of return packets sent from the L1 cache back to the SM. Larger request access sizes result in higher number of returned packets. |
| `% Peak to SM`        | Percentage of peak utilization of the XBAR-to-L1 return path (compare Returns to SM). If this number is high, the workload is likely dominated by scattered reads, thereby causing [warp stalls](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler__warp-scheduler-states). Improving read-coalescing or the L1 hit rate could reduce this utilization. |

|                  |                                                              |
| ---------------- | ------------------------------------------------------------ |
| `(Access Types)` | The various access types, e.g. loads from global memory or reduction operations on surface memory. |
| `Loads`          | The aggregate of all load access types in the same column.   |
| `Stores`         | The aggregate of all store access types in the same column.  |
| `Total`          | The aggregate of all load and store access types in the same column. |

![1603954336166](D:\Notes\raw_images\1603954336166.png)

|                           |                                                              |
| ------------------------- | ------------------------------------------------------------ |
| `Requests`                | For each access type, the total number of [requests](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) made to the L2 cache. This correlates with the [Sector Misses to L2](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables-l1__memory-tables-l1-columns) for the L1 cache. Each request targets one 128 byte cache line. |
| `Sectors`                 | For each access type, the total number of [sectors](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) requested from the L2 cache. Each request accesses one or more sectors. |
| `Sectors/Req`             | The average ratio of sectors to requests for the L2 cache. For the same number of active threads in a warp, smaller numbers imply a more efficient memory access pattern. For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16. Smaller ratios indicate some degree of uniformity or overlapped loads within a cache line. Higher numbers can imply [uncoalesced memory accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) and will result in increased memory traffic. |
| `% Peak`                  | Percentage of peak sustained number of sectors. The "work package" in the L2 cache is a sector. Higher values imply a higher utilization of the unit and can show potential bottlenecks, as it does not necessarily indicate efficient usage. |
| `Hit Rate`                | Hit rate (percentage of requested sectors that do not miss) in the L2 cache. Sectors that miss need to be requested from a later stage, thereby contributing to one of Sector Misses to Device, Sector Misses to System, or Sector Misses to Peer. Higher hit rates imply better performance due to lower access latencies, as the request can be served by L2 instead of a later stage. |
| `Bytes`                   | Total number of bytes requested from L2. This is identical to the number of sectors multiplied by 32 byte, since the minimum access size in L2 is one sector. |
| `Throughput`              | Achieved L2 cache throughput in bytes per second. High values indicate high utilization of the unit. |
| `Sector Misses to Device` | Total number of sectors that miss in L2 and generate [subsequent requests](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables-dram) in [device memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model__metrics-hw-memory). |
| `Sector Misses to System` | Total number of sectors that miss in L2 and generate subsequent requests in [system memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model__metrics-hw-memory). |
| `Sector Misses to Peer`   | Total number of sectors that miss in L2 and generate subsequent requests in [peer memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model__metrics-hw-memory). |

|                  |                                                              |
| ---------------- | ------------------------------------------------------------ |
| `(Access Types)` | The various access types, e.g. loads or reductions originating from L1 cache. |
| `L1/TEX Total`   | Total for all operations originating from the L1 cache.      |
| `GPU Total`      | Total for all operations across all clients of the L2 cache. Independent of having them split out separately in this table. |

![1603954352044](D:\Notes\raw_images\1603954352044.png)

|              |                                                              |
| ------------ | ------------------------------------------------------------ |
| `Sectors`    | For each access type, the total number of [sectors](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder__metrics-quantities) requested from device memory. |
| `% Peak`     | Percentage of peak device memory utilization. Higher values imply a higher utilization of the unit and can show potential bottlenecks, as it does not necessarily indicate efficient usage. |
| `Bytes`      | Total number of bytes transferred between [L2 Cache](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables-l2) and device memory. |
| `Throughput` | Achieved device memory throughput in bytes per second. High values indicate high utilization of the unit. |

|                  |                                                        |
| ---------------- | ------------------------------------------------------ |
| `(Access Types)` | Device memory loads and stores.                        |
| `Total`          | The aggregate for all access types in the same column. |

### Scheduler Statistics

Summary of the activity of the schedulers issuing instructions. 

- Each scheduler maintains a pool of warps that it can issue instructions for. The upper bound of warps in the pool (Theoretical Warps) is limited by the launch configuration. 
- On every cycle each scheduler checks the state of the allocated warps in the pool (Active Warps). 
- Active warps that are not stalled (Eligible Warps) are ready to issue their next instruction. 
- From the set of eligible warps the scheduler selects a single warp from which to issue one or more instructions (Issued Warp). 

On cycles with no eligible warps, the issue slot is skipped and no instruction is issued. Having many skipped issue slots indicates ==poor latency hiding==.

发布指令的调度程序的活动摘要。每个调度器维护一个可以对其发出指令的warps池。池中的warps (理论warps )的上限受到启动配置的限制。在每个周期中，每个调度程序都会检查池中分配的warps(活动warps)的状态。未停止的活动warps(合格的warps)准备发出它们的下一个指令。调度器从一组合格的warp中选择一个warp来发出一个或多个指令(发出的warp)。在没有合适的warps的周期中，将跳过问题slot插槽，不发出任何指令。有许多跳过的问题slot 槽表明隐藏的延迟很差。

![1603954378378](D:\Notes\raw_images\1603954378378.png)

![1603954387285](D:\Notes\raw_images\1603954387285.png)

### Warp State Statistics

Analysis of the states in which all warps spent cycles during the kernel execution. The warp states describe a warp's readiness or inability to issue its next instruction. The warp cycles per instruction define the latency between two consecutive instructions. ==The higher the value, the more warp parallelism is required to hide this latency==. For each warp state, the chart shows the average number of cycles spent in that state per issued instruction. Stalls are not always impacting the overall performance nor are they completely avoidable. Only focus on stall reasons if the schedulers fail to issue every cycle. When executing a kernel with mixed library and user code, these metrics show the combined values.

分析所有翘曲在内核执行期间的循环状态。warp状态描述了一个warp准备好或者不能发出下一个指令的状态。每个指令的翘曲周期定义了两个连续指令之间的延迟。值越高，隐藏这种延迟所需的扭曲并行度就越大。对于每个翘曲状态，图表显示了每个发出指令在该状态中花费的平均周期数。档位并不总是影响整体演出，也不是完全可以避免的。如果调度程序不能发出每个周期，只关注暂停原因。当执行带有混合库和用户代码的内核时，这些指标显示组合的值。

![1603954413925](D:\Notes\raw_images\1603954413925.png)

![1603954430726](D:\Notes\raw_images\1603954430726.png)

### Instruction Statistics

Statistics of the executed low-level assembly instructions (SASS). The instruction mix provides insight into the types and frequency of the executed instructions. A narrow mix of instruction types implies a dependency on few instruction pipelines, while others remain unused. Using multiple pipelines allows hiding latencies and enables parallel execution. Note that 'Instructions/Opcode' and 'Executed Instructions' are measured differently and can diverge if cycles are spent in system calls.

已执行的低级汇编指令(SASS)的统计信息。指令组合提供了对执行指令的类型和频率的洞察。指令类型的混合很窄，意味着依赖于少数指令管道，而其他指令管道仍未使用。使用多个管道可以隐藏延迟并支持并行执行。请注意，“指令/操作码”和“执行指令”的度量是不同的，如果周期花在系统调用上，则可能会发生偏离。

![1603954531346](D:\Notes\raw_images\1603954531346.png)

![1603954541098](D:\Notes\raw_images\1603954541098.png)

![1603954629985](D:\Notes\raw_images\1603954629985.png)

### Launch Statistics

Summary of the configuration used to launch the kernel. The launch configuration defines the size of the kernel grid, the division of the grid into blocks, and the GPU resources needed to execute the kernel. Choosing an efficient launch configuration maximizes device utilization.

用于启动内核的配置摘要。启动配置定义了==内核网格的大小==、==网格的块划分==以及==执行内核所需的GPU资源==。选择有效的启动配置可以最大化设备利用率。

![1603954559468](D:\Notes\raw_images\1603954559468.png)

![1603954569780](D:\Notes\raw_images\1603954569780.png)

### Occupancy

Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps. Another way to view occupancy is the percentage of the hardware's ability to process warps that is actively in use. Higher occupancy does not always result in higher performance, however, low occupancy always reduces the ability to hide latencies, resulting in overall performance degradation. Large discrepancies between the theoretical and the achieved occupancy during execution typically indicates highly imbalanced workloads.

==占用率是每个多处理器的活动翘曲数与可能的最大活动翘曲数的比值==。另一种查看占用率的方法是查看正在使用的硬件处理翘曲的能力的百分比。较高的占用率并不总是会带来较高的性能，但是，较低的占用率总是会降低隐藏延迟的能力，从而导致整体性能下降。在执行期间，理论占用率和实现占用率之间的巨大差异通常表明工作负载高度不平衡。

![1603954587282](D:\Notes\raw_images\1603954587282.png)

![1603954596798](D:\Notes\raw_images\1603954596798.png)

### Source Counters

Source metrics, including ==warp stall reasons==. Sampling Data metrics are periodically sampled over the kernel runtime. They indicate when warps were stalled and couldn't be scheduled. See the documentation for a description of all stall reasons. Only focus on stalls if the schedulers fail to issue every cycle.

来源指标，包括曲速失速原因。抽样数据度量在内核运行时周期性地抽样。他们指出什么时候翘曲被搁置，不能被安排。有关所有失速原因的描述，请参阅文档。只有在调度程序没有发出每个周期的情况下才关注档位。

