## SPEC: Standard Performance Evaluation Corporation

官网：https://www.spec.org/benchmarks.html

Wiki：https://en.wikipedia.org/wiki/Standard_Performance_Evaluation_Corporation



## SPEC 评分套件

SPEC（Standard Performance Evaluation Corporation，标准性能评估组织）是一个全球性的第三方非营利性组织，致力于建立、维护和认证一套应用于计算机的标准化基准评测套件，SPEC组织开发基准测试套件并经过检验然后在SPEC网站上公开测试结果。

### SPEC Benchmark Suites 

- **Cloud：** Measuring and comparing the provisioning, compute, storage, and network resources of IaaS cloud platforms.
- **CPU**： Measuring and comparing combined performance of CPU, memory and compiler. 
  - SPECspeed 2017包含下面4个套件：
  - The SPECspeed 2017 Integer and SPECspeed 2017 Floating Point suites are used for ==comparing time== for a computer to complete single tasks.
  - The SPECrate 2017 Integer and SPECrate 2017 Floating Point suites measure the ==throughput== or work per unit of time。
- Handheld
- **Graphics and Workstation Performance：**测试OpenGL 3D图形系统的性能，测试了多个流行的3D密集的真实应用程序在一个给定的系统上的各种渲染任务。
  - SPECwpc
  - High Performance Computing, OpenMP, MPI, OpenACC, OpenCL
- **Java Client/Server**
  - SPEC服务器应用性能测试是一个全面衡量Web应用中java企业应用服务器性能的基准测试。在这个基准测试中，系统模拟一个现代化企业的电子化业务工作，如客户定购查询、产品生产制造管理、供应商和服务器提供商管理等，给系统以巨大的负载，以全面测试运行典型java业务应用的服务器性能水平。由于它体现了软、硬件平台的性能和成本指标，被金融、电信、证券等关键行业用户作为选择IT系统一项权威的选型指标。
  - JBB
  - jEnterprise
- Mail Servers
- Storage： SPEC SFS 2014
- Power： SPECpower_ssj 2008
- Virtualization： SPECvirt_sc2013
- Web Servers



### SPEC ACCEL

https://www.spec.org/accel/

SPEC ACCEL®基准测试套件测试在OpenCL、OpenACC和openmp4目标 offloading  api下运行的计算密集型并行应用程序的性能。该套件测试了加速器（accelerator）、主机CPU、主机和加速器之间的内存传输、支持库和驱动程序、以及编译器的性能。

基准套件包含19个在OpenCL下运行的应用程序基准,15个在OpenACC下运行,15个使用OpenMP 4目标offloading  。OpenACC和OpenMP套件包括来自NAS平行基准(NPB)、SPEC OMP®2012和其他来自高性能计算(HPC) 应用程序。

应用程序基准

Benchmarks by suite

- Benchmarks in the SPEC ACCEL OpenCL suite
  - [120.kmeans](http://www.spec.org/accel/Docs/120.kmeans.html) (C++) 
  - [126.ge](http://www.spec.org/accel/Docs/126.ge.html) (C++)
- Benchmarks in the SPEC ACCEL OpenACC suite
- Benchmarks in the SPEC ACCEL OpenMP suite

https://www.spec.org/accel/docs/readme1st.html



### SPEC CPU2017 Results

SPEC CPU 2017专注于计算密集型性能,这意味着这些基准强调了性能:
- 处理器——CPU芯片。
- 内存——包括缓存和主内存在内的内存层次。
- 编译器- C、c++和Fortran编译器,包括优化器。

2017年的规范CPU依赖于以上三个——不仅仅是处理器。2017年的CPU CPU不打算强调其他计算机组件,如网络、图形、Java库或I / O系统。请注意, 这些领域还有其他的规范基准。

https://www.spec.org/cpu2017/results/

Results from all publication quarters, broken out by reported metric:

- Speed: [ [SPECspeed 2017 Integer](https://www.spec.org/cpu2017/results/cint2017.html), [SPECspeed 2017 Floating Point](https://www.spec.org/cpu2017/results/cfp2017.html) ]
- Throughput: [ [SPECrate 2017 Integer](https://www.spec.org/cpu2017/results/rint2017.html), [SPECrate 2017 Floating Point](https://www.spec.org/cpu2017/results/rfp2017.html) ]

==速度和吞吐量，然后每种又分整型和浮点，一共四种==。

速度+整型 排名结果列表：

https://www.spec.org/cpu2017/results/cint2017.html

![1603958811959](D:\Notes\raw_images\1603958811959.png)

指标：

![1599116969289](D:\Notes\raw_images\1599116969289.png)

速度+整型 测试结果样例：

https://www.spec.org/cpu2017/results/res2018q1/cpu2017-20171224-02025.html

![1599116882703](D:\Notes\raw_images\1599116882703.png)

吞吐量+整型 测试结果样例：

https://www.spec.org/cpu2017/results/res2017q4/cpu2017-20171211-01439.html

![1603958661311](D:\Notes\raw_images\1603958661311.png)

## CPU 2017 Metrics

#### What are "SPECspeed" and "SPECrate" metrics?

There are many ways to measure computer performance. Among the most common are:

- ==Time== - For example, seconds to complete a workload.
- ==Throughput== - Work completed per unit of time, for example, jobs per hour.

SPECspeed is a time-based metric; SPECrate is a throughput metric.

![1603979365730](D:\Notes\raw_images\1603979365730.png)



#### Q16. What are "base" and "peak" metrics?

SPEC CPU benchmarks are distributed as source code, and must be compiled, which leads to the question:
How should they be compiled? There are many possibilities, ranging from

```
--debug --no-optimize
```

at a low end through highly customized optimization and even source code re-writing at a high end. Any point chosen from that range might seem arbitrary to those whose interests lie at a different point. Nevertheless, choices must be made.

For CPU 2017, SPEC has chosen to allow two points in the range. The first may be of more interest to those who prefer a relatively simple build process; the second may be of more interest to those who are willing to invest more effort in order to achieve better performance.

- The base metrics (such as SPECspeed2017_int_base) require that all modules of a given language in a suite must be compiled using the same flags, in the same order. All reported results must include the base metric.
- The optional peak metrics (such as SPECspeed2017_int_peak) allow greater flexibility. Different compiler options may be used for each benchmark, and feedback-directed optimization is allowed.

Options allowed under the base rules are a subset of those allowed under the peak rules. A legal base result is also legal under the peak rules but a legal peak result is NOT necessarily legal under the base rules.

#### Q17. Which SPEC CPU 2017 metric should I use?

It depends on your needs; you get to choose, depending on how you use computers, and these choices will differ from person to person.
Examples:

- A single user running a variety of generic desktop programs may, perhaps, be interested in SPECspeed2017_int_base.

- A group of scientists running customized modeling programs may, perhaps, be interested in SPECrate2017_fp_peak.

  

### **Benchmarks**

![1603960516892](D:\Notes\raw_images\1603960516892.png)

**运行规则**

https://www.spec.org/cpu2017/Docs/runrules.html#rule_1

