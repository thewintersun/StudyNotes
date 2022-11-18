## TPC

官网地址: http://www.tpc.org/information/benchmarks5.asp

子集（Benchmarks)

**TPC-C：** TPC-C is an On-Line Transaction Processing Benchmark (主要针对数据库)

TPC-C是专门针对联机交易处理系统（OLTP系统）的规范，一般情况下我们也把这类系统称为[业务处理系统](https://baike.baidu.com/item/业务处理系统/7535448)。1992年7月发布。几乎所有在OLTP市场提供软硬平台的国外主流厂商都发布了相应的TPC-C测试结果，随着计算机技术的不断发展，这些测试结果也在不断刷新。

**TPC-DI：**TPC-DI is a benchmark for Data Integration

**TPC-DS：**TPC-DS is a Decision Support Benchmark

**TPC-VMS：**TPC-VMS is a Data Virtualization Benchmark

**TPCx-BB:** TPCx-BB is a Big Data Benchmark

**TPCx-IoT:** TPCx-IoT is a Benchmark for IoT (Internet of Things) Gateway Systems （主要针对网络）

![1599136374882](D:\Notes\raw_images\1599136374882.png)



### TPC-C Benchmark Top Results

http://www.tpc.org/tpcc/results/tpcc_perf_results5.asp?resulttype=all

![1599137033540](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1599137033540.png)

指标: 更多考虑性能和价格问题。

![1599137116293](D:\Notes\raw_images\1599137116293.png)



### 1.2 Characteristics of the TPC-E Workload

https://link.springer.com/chapter/10.1007/978-3-319-15350-6_10

Of the 4 possible workloads to use under TPC-VMS rules, we used the TPC-E workload. TPC Benchmark™ E is composed of a set of transactional operations designed to exercise system functionalities in a manner representative of complex OLTP database application environments. These transactional operations have been given a life-like context, portraying the activity of a brokerage firm, to help users relate intuitively to the components of the benchmark. The brokerage firm must manage customer accounts, execute customer trade orders, and be responsible for the interactions of customers with financial markets. Figure [1](https://link.springer.com/chapter/10.1007/978-3-319-15350-6_10#Fig1)illustrates the transaction flow of the business model portrayed in the benchmark:

The customers generate transactions related to trades, account inquiries, and market research. The brokerage firm in turns interacts with financial markets to execute orders on behalf of the customers and updates relevant account information. The number of customers defined for the brokerage firm can be varied to represent the workloads of different size businesses.

The benchmark is composed of a set of transactions that are executed against three sets of database tables that represent market data, customer data, and broker data. A fourth set of tables contains generic dimension data such as zip codes.

The benchmark has been reduced to simplified form of the application environment. To measure the performance of the OLTP system, a simple Driver generates Transactions and their inputs, submits them to the System Under Test (SUT), and measures the rate of completed Transactions being returned. This number of transactions is considered the performance metric for the benchmark.

### TPC-AI 建立的消息

December 12, 2017 12:00 PM Eastern Standard Time

https://www.businesswire.com/news/home/20171212005281/en/Transaction-Processing-Performance-Council-TPC-Establishes-Artificial

目前的人工智能生态系统包含了从传统到各种各样的技术和产品。“人工智能将成为整个行业的主流，”TPC-AI工作组主席、思科UCS首席技术官Raghunath Nambiar表示。“这些工作负载与TPC的核心焦点区域毗邻，数据密集程度极高。我们新的人工智能工作组的首要任务是开发一套标准化指标，以便在一系列不同的硬件和软件堆栈之间进行公平的价格和性能比较。我们正在积极鼓励行业专家和研究界成员在这方面帮助我们。”