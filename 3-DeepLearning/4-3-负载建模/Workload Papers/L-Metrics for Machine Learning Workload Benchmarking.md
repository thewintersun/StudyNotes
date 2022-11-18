## Metrics for Machine Learning Workload Benchmarking

机构：The University of Texas at Austin, The University of Texas at San Antonio, Dell Inc.

论文地址：

https://researcher.watson.ibm.com/researcher/files/us-ealtman/Snehil_Metrics_for_Machine_Learning_Workload_Benchmarking.pdf

Benchmarking for machine learning workloads should consider accuracy in addition to execution time or throughput. The emerging MLPerf benchmark suite touts Time to Accuracy (TTA) as the metric. 

In this paper, we explore the advantages and disadvantages of different metrics that consider time and accuracy from the perspective of comparing the
hardware used for machine learning training.

Single-threshold training time (e.g., time to accuracy) versus multi-threshold training time is one of the comparisons we articulate. We believe that the choice of a single threshold limits the information that can be revealed from the run of the benchmark and sometimes makes it difficult to interpret the information for further comparison. We find that the Time to Accuracy metric is ==highly sensitive to the specific threshold chosen==, and ==to the seed values== in the machine learning algorithms. We show that merely taking into account the time for training to multiple thresholds makes the metric less sensitive to the specific threshold chosen and the seed values.

- MLPerf

  - The organization is developing two sets of benchmarks, one for training and one for inference.

  - The MLPerf suite includes applications from major areas of machine learning such as image classification, object detection, recommendation, and language translation.

  - Time to Accuracy (TTA) metric

    ![image-20200908111613861](D:\Notes\raw_images\image-20200908111613861.png)

    ![image-20200908112109270](D:\Notes\raw_images\image-20200908112109270.png)

- SPEC 
  
  - The SPEC benchmarks are fixed task benchmarks and execution time for the entire run is used as the metric.
  
- HINT benchmark
  - HINT benchmark [9], a variable time benchmark, considered the quality of the result and used ==quality improvement per second (QUIPS)==
    as a metric. However, nonlinearity in quality improvement makes it challenging to use similar metrics for deep learning benchmarking. 

- DAWNBench 
  - DAWNBench [5] recently introduced the idea of end-to-end training time and proposed a ==Time to Accuracy (TTA) metric==.
  - We call the quality targets as thresholds and refer the time to accuracy (TTA) as the time to the threshold (TTT) interchangeably.
    

If oneâs workload is exactly the same as the benchmark and desired quality is identical, TTT or TTA is a useful metric, but if the workload is different from the benchmark or the required threshold is not the same, the indication given by the area under TTA curve near the quality target is better indicative of
the performance.

![image-20200908141643135](D:\Notes\raw_images\image-20200908141643135.png)

We propose a metric that can increase the information delivered in a single run which can be useful when users of the benchmarks wish to compare different platforms or systems.

==Time to Multiple Thresholds (TTMT)== curves and ==Average Time to Multiple Thresholds (ATTMT)==

The Time to Multiple Thresholds (TTMT) curves are plotted by collecting and joining the points (ti, Ti), where Ti is one of the various chosen thresholds and ti is the time to reach that particular threshold (time to reach Ti). Rather than having a single TTA value, TTA values for a variety of thresholds is captured in a TTMT curve. the last threshold (Tl) is chosen the same as the quality target decided by MLPerf.

we define a metric Average Time to Multiple Thresholds (ATTMT). This metric is defined for a certain range of thresholds, and the scalar value is equal to the arithmetic mean of the TTA for that threshold range. ATTMT will use the same units as used by TTA and can be specified with three parameters:
the low-end of the range, the high-end of the range, and the granularity at which the threshold varies.

