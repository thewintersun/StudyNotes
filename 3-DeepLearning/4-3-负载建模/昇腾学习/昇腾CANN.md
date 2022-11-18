## CANN

AI场景的异构计算架构，通过提供多层次的编程接口，支持用户快速构建基于昇腾平台的AI应用和业务。

官方地址：https://www.huaweicloud.com/ascend/cann

下载中心：https://www.huaweicloud.com/ascend/cann-download

### 技术堆栈

![img](D:\Notes\raw_images\cannjishuduizhan.png)

### 核心组件

- ##### AscendCL 昇腾统一编程语言

  统一API适配昇腾全系列硬件，实现软硬件解耦。并且通过第三方框架调用AscendCL接口，使用昇腾AI处理器的计算能力，充分释放昇腾系统多样化算力，使能开发者快速开发AI应用。

- ##### TBE 编程语言

  用==Python语言开发算子==，通过==内置编译器将算子转换为高性能昇腾计算指令==，同时提供了完善的面向用户的开发集成环境，便于用户实现算子开发的调试、调优及诊断。

- ##### HCCL 华为集合通信库

  实现多昇腾AI处理器的collective communication通信库。并在集群训练中能够提供多机多卡间集合通信功能、梯度聚合功能和hcom集群通信算子，在分布式训练中不同昇腾AI处理器之间提供高效的数据传输能力。

- ##### DVPP 数字视觉预处理

  实现视频解码(VDEC)、视频编码(VENC)、JPEG解码(JPEGD)、JPEG编码(JPEGE)、PNG解码(PNGD)、VPC(预处理)。通过硬件加速进行图像预处理，能够降低对CPU的使用，提升图像预处理并行处理能力。

