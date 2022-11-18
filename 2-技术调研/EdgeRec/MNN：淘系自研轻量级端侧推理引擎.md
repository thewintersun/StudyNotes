## MNN：淘系自研轻量级端侧推理引擎

项目地址：https://github.com/alibaba/MNN

### 简介

MNN是一个高效、轻量的深度学习框架。它支持深度模型推理与训练，尤其在端侧的推理与训练性能在业界处于领先地位。目前，MNN已经在阿里巴巴的==手机淘宝、手机天猫、优酷、钉钉、闲鱼==等20多个App中使用，覆盖==直播、短视频、搜索推荐、商品图像搜索、互动营销、权益发放、安全风控==等70多个场景。此外，==IoT等场景==下也有若干应用。

MNN的架构设计理念与性能数据在MLSys 2020上面发表。Paper [在此处](https://arxiv.org/pdf/2002.12418.pdf)。

MNN的使用文档统一放在语雀，请移步至[语雀文档](https://www.yuque.com/mnn/cn)。

[MNN官网](http://www.mnn.zone/)上还可以下载MNN团队全新力作MNN工作台，涵盖开箱即用模型、可视化训练等工具，更可以一键部署到多端设备。



### MNN 特色

#### 高性能

- 不依赖任何第三方计算库，依靠大量手写汇编实现核心运算，充分发挥ARM CPU的算力。
- iOS设备上可以开启GPU加速（Metal），常用模型上快于苹果原生的CoreML。
- Android上提供了`OpenCL`、`Vulkan`、`OpenGL`三套方案，尽可能多地满足设备需求，针对主流GPU（`Adreno`和`Mali`）做了深度调优。
- 卷积、转置卷积算法高效稳定，对于任意形状的卷积均能高效运行，广泛运用了 Winograd 卷积算法，对3x3 -> 7x7之类的对称卷积有高效的实现。
- 针对ARM V2的新架构额外作了优化，新设备可利用FP16半精度计算的特性获得两倍提速。

#### 轻量性

- 针对端侧设备特点深度定制和裁剪，无任何依赖，可以方便地部署到移动设备和各种嵌入式设备中。
- iOS平台：armv7+arm64 ==静态库大小5MB==左右，链接生成可执行文件增加大小620KB左右，metallib文件600KB左右。
- Android平台：so大小400KB左右，OpenCL库400KB左右，Vulkan库400KB左右。

#### 通用性

- 支持`Tensorflow`、`Caffe`、`ONNX`等主流模型文件格式，支持`CNN`、`RNN`、`GAN`等常用网络。
- 转换器支持149个`Tensorflow`OP、58个`TFLite` OP、47个`Caffe` OP、74个`ONNX` OP；各计算设备后端支持的MNN OP数：CPU 111个，ARM V8.2 6个，Metal 55个，OpenCL 43个，Vulkan 32个。
- 支持iOS 8.0+、Android 4.3+和具有POSIX接口的嵌入式设备。
- 支持异构设备混合计算，目前支持CPU和GPU。

#### 易用性

- 有高效的图像处理模块，覆盖常见的形变、转换等需求，一般情况下，无需额外引入libyuv 或 opencv库处理图像。
- 支持回调机制，可以在网络运行中插入回调，提取数据或者控制运行走向。
- 支持只运行网络中的一部分，或者指定CPU和GPU间并行运行。
- （BETA）MNN Python API，让算法工程师可以轻松地使用MNN构图、训练、量化训练，无需编写C++。



### 架构设计

![image-20220209154401592](D:\Notes\raw_images\image-20220209154401592.png)

MNN可以分为Converter和Interpreter两部分。

Converter由Frontends和Graph Optimize构成。前者负责支持不同的训练框架，==MNN当前支持Tensorflow(Lite)、Caffe和ONNX(PyTorch/MXNet的模型可先转为ONNX模型再转到MNN)==；==后者通过算子融合、算子替代、布局调整等方式优化图==。

Interpreter由Engine和Backends构成。前者负责模型的加载、计算图的调度；后者包含各计算设备下的内存分配、Op实现。在Engine和Backends中，MNN应用了多种优化方案，包括在卷积和反卷积中应用Winograd算法、在矩阵乘法中应用Strassen算法、低精度计算、Neon优化、手写汇编、多线程优化、内存复用、异构计算等。



### License

Apache 2.0

MNN参与人员：淘宝技术部、搜索工程团队、达摩院团队、优酷等集团员工。

