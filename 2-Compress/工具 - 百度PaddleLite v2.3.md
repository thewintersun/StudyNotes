## 模型压缩高达75%，推理速度提升超20%，百度Paddle Lite v2.3正式发布

文章：https://mp.weixin.qq.com/s/rLcpQfqTcW_WWXOHVsYBHg

项目：https://github.com/PaddlePaddle/PaddleSlim

## Features

### Pruning

- Uniform pruning of convolution
- ==Sensitivity-based== prunning
- Automated pruning based ==evolution search strategy==
- Support pruning of various deep architectures such as VGG, ResNet, and MobileNet.
- Support ==self-defined range of pruning==, i.e., layers to be pruned.

### Fixed Point Quantization

- Training aware
  - Dynamic strategy: During inference, we quantize models with hyperparameters dynamically estimated from small batches of samples.
  - Static strategy: During inference, we quantize models with the same hyperparameters estimated from training data.
  - Support layer-wise and channel-wise quantization.
- **Post training**

### Knowledge Distillation

- **Naive knowledge distillation:** transfers dark knowledge by merging the teacher and student model into the same Program
- **Paddle large-scale scalable knowledge distillation framework Pantheon:** a universal solution for knowledge distillation, more flexible than the naive knowledge distillation, and easier to scale to the large-scale applications.
  - Decouple the teacher and student models --- they run in different processes in the same or different nodes, and transfer knowledge via TCP/IP ports or local files;
  - Friendly to assemble multiple teacher models and each of them can work in either online or offline mode independently;
  - Merge knowledge from different teachers and make batch data for the student model automatically;
  - Support the large-scale knowledge prediction of teacher models on multiple devices.

### Neural Architecture Search

- Neural architecture search ==based on evolution strategy==.
- Support ==distributed search==.
- ==One-Shot== neural architecture search.
- Support FLOPs and latency constrained search.
- Support the latency estimation on different hardware and platforms.

---

如今，诸如计算机视觉、智能语音交互等基于深度学习的AI技术，在满足日常应用层面上已经非常成熟。比如，人脸识别闸机在机场、地铁、办公楼、学校等地方随处可见。什么都不用带，只要刷个脸就行，方便快捷又省事！

当有人经过闸机时，可以在0.1-0.3秒内完成人脸实时跟踪，并在0.2秒内完成高安全性的静默活体检测及人脸比对，如此高效的响应速度，你知道是怎么做到的吗？

目前深度学习在各个领域轻松碾压传统算法，不过真正用到实际项目中却面临两大问题：计算量巨大；模型占用很高的内存（深度学习模型可能会高达几百M）。

为了更好地应对这些实际业务需求，解决终端系统资源有限等问题，百度深度学习平台飞桨(PaddlePaddle)对端侧推理引擎Paddle Lite进行了新一轮升级，v2.3版本正式全新上线！

Paddle Lite v2.3新功能包括：

- 支持“无校准数据的训练后量化”方法，模型压缩高达75%；
- 优化网络结构和OP，ARM CPU推理速度最高提升超20%；
- 简化模型优化工具操作流程，支持一键操作，用户上手更容易。

此次升级带来了以下几个方面的变化：



 **支持“无校准数据的训练后量化”方法** 

 **模型压缩高达75%** 

在手机等终端设备上部署深度学习模型，通常要兼顾推理速度和存储空间。一方面要求推理速度越快越好，另一方面要求模型更加的轻量化。为了解决这一问题，模型量化技术尤其关键。

模型量化是指使用较少比特数表示神经网络的权重和激活，能够大大降低模型的体积，解决终端设备存储空间有限的问题，同时加快了模型推理速度。将模型中特定OP权重从FP32类型量化成INT8/16类型，可以大幅减小模型体积。经验证，将权重量化为INT16类型，量化模型的体积降低50%；将权重量化为INT8类型，量化模型的体积降低75%。 

Paddle Lite结合飞桨量化压缩工具PaddleSlim，为开发者提供了三种产出量化模型的方法：量化训练、有校准数据的训练后量化和无校准数据的训练后量化。

==其中“无校准数据的训练后量化”是本次Paddle Lite新版本重要新增内容之一==。



![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmNjwYZ02A6cWzyJVMcZ8njPTDYkf90VmSWicLGVrxGw0XE3Vp2CLOpHg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​															图1三种产出量化模型方法的处理示意图

“无校准数据的训练后量化”方法，在维持精度几乎不变的情况下，不需要样本数据，对于开发者来说使用更简便，应用范围也更广泛。

当然，如果希望同时减小模型体积和加快模型推理速度，开发者可以尝试采用PaddleSlim“有校准数据的训练后量化”方法和“量化训练”方法。

PaddleSlim除了量化功能以外，还集成了模型压缩中常用的剪裁、蒸馏、模型结构搜索、模型硬件搜索等方法。更多详细的介绍，请参见Github：

> https://github.com/PaddlePaddle/PaddleSlim

下面以MoblieNetV1、MoblieNetv2和ResNet50模型为例，介绍本方法所获得的效果。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmlMO0DL7Lv5wnljI2dibcQj4QiaibcjTXYibH7Xv61icnLPJfDvD8gUSibVsw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​										图2 “无校准数据的训练后量化”方法产出的量化模型体积对比图

由图2可知，INT16格式的量化模型，相比FP32，模型体积降低50%；INT8 格式的量化模型，相比FP32，模型体积降低75%。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmUOqhSZZ0OFib34CtWq4d2oNyArzJG4ZQI2mqQpukCGW5OeTPYDFEuSw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​										图3 “无校准数据的训练后量化”方法产出的量化模型准确率对比图

由图3可知，INT16格式的量化模型，相比FP32，准确率不变；INT8格式的量化模型，相比FP32，准确率仅微弱降低。



 **ARM CPU推理速度最高提升超20%** 

Paddle Lite v2.3在ARM CPU性能优化方面的主要更新包括：

- 针对Kernel Size为3*3的Conv，实现Winograd方法，包括F(6,3)和F(2, 3)。因Winograd相比普通算法从计算量上有大幅减少，该实现可以明显提升有相关OP的模型性能，比如ResNet50和SqueezeNet；
- 针对Conv后激活为Relu6 或是LeakyRelu的模型，==添加Conv+Relu6/LeakyRelu 融合==，从而可以减少单独的激活函数需要的访存耗时；
- 针对PaddlePaddle1.6 OP升级，如支持任意Padding的Conv和Pooling，Paddle Lite增加相关支持。该工作使得Tensorflow模型转换时，==一个Tensorflow Conv 对应一个Paddle Conv, 而非Padding+Conv 两个OP==，从而可以提升Tensorflow模型的推理性能。

图4给出了Caffe框架的MobileNetV1、MobileNetV2 和ResNet50三个模型在Paddle Lite，NCNN和MNN框架上的推理时延对比图。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0Ym9QGX84RmF0f5mZlWrkrFzGDCIoIRI7lItVT3J7MicCQIJ0opKoBMFZA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​															图4 Caffe框架模型的推理时延对比

由图4可知，Paddle Lite性能整体优于其他框架。如ResNet50模型，在高通骁龙845上，Paddle Lite相比其他框架，比MNN快10.259%，比NCNN快17.094%。

对于ONNX 公开模型如ShuffleNet、SqueezeNet和ResNet50, 在Paddle Lite、MNN和NCNN框架进行推理时延对比，其结果如图5所示。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmZ7lkrvQvaex9K3S42aQCqK4NMUaToncic9ECS3MyZwmKz0KFDyia3dDg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​															图5 ONNX框架模型的推理时延对比

由图5可知，Paddle Lite性能整体优于其他框架。如ShuffleNet模型，在高通骁龙845上，Paddle Lite相比其他框架，比MNN快21.185%，比NCNN快26.36%。 

Tensorflow公开模型，比如MnasNet、 MobileNetV1和ResNet101，Paddle Lite与MNN推理框架在推理时延性能指标上进行对比，结果如图6所示。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmKjQXInGicbreuNfibtMBvsSal1Lgt0TpaSZfDicicBCxicTntTBaKYc13aA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

​																图6 Tensorflow框架模型的推理时延对比

由图6可知，Paddle Lite性能整体优于MNN框架。如MnasNet模型，在高通骁龙855上，Paddle Lite比MNN快12.06%；在高通骁龙845上，Paddle Lite比MNN快18.91%；在高通骁龙835上，Paddle Lite比MNN快18.61%。

新版本更详细的性能数据，请参见GitHub的Benchmark。



 **简化模型优化工具操作流程** 

 **支持一键操作，用户上手更容易** 

对于第三方来源（Tensorflow、Caffe、ONNX）模型，一般需要经过两次转化才能获得Paddle Lite的优化模型。先使用x2paddle工具将第三方模型转化为PaddlePaddle格式，再使用模型优化工具转换为Padde Lite支持的模型。同时，转换后的Paddle Lite模型，通常包括模型结构和参数两个文件。操作繁琐，用户体验不太好。

针对上述问题，Paddle Lite v2.3对原模型优化工具model_optimize_tool 进行了升级，推出版模型优化工具——opt。opt包括以下三个亮点：

- 提供一键式脚本(auto_transformer.sh)，支持一键完成从各类框架模型到Paddle Lite模型（含OP算子融合、内存复用等优化操作）的所有优化处理操作；
- 优化后的模型最终只生成一个.nb文件，此文件包含模型网络结构和参数信息。同时提供加载模型.nb文件的API接口：set_model_from_file(nb_path)，接口的具体内容请见【Model Load API】。原有的模型加载方式仍然支持；
- 提供丰富的日志信息，比如支持查看某个模型用到哪些算子；还支持查看Paddle Lite支持哪些硬件，以及这些硬件分别支持哪些算子（如图7所示），进而了解Paddle Lite对模型的支持情况。

![img](https://mmbiz.qpic.cn/mmbiz_png/v1LbPPWiaSt78cuEx9P4nzvajeHCIK0YmlibqphsS6pfP1Tehl9EhuunDz43oXOHNtByC59RIVWcU2htH418CVsg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

关于opt的更详细介绍，请参见GitHub的opt工具介绍与使用。



 **其他升级** 

**1.文档官网升级**

为了提高文档可读性、改善文档的视觉效果、方便用户快速查找文档并轻松上手使用Paddle Lite，对Paddle Lite文档进行了全面升级。文档目录清晰可见，搜索功能更强大、为用户提供了更好的阅读体验。

同时，Paddle Lite v2.3完善了部分文档内容，并新增一些使用文档，如“有校准数据的训练后量化方法”、“无校准数据的训练后量化方法”使用文档等。

**2.Paddle Lite Demo仓库的案例升级**

对现有Paddle Lite Demo仓库的案例进行了内容升级，并新增了Demo。例如在Android Demo中，新增人脸检测（face-detection）Demo、YOLOv3目标检测Demo和人像分割（Human-Segment）Demo。用户可以方便地根据Demo进行实验并参考实现新应用的开发。另外，在Paddle Lite仓库下的CXX Demo库，新增了口罩识别案例，为此次疫情做些力所能及的贡献。感兴趣的小伙伴们可以在Paddle Lite仓库下载口罩识别Demo，进行实验。

同时，为了提高API接口易用性，升级了C++ API接口和Java API接口。在Java API接口，新增设置和返回数据类型，以支持不同类型的输入。如果您想了解更多关于Paddle Lite的相关内容，请参阅以下文档。



Paddle Lite的Github链接：

> https://github.com/PaddlePaddle/Paddle-Lite

Paddle Lite的文档链接：

> https://paddle-lite.readthedocs.io/zh/latest/index.html

Paddle Lite Demo的链接：

> https://github.com/PaddlePaddle/Paddle-Lite-Demo

Paddle Lite口罩识别Demo：

> https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/cxx

Benchmark：

> https://paddle-lite.readthedocs.io/zh/latest/benchmark/benchmark.html

auto_transformer.sh：

> https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.3/lite/tools/auto_transform.sh

Model Load API：

> https://paddle-lite.readthedocs.io/zh/latest/api_reference/cxx_api_doc.html#set-model-from-file-model-dir

opt工具介绍与使用：

> https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html

PaddlePaddle的Github链接：

> https://github.com/paddlepaddle

如果使用过程中遇到任何问题，大家可通过Paddle Lite官方QQ群与开发人员进行技术交流及问题反馈。