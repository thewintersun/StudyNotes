## ATC工具使用指导

地址： https://support.huaweicloud.com/tg-Inference-cann/atlasatc_16_0002.html

本文介绍如何将开源框架的网络模型（如Caffe、TensorFlow等）以及单算子Json文件，通过ATC（Ascend Tensor Compiler）将其转换成昇腾AI处理器支持的离线模型，模型转换过程中可以==实现算子调度的优化、权值数据重排、内存使用优化==等，可以==脱离设备完成==模型的预处理。

#### 工具功能架构

ATC工具功能架构如图1所示。从图1中可以看出，用户可以将原始模型通过ATC工具转换成适配昇腾AI处理器的离线模型，也可以将转换后的离线模型转成json文件，方便文件查看。用户也可以直接将原始模型文件通过ATC工具转成json文件。

![1614047895039](D:\Notes\raw_images\1614047895039.png)

图1 ATC工具功能架构

#### 工具运行流程

使用ATC工具进行模型转换的总体流程如图2所示。

![1614047915680](D:\Notes\raw_images\1614047915680.png)

详细流程说明如下：

1. 使用ATC工具之前，请先在开发环境安装ATC，获取相关路径下的ATC工具，详细说明请参见获取ATC工具。
2. 准备要进行转换的模型或单算子json文件，并上传到开发环境，详细说明请参见转换样例。
3. 使用ATC工具进行模型转换，在配置相关参数时，根据实际情况选择是否进行AIPP配置。

> AIPP是昇腾AI处理器提供的硬件图像预处理模块，包括色域转换，图像归一化（减均值/乘系数）和抠图（指定抠图起始点，抠出神经网络需要大小的图片）等功能。DVPP模块输出的图片多为对齐后的YUV420SP类型，不支持输出RGB图片。因此，业务流需要使用AIPP模块转换对齐后YUV420SP类型图片的格式，并抠出模型需要的输入图片。

#### 关键概念

- AIPP

  AIPP（AI Pre Process）是昇腾AI处理器提供的==硬件图像预处理模块==，包括==色域转换==，==图像归一化（减均值/乘系数）==和==抠图（指定抠图起始点，抠出神经网络需要大小的图片）==等功能。

- YUV420SP

  有损图像颜色编码格式，常用为YUV420SP_UV、YUV420SP_VU两种格式。

- 知识库

  ==存储auto tune调优后的Schedule==，在后续算子编译中直接使用。

- cost model

  评估器，auto tune过程中如果没有命中知识库，则==通过cost model评估tiling空间中tiling的优劣，选择最优tiling数据==。

- 数据排布格式（format）

  在深度学习框架中，多维数据通过多维数组存储，比如卷积神经网络的特征图用四维数组保存，四个维度分别为批量大小（Batch,N）、特征图高度（Height,H）、特征图宽度（Width,W）以及特征图通道（Channels,C）。

  由于数据只能线性存储，因为这四个维度有对应的顺序。不同深度学习框架会按照不同的顺序存储特征图数据，比如Caffe，排列顺序为[Batch,Channels,Height,Width]，即NCHW。Tensorflow中，排列顺序为[Batch,Height,Width,Channels]，即NHWC。

  如图3所示，以一张格式为RGB的图片为例，NCHW实际存储的是“RRRRRRGGGGGGBBBBBB”，同一通道的所有像素值顺序存储在一起，而NHWC实际存储的则是“RGBRGBRGBRGBRGBRGB”，多个通道的同一位置的像素值顺序存储在一起。

  图3 NCHW和NHWC
  ![img](https://support.huaweicloud.com/tg-Inference-cann/figure/zh-cn_image_0280770995.png)

  昇腾AI软件栈中，为了提高数据访问效率，所有==张量数据统一采用NC1HWC0的五维数据格式==。其中C0与微架构强相关，等于AI Core中矩阵计算单元的大小，对于==FP16类型为16==，对于==INT8类型则为32==，这部分数据需要连续存储；==C1=(C+C0-1)/C0。如果结果不整除，取上整数==。

  例如，将NHWC -> NC1HWC0的转换过程为：

  1. ==将NHWC数据在C维度进行分割，变成C1份NHWC0==。
  2. 将C1份NHWC0在内存中连续排列，由此变成NC1HWC0。

  NHWC->NC1HWC0的转换场景示例：

  1. 首层RGB图像通过AIPP转换为NC1HWC0格式。
  2. ==中间层Feature Map每层输出为NC1HWC0格式，在搬运过程中需要重排==。
