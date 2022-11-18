# TBE算子开发框架

#### 什么是TVM

随着深度学习的广泛应用，大量的深度学习框架及深度学习硬件平台应运而生，但不同平台的神经网络模型难以在其他硬件平台便捷的运行，无法充分利用新平台的运算性能。TVM（Tensor Virtual Machine）的诞生解决了以上问题，它是一个开源深度学习编译栈，它通过统一的中间表达（Intermediate Representation）堆栈连接深度学习模型和后端硬件平台，通过统一的结构优化Schedule，可以支持CPU、GPU和特定的加速器平台和语言。

TVM的架构详细介绍请参考https://tvm.apache.org/。

#### TBE简介

TBE（Tensor Boost Engine）提供了基于TVM框架的自定义算子开发能力，通过TBE提供的API和自定义算子编程开发界面可以完成相应神经网络算子的开发。

TBE的逻辑架构如图1所示。

TBE工具给用户提供了多层灵活的算子开发方式，用户可以根据对硬件的理解程度自由选择，利用工具的优化和代码生成能力，生成昇腾AI处理器的高性能可执行算子。

图1 TBE在软件栈中的逻辑架构图
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0228528864.png)

- 前端框架：包含第三方开源框架Tensorflow(TF：Google机器学习开源框架)、Caffe（快速特征嵌入的卷积结构）。
- 图引擎（Graph Engine：GE)：Graph Engine是华为基于昇腾AI处理器软件栈对不同的机器学习框架提供统一的IR接口，对接上层网络模型框架，例如Tensorflow、Caffe，GE的主要功能包括图准备、图拆分、图优化、图编译、图加载、图执行和图管理等（此处图指网络模型拓扑图）。
- 融合引擎（Fusion Engine：FE）：FE负责对接GE和TBE算子，具备算子信息库的加载与管理、融合规则管理、原图融合和子图优化的能力。GE在子图优化阶段将子图传递给FE，FE根据算子信息库以及FE融合优化进行预编译，例如修改数据类型、插入转换算子等，该子图将再次传递给GE进行子图合并及子图优化。
- 张量加速引擎（TBE）：TBE通过IR定义为GE的图推导提供必要的算子信息，通过算子信息库和融合规则为FE提供子图优化信息和TBE算子调用信息，TBE生成的二进制对接昇腾AI处理器，最终生成网络在昇腾AI处理器上的执行任务。

#### TBE开发交付件和编译流程

算子开发完成后在昇腾AI处理器硬件平台上的运行的架构如图2所示。

图2 TBE算子运行架构图
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0228528866.png)

上图中黄色底纹的图示为开发人员在自定义算子开发时需要实现的交付件。

**算子适配插件：**基于第三方框架（Tensorflow/Caffe）进行自定义算子开发的场景，开发人员完成自定义算子的实现代码后，需要进行适配插件的开发将基于第三方框架的算子映射成适昇腾AI处理器的算子，将算子信息注册到GE中。基于第三方框架的网络运行时，首先会加载并调用算子适配插件信息，将第三方框架网络中的算子进行解析并映射成昇腾AI处理器中的算子。

**算子原型库：**算子原型定义规定了在昇腾AI处理器上可运行算子的约束，主要体现算子的数学含义，包含定义算子输入、输出、属性和取值范围，基本参数的校验和shape的推导。网络运行时，GE会调用算子原型库的校验接口进行基本参数的校验，校验通过后，会根据原型库中的推导函数推导每个节点的输出shape与dtype，进行输出tensor的静态内存的分配。

**算子信息库：**算子信息库主要体现算子在昇腾AI处理器上物理实现的限制，包括算子的输入输出dtype、format以及输入shape信息。网络运行时，FE会根据算子信息库中的算子信息做基本校验，判断是否需要为算子插入合适的转换节点，并根据算子信息库中信息找到对应的算子实现文件进行编译，生成算子二进制文件进行执行。

**算子实现：** 算子实现的python文件，包含算子的计算实现及Schedule实现。

加载TBE算子进行模型转换的详细流程如下图所示。

图3 加载TBE算子进行模型转换
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0228528868.png)

1. 将原始第三方网络模型（Tensorflow/Caffe）下发给GE。

   注：网络模型的拓扑图后续简称为图。

2. GE调用算子插件，将Tensorflow/Caffe网络模型中的算子映射为适配昇腾AI处理器的算子，从而将原始Tensorflow/Caffe图解析为适配昇腾AI处理器的图。

3. 调用算子原型库校验接口进行基本参数的校验，校验通过后，会根据原型库中的推导函数推导每个节点的输出shape与dtype，进行输出tensor的静态内存的分配。

4. GE向FE发送图优化请求，并将图下发给FE，FE匹配融合规则进行图融合，并根据fe.ini中的配置进行算子选择，选择优先级最高的算子类型进行算子匹配（默认自定义算子优先级最高），最后将优化后的整图返回给GE。

5. GE根据图中数据将图拆分为子图并下发给FE，FE首先在子图内部插入转换算子，然后按照当前子图流程进行TBE算子预编译，对TBE算子进行UB融合，并根据算子信息库中算子信息找到算子实现将其编译成算子kernel（算子的*.o与*.json），最后将优化后子图返回给GE。

6. GE进行图编译，包含内存分配、流资源分配等，并向FE发送tasking请求，FE返回算子的taskinfo信息给GE，图编译完成之后生成适配昇腾AI处理器的离线模型文件（*.om）。

**![img](https://res-img2.huaweicloud.com/content/dam/cloudbu-site/archive/china/zh-cn/support/resource/framework/v3/images/support-doc-new-notice.svg)须知：**

调试工具（例如：objdump、ld、clang工具等）为产品业务运行需要，根据网络结构编译TBE算子时需要调用编译器在线编译。具体流程如下：GE在子图优化阶段将子图传递给FE，FE根据算子信息库以及FE融合优化进行预编译，例如修改数据类型、插入换算子等，该子图将再次传递给GE进行子图合并及子图优化，GE调用构建网络。

#### TBE功能框架

TBE内部包含了特性域语言（Domain-Specific Language，DSL）模块、调度（Schedule）模块、中间表示（Intermediate Representation，IR）模块、编译优化（Pass）模块以及代码生成（CodeGen）模块如图4所示。

图4 TBE功能框架
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0242852740.png)

- DSL模块：面向开发者，提供算子逻辑的编写的接口（Compute接口），使用接口来编写算子。
- Schedule模块：用于描述指定shape下算子如何在昇腾AI处理器上进行切分，包括Cube类算子的切分、Vector类算子的切分，它们仍然使用的是社区提供的调度原语来描述。
- IR模块：借用社区的IR来表示的，包括IR变形、AST树的维护等功能。
- 编译优化（Pass）：对生成的IR进行编译优化，优化的方式有双缓冲（Double Buffer）、流水线（Pipeline）同步、内存分配管理、指令映射、分块适配矩阵计算单元等。
- 代码生成模块（CodeGen）：CodeGen生成类C代码的临时文件，这个临时代码文件可以通过编译器生成算子的实现文件，可被网络模型直接加载调用。

#### 算子开发流程

TBE算子开发分为计算逻辑编写和调度开发：

1. TBE算子开发分为计算过程的编写与调度开发，TBE提供直接基于特定域语言编写（DSL）以及TIK开发算子的计算过程和调度过程。算子计算过程描述指明算子的计算方法和步骤，而调度过程描述完成数据切块和数据流向的规划。算子每次计算都按照固定数据形状进行处理，这就需要提前针对在昇腾AI处理器中的不同计算单元上执行的算子进行数据形状切分，如矩阵计算单元、向量计算单元以及AI CPU上执行的算子对输入数据形状的需求各不相同。
2. 在完成算子的基本实现过程定义后，需要启动调度模块中分块（Tiling）子模块，对算子中的数据按照调度描述进行切分，同时指定好数据的搬运流程，确保在硬件上的执行达到最优。除了数据形状切分之外，TBE的算子融合和优化能力也是由调度模块中的融合（Fusion）子模块提供的。
3. 算子编写完成后，需要生成中间表示来进一步优化，而中间表示模块通过类似于TVM的IR（Intermediate Representation）格式来进行中间表示的生成。在中间表示生成后，需要将模块针对各种应用场景进行编译优化，优化的方式有双缓冲（Double Buffer）、流水线（Pipeline）同步、内存分配管理、指令映射、分块适配矩阵计算单元等。
4. 在算子经过编译器传递模块处理后，由==CodeGen生成类C代码的临时文件==，这个临时代码文件可以通过编译器生成算子的实现文件，可被网络模型直接加载调用。

综上所述，一个完整的自定义算子TBE中的子模块完成整个开发流程，从基于DSL或者TIK提供算子计算逻辑和调度描述，构成算子原型后，由调度模块进行数据切分和算子融合，进入中间表示模块，生成算子的中间表示。编译器传递模块以中间表示进行内存分配等编译优化，最后由代码生成模块产生类C代码可供编译器直接编译。张量加速引擎在算子的定义过程不但完成了算子编写，而且还完成了相关的优化，提升了算子的执行性能。

### TBE算子开发方式

昇腾AI软件栈提供了TBE（Tensor Boost Engine：张量加速引擎）算子开发框架，开发者可以基于此框架使用Python语言开发自定义算子，通过TBE进行算子开发有以下几种方式：

- [DSL（Domain-Specific Language）开发](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0063.html)

  为了方便开发者进行自定义算子开发，昇腾AI软件栈借鉴了TVM中的TOPI机制，预先提供一些常用运算的调度，封装成一个个运算接口，称为基于TBE DSL开发。开发者只需要利用这些特定域语言声明计算的流程，再使用自动调度（Auto Schedule）机制，指定目标生成代码，即可进一步被编译成专用内核。

- [TIK开发](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0077.html)

  TIK（Tensor Iterator Kernel）是一种基于Python语言的动态编程框架，呈现为一个Python模块，运行于Host CPU上。开发者可以通过调用TIK提供的API基于Python语言编写自定义算子，即TIK DSL，然后TIK编译器会将TIK DSL编译为昇腾AI处理器应用程序的二进制文件。

  基于TIK的自定义算子开发，提供了对Buffer的管理和数据自动同步机制，需要开发者对Davinci架构有一定的了解，但算子的schedule需要开发人员自己规划。

整个计算过程可以表现为多个输入张量经过一个计算节点得到多个张量的过程。TIK方式、TBE DSL的开发流程本质上是一样的，只不过开发的抽象层次不一样而已。

| 参数     | TBE DSL方式                                                  | TIK方式                                                      |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 语言     | Python                                                       | Python                                                       |
| 运用场景 | 常用于各种算术逻辑简单向量运算，或内置支持的矩阵运算及池化运算，例如elewise类操作。 | 适用各类算子的开发，对于无法通过lambda表达描述的复杂计算场景也有很好的支持，例如排序类操作。 |
| 入门难度 | 较低                                                         | 较高                                                         |
| 适用人群 | 入门开发者，需要了解NN、TBE DSL相关知识。                    | 高级开发者，需要了解NN，深入理解昇腾AI处理器架构、指令集、数据搬运等相关知识。 |
| 特点     | TBE DSL接口已高度封装，开发者仅需要使用DSL接口完成计算过程的表达，后续的Schedule创建、优化及编译都可通过已有接口一键式完成。 | 入门难度高，程序员直接使用TIK提供的API完成计算过程的描述及Schedule过程，需要手工控制数据搬运的参数和Schedule。开发者无需关注Buffer地址的分配及数据同步处理，由TIK工具进行管理。 |
| 不足     | 某些场景下性能可能较低，复杂算子逻辑无法支持表达。           | TIK对数据的操作更加灵活，但需要手工控制数据搬运的参数和Schedule过程。代码编写接近底层硬件架构，过程优化等基于特定硬件特性。 |

**![img](https://res-img2.huaweicloud.com/content/dam/cloudbu-site/archive/china/zh-cn/support/resource/framework/v3/images/support-doc-new-notice.svg)须知：**

- 当前仅支持用户开发Vector算子，由于开发高性能Cube算子难度较大，暂由华为交付。
- 当前暂不开放Unified Buffer与L1 Buffer之间的通路。

#### DSL介绍

为了方便开发者进行自定义算子开发，TBE（Tensor Boost Engine）提供了一套计算接口供开发者用于组装算子的计算逻辑，使得70%以上的算子可以基于这些接口进行开发，极大的降低自定义算子的开发难度。TBE提供的这套计算接口，称之为DSL（Domain-Specific Language）。基于DSL开发的算子，可以直接使用TBE提供的Auto Schedule机制，自动完成调度过程，省去最复杂的调度编写过程。

TBE DSL算子的功能框架如图1所示。

图1 TBE DSL功能框架
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0242854499.png)

1. 开发者调用TBE提供的DSL接口进行计算逻辑的描述，指明算子的计算方法和步骤。
2. 计算逻辑开发完成后，开发者可==调用Auto Schedule接口==，启动自动调度，自动调度时TBE会根据计算类型自动选择合适的调度模板，完成数据切块和数据流向的划分，确保在硬件执行上达到最优。
3. Auto Schedule调度完成后，会生成类似于TVM的==IR（Intermediate Representation）==的中间表示。
4. 编译优化（Pass）会对算子生成的IR进行编译优化，优化的方式有==双缓冲（Double Buffer）、流水线（Pipeline）同步、内存分配管理、指令映射、分块适配矩阵计算单元==等。
5. 算子经Pass处理后，由CodeGen生成类==C代码的临时文件==，这个临时代码文件可通过编译器生成算子的实现文件，可被网络模型直接加载调用。

代码示例如下所示：

```python
    //初始化输入tensor，为输入tensor进行占位
    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)
   //调用计算接口实现data_x + data_y
    res = te.lang.cce.vadd(data_x, data_y)
   //调用auto_schedule接口实现自动调度
    with tvm.target.cce():
        schedule = topi.generic.auto_schedule(res)
    //配置编译参数并进行编译
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
```

#### 计算接口列表

TBE DSL提供的计算接口主要涵盖向量运算，包括Element-wise类操作接口、Reduction 操作接口、Broadcast操作接口、Index操作接口、Concat操作接口、卷积接口、4D/5D互转接口、矩阵计算接口等。

您可以在ATC的安装目录下的“/python/site-packages”目录下查看接口的定义文件。

| 计算接口分类   | 计算接口描述                                                 | 参考                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Element-wise   | 对Tensor中每个原子值分别做相同操作，例如te.lang.cce.vadd 即对两个数值x，y求和。 | [Element-wise计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0003.html) |
| Reduction      | 对Tensor按轴进行操作，例如te.lang.cce.sum(data,axis,keepdims=False)表示对data按axis进行累加。 | [Reduction计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0036.html) |
| Broadcast      | 对Tensor按照目标shape进行广播，如shape为（3,1,2）的Tensor广播成(3,3,2)的Tensor。 | [Broadcast计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0042.html) |
| Segment        | 对Tensor进行分段操作。                                       | [Segment计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0044.html) |
| Inplace        | 对Tensor进行按行计算。                                       | [Inplace计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0050.html) |
| 取整计算接口   | cast类型计算接口，对输入tensor中的每个元素按照一定的规则进行取整操作。 | [取整计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0054.html) |
| Concat         | 在指定轴上对输入的多个Tensor进行重新连接。                   | [Concat计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0059.html) |
| 卷积           | 在给定5HD格式的Data和FracZ格式的Weight的情况下计算float16的2-D卷积。 | [卷积计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0061.html) |
| 4D/5D互转      | 4-D **“NCHW”**数据格式与5-D **“NC1HWC0”**数据格式间相互转换。 | [4D/5D互转接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0063.html) |
| Matmul         | 矩阵乘计算。                                                 | [Matmul计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0066.html) |
| pooling2d      | 通过不同的池化方式，对tensor_in上不同的滑动窗口进行信号的采样。 | [Pooling2d计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0068.html) |
| common计算接口 | 对输入数据进行round_to操作或者数据类型转换操作。             | [Common计算接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0070.html) |
| 编译接口       | 包含调度及编译等接口。                                       | [编译接口](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_07_0073.html) |

#### Auto Schedule

基于TBE DSL编写一个算子，就是通过组合DSL接口表达算子的计算逻辑，然后调用Auto Schedule进行算子的自动调度，完成数据切块和数据流向的划分。==Auto Schedule机制是TBE底层的默认Schedule调优机制==，开发者无法在算子开发代码过程中进行控制，下面简要介绍Auto Schedule的原理。

如下是基于DSL进行算子开发的示例，实现对x取指数，然后在轴0上进行累加降维，再取倒数的功能。

```python
x = tvm.placeholder((512, 1024), "float16")
exp_x = te.lang.cce.vexp(x)
reduce_exp_x = te.lang.cce.sum(exp_x, axis = 0)
res = te.lang.cce.vrec(reduce_exp_x)

with tvm.target.cce():
    sch = topi.generic.auto_schedule(res)
```

开发者调用**topi.generic.auto_schedule()**接口开启TBE的自动调度，自动调度的总体流程如[图1](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0019.html#ZH-CN_TOPIC_0279010563__zh-cn_topic_0228422170_fig156851416102110)所示。

图1 Auto Schedule总体流程
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0242854300.png)

1. 调用Auto Schedule接口的时候，就是传递了一个compute的语法树，TBE中每一个compute语句在进行编译的时候都会被加上tag_scope标志，如下所示。

   添加tag_scope的操作如下：

   ```
   with tvm.tag_scope(op):
       tmp = tvm.compute(shape, lambda_func, name=name)
   ```

   如图2所示，左侧的compute语法树，也叫==抽象语法树（Abstract Syntax Tree: AST）==，编译过程中，会对每一个compute语句加上tag_scope标志。

   图2 计算语句与tag_scope标志对应示例
   ![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0242854304.png)

2. 根据scope标识，识别出对应pattern，TBE当前支持的pattern类型有：elewise、reduce、segment、concat、conv、depthwise、pooling2d等。TBE会按照pattern规则对AST进行切分，例如最简单的一条pattern规则是elewise可以和其他的pattern连在一起，reduce、segment、concat不能在一个AST子图内。

3. 完成AST子图切分后，TBE会创建并初始化Schedule对象。

4. Schedule执行过程中==首先找到AST子图的边界==，然后对每一个子图==根据其pattern选择一个合适的Schedule模板进行调度==。调度过程主要包括==数据流管理、tiling以及指令映射==等。

#### TIK介绍

TIK（Tensor Iterator Kernel）是一种基于Python语言的动态编程框架，呈现为一个Python模块，运行于Host CPU上。开发者可以通过调用TIK提供的API基于Python语言编写自定义算子，然后TIK编译器会编译为昇腾AI处理器应用程序的二进制文件。

#### TIK的优势

TIK算子开发方式是一种灵活的开发方式。TIK代码在算子开发效率和算子性能自动优化上有着一定的优势：

1. ==自动内存分配和自动数据依赖规划==，让用户用串行编程的思路既可以写出高性能的并行计算的算子。
2. 通过==手动调度==可以更加精确的控制数据搬运和计算流程，从而实现更高的性能，将昇腾AI处理器的能力发挥到极致。

#### TIK算子开发流程

基于TIK API编写Python程序的通用步骤，如图1所示。

图1 算子实现流程
![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0255054314.png)

主要步骤为：

1. Python模块导入。

   ```python
   from te import tik↵
   ```

   “te.tik”：提供了所有TIK相关的python函数，具体请参考ATC安装目录下的“python/site-packages/te/tik”。

2. 构建TIK DSL容器。

   ```python
   from te import tik tik_instance = tik.Tik()
   ```

3. 在AI Core的外部存储和内部存储中定义输入数据、输出数据。

   ```python
   data_A = tik_instance.Tensor("float16", (128,), name="data_A", scope=tik.scope_gm)
   data_B = tik_instance.Tensor("float16", (128,), name="data_B", scope=tik.scope_gm)
   data_C = tik_instance.Tensor("float16", (128,), name="data_C", scope=tik.scope_gm)
   
   data_A_ub = tik_instance.Tensor("float16", (128,), name="data_A_ub", scope=tik.scope_ubuf)
   data_B_ub = tik_instance.Tensor("float16", (128,), name="data_B_ub", scope=tik.scope_ubuf)
   data_C_ub = tik_instance.Tensor("float16", (128,), name="data_C_ub", scope=tik.scope_ubuf)
   ```

4. 将外部存储中的数据搬入AI Core内部存储（比如Unified Buffer）中。

   ```python
   tik_instance.data_move(data_A_ub, data_A, 0, 1, 128 //16, 0, 0)
   tik_instance.data_move(data_B_ub, data_B, 0, 1, 128 //16, 0, 0)
   ```

5. 进行计算。

   ```python
repeat = tik_instance.Scalar('int32')
   repeat.set_as(1)
tik_instance.vec_abs(128, data_C_ub[0], data_A_ub[0], data_B_ub[0], repeat, 8, 8, 8)
   ```

6. 搬出到外部存储。

   ```python
tik_instance.data_move(data_C, data_C_ub, 0, 1, 128 //16, 0, 0)
   ```

7. 将TIK DSL容器中的语句，编译成昇腾AI处理器可执行的代码。

   ```python
#其中kernel_name决定了算子昇腾AI处理器上可执行的二进制文件的名称，inputs为从外部存储中加载的数据，outputs对应计算后搬运到外部存储中的数据。
   tik_instance.BuildCCE(kernel_name="simple_add",inputs=[data_A,data_B],outputs=[data_C])
   ```
   
   

