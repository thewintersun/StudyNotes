## Single Path One-Shot Neural Architecture Search with Uniform Sampling

地址：https://arxiv.org/abs/1904.00420

作者：Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, Jian Sun

机构：清华大学，香港中文大学，旷世

文章地址: https://zhuanlan.zhihu.com/p/61144700



### 摘要

One-shot method is a powerful Neural Architecture Search (NAS) framework, but its training is non-trivial and it is difficult to achieve competitive results on large scale datasets like ImageNet. In this work, we propose a Single Path One-Shot model to address its main challenge in the training. 

Our central idea is to ==construct a simplified supernet==, Single Path Supernet, which is ==trained by an uniform path sampling method==. All underlying architectures (and their weights) get trained fully and equally. Once we have a trained supernet, we ==apply an evolutionary algorithm== to efficiently search the best-performing architectures ==without any fine tuning==. 

Comprehensive experiments verify that our approach is flexible and effective. It is easy to train and fast to search. It effortlessly supports complex search spaces (e.g., building blocks, channel, mixed-precision quantization) and different search constraints (e.g., FLOPs, latency). It is thus convenient to use for various needs. It achieves start-of-the-art performance on the large dataset ImageNet.

> 这是「旷视Model组」知乎专栏的第 2 篇，提出了一个单路径 One-Shot 模型，其核心思想是构建一个简化的超网络——单路径超网络（Single Path Supernet），该方法在大型数据集 ImageNet 上取得了当前最优结果。

### **导语**

一步法（One-Shot）是一个强大的神经网络模型搜索（Neural Architecture Search/NAS）框架，但是它的训练相对复杂，并且很难在大型数据集（比如 ImageNet）上取得较有竞争力的结果。

本文中，旷视研究院提出一个单路径 One-Shot 模型，以解决训练过程中面对的主要挑战，其核心思想是构建一个简化的超网络——单路径超网络（Single Path Supernet），这个网络按照==均匀的路径采样方法==进行训练。所有子结构（及其权重）获得充分而平等的训练。基于这个已训练的超网络，可以通过进化算法快速地搜索最优子结构，其中无需对任何子结构进行微调。

对比实验证明了这一方法的灵活性和有效性，不仅易于训练和快速搜索，并且可以轻松支持不同的复杂搜索空间（比如构造单元，通道数，混合精度量化）和搜索约束（比如 FLOPs，速度），从而便于满足多种需求。这一方法在大型数据集 ImageNet 上取得了当前最优结果。

### **简介**

深度学习终结了手工设计特征的时代，同时解决了权重优化问题。NAS 则旨在通过另一个方法——模型搜索（architecture search），终结人工设计架构。

早期的 NAS 方法使用嵌套式优化，从搜索空间采样出模型结构，接着从头训练其权重，缺点是对于大型数据集来讲计算量过大。新近的 NAS 方法则采用权重共享策略减少计算量。本文提出的超网络则包含所有子结构，只训练一次，所有子结构便可以直接从超网络获得其权重，无需从头训练。即使在大型数据集上，计算也十分高效。

大多数权重共享方法使用连续的搜索空间，将模型结构分布不断参数化。这些参数在超网络训练期间与网络权重进行联合优化。因此可以在优化时进行模型搜索，从分布中采样最优的架构。其公式化表示优雅而理论完备。但是存在两个问题：

第一，==超网络的权重深度耦合。目前尚不清楚为什么特定结构的复用权重（inherited weights）依然有效==。第二，==联合优化导致了模型参数和超网络权重的进一步耦合==。梯度方法的贪婪天性不可避免地在结构分布和超网络权重中引入偏差。这很容易误导模型搜索。精细地微调超参数和优化过程被用于先前方法之中。

One-shot 是一种新范式。它定义了超网络，并以相似的方式做权重复用。但是==并没有将模型结构分布参数化==。模型搜索从超网络训练中解耦，并且解决步骤是独立的。因此，One-shot ==具有序列性==。它结合了上述嵌套式和联合式优化方法的优点，因此灵活而高效。

尽管第二个问题已解决，现有 One-shot 并未很好地解决第一个问题。超网络的权重依然耦合。优化依然复杂，并包含敏感的超参数，导致在大型数据集上表现欠佳。

本文方法的动机旨在吸收 One-shot 的优点，克服其缺点。One-shot 成功的关键是使用复用权重的模型的精度可以用来预测从头训练模型的精度。因此，旷视研究院提出，超网络训练应是随机的。这样所有子结构的权重能够被同时且均匀地优化。

为减少超网络的权重耦合，旷视研究院提出==一个单路径超网络，在每次迭代训练中只有单路径结构被激活==。训练中不需要任何超参数来指导子结构的选择，采用均匀采样的方式，平等对待所有子结构。

本文方法简单而灵活，超网络训练中没有超参数。其简易性允许设计丰富的搜索空间，包括新设计通道单元和位宽单元。模型搜索过程十分高效，因为只需要基于超网络的权重进行前向计算。进化算法则用来轻松支持多种约束。

综合性消融实验及在大型数据集（ImageNet）上的实验证明了这一方法在精度、内存消耗、训练时间、模型搜索的有效性及灵活性方面都表现良好，达到了当前最优的结果。

### **本文 One-Shot NAS**

#### **One-Shot NAS方法回顾**

如上所述，耦合式模型搜索和权重优化是存在挑战和问题的。通过回顾发现，早期使用嵌套优化的 NAS 方法在于解决公式 (1) 和 (2) 的优化问题，这不禁引起思考，问题解耦和权重共享的优点是否可以兼得？

![img](https://pic3.zhimg.com/80/v2-7a27c78d34e0f0a5d409fd6eaedf6196_720w.png)

![img](https://pic3.zhimg.com/80/v2-ce45583b336676a3bff28b6bf546d642_720w.png)

这一考虑诞生了所谓的 One-shot 。这些方法依然只训练一次超网络，并允许各结构共享其中的权重。

但是，超网络训练及模型搜索作为先后次序的两个步骤是解耦的。请注意，这不同于嵌套优化或联合优化。

首先，超网络权重被优化为：

![img](https://pic2.zhimg.com/80/v2-38c68c89daec4e55cba0c5fdab0fc0ad_720w.png)

相比公式 (4) ：

![img](https://pic1.zhimg.com/80/v2-9cd5787e566021e761e16221a4b17b94_720w.png)

公式 (5) 已经不存在搜 索空间的连续参数化，只有网络权重被优化。

其次，搜索部分被表示为：

![img](https://pic1.zhimg.com/80/v2-783ef833e9bc2a62bd546ffc77ebbc34_720w.png)

公式 (6) 与公式 (1) 、 (2) 的最大区别是其权重是预先初始化的。评估 ![[公式]](https://www.zhihu.com/equation?tex=ACC_%7Bval%7D%28%5Ccdot%29)

仅需要推理。没有微调或者再训练。因此搜索非常**有效**。

通过借助进化算法，搜索同样非常**灵活。**像等式 (3) ![[公式]](https://www.zhihu.com/equation?tex=Lantency%28a%5E%7Bx%7D%29%5Cleq+Lat_%7Bmax%7D+++++%283%29)

对模型结构进行的约束可以精确地满足。并且一旦训练好一个超网络，可在同一超网络内基于不同约束（比如 100ms 和 200ms 延迟）重复搜索。这些特性是先前方法所缺失的，将使 One-Shot NAS 方法对实际任务更具吸引力。

但依然存在一个问题。在等式 (5) 中，超网络训练的图节点权重是耦合的，复用权重是否适用于任意子结构尚不清楚。

### **单路径超网络和均匀采样**

按照基本原理重新出发，会使 One-Shot 范式更有效。在等式 (5) 中，模型搜索成功的关键在于，在验证集中，使用复用权重 ![[公式]](https://www.zhihu.com/equation?tex=W_%7BA%7D%28a%29) （没有额外的微调）的任意子结构的精度是高度可信的。正如等式 (1) 是理想情况，需要权重 ![[公式]](https://www.zhihu.com/equation?tex=W_%7BA%7D%28a%29) 近似最优权重 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Ba%7D) 。近似的效果和训练损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Btrain%7D%28N%28a%EF%BC%8CW_%7BA%7D%28a%29%29%29) 被最小化的程度成正比。这推导出一个原则：超网络权重![[公式]](https://www.zhihu.com/equation?tex=W_%7BA%7D%28a%29) 的优化应当与搜索空间中所有子结构的优化**同时**进行。这可表示为：

![img](https://pic1.zhimg.com/80/v2-bf266755e562a57641627eedd65739f4_720w.png)

请注意，等式 (7) 是等式 (5) 的实现。在优化的每一步中，子结构是随机采样的，只有对应的权重 ![[公式]](https://www.zhihu.com/equation?tex=W_%7BA%7D%28a%29) 被激活和更新。这不仅节省内存空间，而且高效。由此，超网络本身不再是一个有效的网络，而变成一个随机的网络。

为减少节点权重之间的协同适应，旷视研究院提出==最大化简化搜索空间 ![[公式]](https://www.zhihu.com/equation?tex=A)==。它只包含单一路径架构，如图 1 所示。

![img](https://pic3.zhimg.com/80/v2-31b1113630741279d33a25b49f6faee6_720w.png)

​																			图 1：单一路径超网络架构图

==在每次训练时仅保留一个。不存在任何调优==。训练在本文实验中收敛良好。

先验分布 ![[公式]](https://www.zhihu.com/equation?tex=%5CGamma%28A%29) 很重要。旷视研究员通过实验发现，均匀采样已经足够好。这并不让人惊讶。这是因为实际任务常常期望有多个模型结构，以满足不同的约束。图 2 表明，==两个采样方法皆工作良好，均匀约束采样方法表现稍好，本文默认使用它==。

![img](https://pic3.zhimg.com/80/v2-62dc5d4e34641c6df70a4859c6f0fe52_720w.png)

​														图 2：不同采样策略的单路径超网络的进化模型搜索

本文注意到，在优化时根据结构分布采样一个路径已经出现在之前的权重共享方法之中，区别在于，在本文的训练中（等式(7)）分布 ![[公式]](https://www.zhihu.com/equation?tex=%5CGamma%28A%29) 是一个固定的先验，而在先前方法中，它是可学习和更新的（等式(4)），后者会使超网络权重和结构参数优化高度关联。

请注意，本文并未声明在训练时一个固定的先验分布天生优于优化分布。不存在这样的理论保证。本文更优的结果可能是受益于这一事实：当前优化技术的成熟度不足以满足等式 (4) 中的联合优化需求。

### **超网络与选择单元**

选择单元用于构建一个超网络。图 1 给出了一个实例。一个选择单元包含多个选择。对于本文提出的单路径超网络，每个选择单元一次只调用一个选择。一个路径的获得是通过随机采样所有选择单元实现的。

本文方法的简易性允许定义不同类型的选择单元，以搜索不同的结构变量。具体而言，旷视研究院提出两个全新的选择单元，以支持复杂的搜索空间。

**通道数搜索。**选择单元旨在搜索一个卷积层的通道数。其主要思想是预先分配一个带有最大通道数的权重张量。在超网络训练期间，系统随机选择通道数并分割出用于卷积的相应的子张量。详见图 4。

![img](https://pic1.zhimg.com/80/v2-f29b1bca4385df820e0c5f95ef9827ec_720w.png)

​																	图 4：用于通道数搜索的选择单元

**混合精度量化搜索。**选择单元可以搜索卷积层权重和特征的量化精度。在超网络训练中，特征图的位宽和和权重被随机选取。详见图 5。

![img](https://pic1.zhimg.com/80/v2-7fde7aaaa815bc8006f5996d6fbe5780_720w.png)

​																	图 5：用于混合精度量化搜索的选择单元

#### **基于进化算法的模型搜索**

针对等式 (6) 中的模型搜索，先前的 One-shot 工作使用随机搜索。这在大的搜索空间中并不奏效。因此，本文使用了进化算法，同时扬弃了从头开始训练每个子结构的缺点，只涉及推理部分，因此非常高效。详见算法 1。

![img](https://pic2.zhimg.com/80/v2-c228e5d6378f1686e061c2b86d1ad479_720w.png)

​																算法 1：基于进化算法的模型搜索

图 3 描绘了使用进化算法和随机搜索两种方法在进化迭代时的验证集精度。很明显==进化算法搜索更有效==。

![img](https://pic3.zhimg.com/80/v2-c22d78407a10335f9c8fae96b8db55fe_720w.png)

​																图 3：进化算法搜索对比随机搜索

进化算法可以灵活处理等式 (3) 的不同约束，因为变异和交叉过程是可控的，以产生满足约束条件的合适候选。

### **总结**

单路径超网络、均匀采样训练策略、基于进化算法的模型搜索、丰富的搜索空间设计，上述多种设计使得本文方法简单、高效和灵活。表 1 给出了本文方法与其他权重共享方法的一个全方位、多维度对比结果。

![1587352746014](D:\Notes\raw_images\1587352746014.png)

​														表 1：本文方法对比当前权重共享 SOTA 方法

### **实验**

所有实验是在 ImageNet 上进行的。验证集和测试集的设定遵从 Proxyless NAS [4]。对于超网络的训练，以及（进化搜索之后）最优模型结构的从头再训练，本文使用和 [17] 一样的设定。

#### **构造单元搜索**

构造单元（building block）的设计灵感来自手工设计网络的代表作——ShuffleNet v2。表 2 给出了超网络的整体架构。共有 20 个选择单元。

![img](https://pic3.zhimg.com/80/v2-069ab047c97c6cd6ed064d941b6e84d2_720w.png)

​																					表 2：超网络架构

表 3 给出了结果。为了对比，本文设置了一系列基线，如下：1）只选择一个特定的单元选择；2）从搜索空间中随机选择一些候选；3）使用随机搜索替代本文的进化算法模型搜索。

![img](https://pic2.zhimg.com/80/v2-9efb803c5aa9ece03969d6d7c20c0385_720w.png)

​																		表 3：构造单元搜索结果

#### **通道数搜索**

搜索卷积层的通道数非常有挑战，如图 4 所示，本文提出一个全新的选择单元用于通道数搜索，并首先在基线结构 “all choice 3”（见表 3）做了评估，结果如表 4 （第一部分）所示；为进一步提升精度，本文对构造单元和通道做了联合搜索。结果如表 4（第二部分）所示。

![1587352825192](D:\Notes\raw_images\1587352825192.png)

​																			表 4：通道数搜索

#### **对比SOTA方法**

虽然表 4 展示了本文方法的优越性，但是由于搜索空间和训练方法的不同，存在不公平对比的可能性。为直接进行对比，本文采用和 Proxyless NAS [4]、FBNet [26] 相同的搜索空间，并在同一设置下再训练已搜索的模型，具体对比结果如表 5 所示：

![img](https://pic2.zhimg.com/80/v2-6668ffb3fe69b745cdf3244480ff6a89_720w.png)

​														表 5：本文方法与当前 SOTA NAS 方法的对比

#### **混合精度量化搜索**

低功率设备部署模型时需要用到量化技术。但是在通道数与位宽之间找到一个权衡不无挑战。对于这些问题，本文方法可迎刃而解。

这里的搜索空间包含上述的通道搜索空间和混合精度量化搜索空间，后者使用一个全新的选择单元搜索权重和特征图的位宽，如图 5 所示。

在超网络训练中，对于每个选择单元，特征位宽和权重位宽是随机采样的。他们在进化步骤中被确定。具体实验结果如表 6 所示：

![img](https://pic2.zhimg.com/80/v2-814647a5b2f8642284743a2767840dc5_720w.png)

​																	表 6：混合精度量化搜索的结果

#### **搜索成本分析**

搜索成本在 NAS 中是一件要紧的事。本文给出了与先前方法 [4] [26] 的一些对比结果，如表 7 所示：

![img](https://pic4.zhimg.com/80/v2-6f1e368876e9b5fe65dbebfab3c68dcb_720w.png)

​																													表 7：搜索成本

![1587352926569](D:\Notes\raw_images\1587352926569.png)

