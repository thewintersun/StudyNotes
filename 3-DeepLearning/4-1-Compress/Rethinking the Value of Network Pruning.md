## Rethinking the Value of Network Pruning

论文地址：https://arxiv.org/abs/1810.05270

发表：ICLR 2019

作者：Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell

项目地址：https://github.com/Eric-mingjie/rethinking-network-pruning



**摘要**：神经网络剪枝是降低深度模型计算成本的常用方式。典型的剪枝算法分为三个步骤：训练（大型模型）、剪枝和微调。在剪枝期间，我们需要根据某个标准修剪掉冗余的权重，并保留重要的权重以保证模型的准确率。在本文中，我们有许多与以往的研究冲突的惊人发现。我们测试了 6 个当前最优剪枝算法，微调剪枝模型的性能只相当于用随机初始化权重训练剪枝模型，有时甚至还不如后者。对于采用预定义目标网络架构的剪枝算法，可以不用典型三步流程，直接从头开始训练目标网络。我们在多个网络架构、数据集和任务上对大量剪枝算法的观察结果是一致的。

结果表明：

1）训练过参数化的大型模型不一定会得到高效的最终模型；

2）学习大型模型的「重要」权重对于较小的剪枝模型未必有用；

3）最终模型的高效率不是由一组继承的「重要」权重决定的，而是剪枝架构本身带来的，这表明一些剪枝算法的作用可以被视为执行网络架构搜索。



### 介绍

刚刚，reddit 上出现了一篇关于论文《Rethinking the Value of Network Pruning》的讨论，该论文的观点似乎与近期神经网络剪枝方面论文的结论相矛盾。这非常令人惊讶，它甚至会改变我们对神经网络的固有观点，即神经网络的过参数化对模型训练与拟合是有益的。

**1 引言**

过参数化是深度神经网络的公认特性（Denton et al., 2014; Ba & Caruana, 2014），该特性导致高计算成本和高内存占用。作为一种补救措施，神经网络剪枝（LeCun et al., 1990; Hassibi & Stork, 1993; Han et al., 2015; Molchanov et al., 2016; Li et al., 2017）可以在计算成本有限的情况下，提高深度网络的效率。网络剪枝的典型过程包括三步：1）训练一个大的过参数化模型，2）根据某个标准修剪训练好的模型，3）微调修剪后的模型，以恢复损失的性能。

![img](https://image.jiqizhixin.com/uploads/editor/05de81bc-4bab-4fbb-b5f0-aae46821269e/1540183001169.png)

​																	*图 1：典型的三步网络剪枝步骤。*

一般来说，剪枝过程背后有两个共识：一，大家都认为首先训练一个大的过参数化模型很重要（Luo et al., 2017），因为这样会提供高性能模型（更强的表征和优化能力），而人们可以安全地删除冗余参数而不会对准确率造成显著损伤。因此，这种观点很普遍，人们认为该方法比从头开始直接训练更小的网络（一种常用的基线方法）更优秀。二，剪枝后的架构和相关的权重被认为是获得最终高效模型的关键。因此大多现有的剪枝技术选择微调剪枝后的模型，而不是从头开始训练它。剪枝后保留下来的权重通常被视为至关重要，因为准确地选择一组重要的权重并不容易。

在本研究中，我们认为上面提到的两种观点都未必正确。针对多个数据集及多个网络架构，我们对当前最优剪枝算法进行了大量实证评估，得出了两个令人惊讶的观察结果。==首先，对于具备预定义目标网络架构的剪枝算法（图 2），从随机初始化直接训练小目标模型能实现与使用经典三步流程相同的性能==。在这种情况下，我们不需要从训练大规模模型开始，而是可以直接从头训练剪枝后模型。==其次，对于没有预定义目标网络的剪枝算法，从头开始训练剪枝后的模型也可以实现与微调相当甚至更好的性能==。

这一观察结果表明，==对于这些剪枝算法而言，重要的是获得的网络架构而不是保留的权重==，尽管这些目标架构还是需要训练大模型才能获得。我们的结果和文献中结果的矛盾之处在于超参数的选择、数据增强策略和评估基线模型的计算力限制。

![img](https://image.jiqizhixin.com/uploads/editor/85038750-5b55-4c93-b805-451f844a2930/1540183001266.png)

*图 2：预定义和未预定义（自动发现）的目标架构之间的区别。稀疏度 x 是用户指定的，a、b、c、d 由剪枝算法决定。*

该论文的结果表明，我们需要重新审视现有的网络剪枝算法。第一阶段的过参数化似乎不像之前想象得那么有效。此外，继承大模型的权重不一定是最优选择，而且可能导致剪枝后的模型陷入局部极小值，即使这些权重按剪枝标准来看是「重要的」。该论文的结果显示自动剪枝算法的价值可能在于识别高效结构、执行隐性架构搜索，而不是选择「重要的」权重。

**4 实验**

在我们的实验中，我们==使用 Scratch-E 来表示用相同的 epoch 数训练小型剪枝模型==，==使用 Scratch-B 来表示用相同的计算预算来训练==。（例如，在 ImageNet 上，如果剪枝模型节省了超过两倍的 FLOPs，我们只需要在训练 Scratch-B 的时候加倍 epoch 数，这相当于比大型模型训练有更少的计算预算）。

**4.1 预定义的目标架构**

![img](https://image.jiqizhixin.com/uploads/editor/b268f8c8-3047-441e-adfd-b2698e9b8cc2/1540183001842.png)

*表 1：基于==通道剪枝（Li et al., 2017）的 L1 范数==（准确率）结果。「Pruned Model」指大模型修剪后的模型。模型配置和「Pruned Model」均来自原论文。*

![img](https://image.jiqizhixin.com/uploads/editor/9bf53111-1a2a-4609-8a6a-6bfb8ba92cc9/1540183001559.png)

*表 2：==ThiNet（Luo et al., 2017）的（准确率）==结果。「VGG-GAP」和「ResNet50-30%」这些模型是按照 Luo 等人 (2017) 论文中定义的配置环境修剪后的模型。为调节我们的实现所使用框架与原论文框架不同所造成的影响，我们对比了未修剪大型模型的准确率下降程度。例如，对于修剪后的模型 VGG-Conv，−1.23 是相对于左侧的 71.03 而言的，71.03 是原论文中未修剪模型 VGG-16 的准确率；−2.75 是相对于左侧的 71.51 而言的，71.51 是我们的实现中 VGG-16 的准确率。*

**4.2 自动发现的目标架构**

![img](https://image.jiqizhixin.com/uploads/editor/13a73cd4-6118-4279-b7a3-eb45d5d7fbb2/1540183003531.png)

*表 4：==Network Slimming (Liu et al., 2017) 的（准确率）==结果。「Prune Ratio」表示整个网络中被剪掉的通道的比例。每个模型的 Prune Ratio 都和原论文一致。*

**4.3 向目标检测任务的迁移学习**

![img](https://image.jiqizhixin.com/uploads/editor/19083844-e0af-4afa-ac74-a273f0208b6e/1540183002288.png)

*表 7：剪枝在==检测任务上的（mAP）==结果。剪枝后的模型来自于 Li et al. (2017)。Prune-C 指在分类预训练权重上剪枝，Prune-D 指在迁移至检测任务之后的权重上进行剪枝。Scratch-E/B 表示从头开始在分类任务上预训练剪枝后的模型，然后迁移至检测任务。*

**5 用网络剪枝进行架构搜索**

![img](https://image.jiqizhixin.com/uploads/editor/da561e78-b66f-4bb5-a930-09cb560edcf3/1540183002686.png)

*图 3：不同方法的剪枝后架构，所有模型都是从头训练的，平均运行 5 次。自动剪枝方法（左：Network Slimming (Liu et al., 2017)，右：==非结构化剪枝 (Han et al., 2015)）获得的架构比在整个网络中均匀修剪通道或稀疏权重的方法具备更高的参数效率==。*

**剪枝后架构的设计原则**

如果自动发现的架构参数效率更高，研究者可能会想：可以从中得出如何设计更好架构的普遍原则吗？为解答该问题，我们分别对 Network Slimming 和非结构化剪枝进行了两次实验，使用的数据集分别是 VGG-19 和 CIFAR-100。

对于 Network Slimming，我们使用剪枝后架构每个层阶段（具备相同特征图大小的层）的平均通道数来构建新的架构集合，我们将该方法称为「Guided Pruning」；对于非结构化剪枝，我们分析了剪枝后架构的稀疏度模式（图 4），并用它们构建新的稀疏模型集合，我们将该方法称为「Guided Sparsifying」，结果见图 5。可以看到对于 Network Slimming（左）和非结构化剪枝（右），指导设计出的架构（绿色）性能与剪枝后架构（蓝色）不相上下。

有趣的是，这些指导设计模式可迁移至不同数据集上的不同架构。我们把在 CIFAR-10 上训练的 VGG-16 剪枝后架构的模式提取出来，用于在 CIFAR-100 上设计高效的 VGG-19。这些架构集合被标注为「Transferred Guided Pruning/Sparsifying」。从下图中我们可以看这些架构（红褐色）的性能比直接在 VGG-19 和 CIFAR-100 上剪枝的架构（蓝色）稍差，比均匀修剪／稀疏化（红色）的架构性能好得多。在这种情况下，===研究者不必在目标数据集上训练大模型来找到高效模型，因为可迁移的设计模式能够帮助我们直接获得高效架构==。

![img](https://image.jiqizhixin.com/uploads/editor/fe6d54f9-2b01-4cf8-a878-fc5a341bfb82/1540183004146.png)

​								*图 5：不同方法的剪枝后架构，所有模型都是从头训练的，平均运行 5 次。*

**6 讨论及结论**

我们建议未来的剪枝方法基于强大的基线模型进行评估，尤其是在目标剪枝架构已经被预定义的情况下。除了高准确率，从头训练预定义目标模型还具备以下优势：

- 由于模型较小，我们可以使用更少的 GPU 内存来训练，可能比训练原始的大型模型更快。
- 剪枝标准与步骤有时需要逐层微调 (Luo et al., 2017)，或针对不同的网络架构进行调整，现在这些都不必要。
- 我们避免了调整剪枝步骤中涉及的额外超参数。

我们的结果支持使用剪枝方法寻找高效架构或稀疏度模式，可以通过自动剪枝方法来进行。此外，在有些案例中，传统的剪枝方法仍然比从头开始训练模型快得多，比如以下两种情况：

- 给定预训练的大模型，且几乎没有训练预算的情况；
- 需要获取不同大小的多个模型，在这种情况下研究者可以训练一个大模型，然后按不同比例进行修剪。

总之，我们的实验展示了从头训练小型剪枝后模型几乎总能得到与按照经典「训练、剪枝、微调」步骤训练出的模型相当或更高的准确率。这改变了我们对过参数化必要性、继承权重有效性的理解。我们进一步展示了自动剪枝算法的价值，它可以用于寻找高效架构和提供架构设计原则。

*参考内容：https://www.reddit.com/r/MachineLearning/comments/9q5t92/r_recent_rethinking_the_value_of_network_pruning/*
