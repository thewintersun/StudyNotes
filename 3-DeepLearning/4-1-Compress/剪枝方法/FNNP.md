## FNNP: Fast Neural Network Pruning Using Adaptive Batch Normalization

论文地址：https://openreview.net/forum?id=rJeUPlrYvr

作者：Bailin Li, Bowen Wu, Jiang Su, Guangrun Wang

机构： 中山大学，DMAI Inc.  

发表：ICLR2020 盲审中  Under Blind Review 

代码：https://github.com/anonymous47823493/FNNP  PyTorch，基于Intel的代码

 

是目前prune这个领域的SOTA

### 摘要

Finding out the computational redundant part of a trained Deep Neural Network (DNN) is the key question that pruning algorithms target on. Many algorithms try to predict model performance of the pruned sub-nets by introducing various evaluation methods. But they are either inaccurate or very complicated for general application. 

模型压缩的主要问题是找到已经训练好的模型中的赘余连接。同时现在已经有许多评估方法对剪枝后的子网络进行评估，但一般来说他们往往不是很准确或者比较复杂。

In this work, we present a pruning method called Fast Neural Network Pruning (FNNP), in which a simple yet efficient evaluation component called ABN-based evaluation is applied to unveil a strong correlation between different pruned DNN structures and their final settled accuracy. This strong correlation allows us to fast spot the pruned candidates with highest potential accuracy without actually fine tuning them. FNNP does not require any extra regularization or supervision introduced to a common DNN training pipeline but still can achieve better accuracy than many carefully-designed pruning methods. 

本文提出了一种快速的剪枝方法（FNNP），其中使用了一个简单但有效的评估组件，基于ABN的评估，以揭示不同的修剪DNN结构与其最终确定的准确性之间的强相关性。这种强相关性使我们能够以最高的潜在准确度快速找到经过修剪的候选对象，而无需实际对其进行微调。FNNP不需要对常规神经网络的训练进行任何的正则化或者监督，但是该方法与许多经过精心设计的剪枝方法相比，仍然可以实现更高的准确率。

In the experiments of pruning MobileNet V1 and ResNet-50, FNNP outperforms all compared methods by up to 3.8%. Even in the more challenging experiments of pruning the compact model of MobileNet V1, our FNNP achieves the highest accuracy of 70.7% with an overall 50% operations (FLOPs) pruned. All accuracy data are Top-1 ImageNet classification accuracy. Source code and models are accessible to open-source community. 

本论文主要介绍一种快速的模型剪枝方法，并且提出了基于自适应BN的剪枝模型评估方法。

主要解决的是对修剪之后的Subnet,提出了一种更加好的Evaluation方法

- 一般的Prune先剪出一个SubNet,然后选择比较有价值的去finetune得到最后模型
  - 如果直接拿剪掉之后的Subnet的accuracy不合理
  - 如果每个subnet都需要finetune来看效果的话, 太花时间
- 原本的方法是用global的BN stastic来作为subnet的, 但是这样显然不合理
  - 所以这里就改了,用AdaBN之后的结果来衡量Prune
  - 传统的方式叫Vanilla Evaluation,本文和它对比了

### 介绍

神经网络剪枝的目的是在精度允许的情况下，降低模型的复杂度（也就是从原始的网络中去除赘余连接）。剪枝后的模型通常会导致功率或硬件资源预算减少，因此，对于将其部署到低功耗前端系统中尤其有意义。 但是，修剪后模型准确性对最终交付的模型几乎没有任何贡献，这并不是一个小问题。

如果将原始的网络看做一个搜索空间，则在保持网络结构的同时，搜索保留最有价值的子网络（这里我的理解是，在保存子网络模型结构与原始模型相同的同时，去除原始模型不重要的连接，仅仅保存那些对结果最有价值的连接）。目前针对这个问题的模型剪枝的的方法主要有两种：第一个是影响搜索空间，以便更轻松地找到有趣的部分；另外一个则是改变搜索方法，例如通过引入强化学习代理【1】，进化算法【2】，知识蒸馏等。

> 【1】Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. Amc: Automl for model
> compression and acceleration on mobile devices. In Proceedings of the European Conference on
> Computer Vision (ECCV), pp. 784–800, 2018c.
>
> 【2】Zechun Liu, Haoyuan Mu, Xiangyu Zhang, Zichao Guo, Xin Yang, Tim Kwang-Ting Cheng, and Jian Sun. Metapruning: Meta learning for automatic neural network channel pruning. arXiv preprint arXiv:1903.10258, 2019b.

但是，这里需要大量的搜索工作才能获得较好的剪枝模型主要有两个原因：==评估方法的不正确和算法的复杂度高==。

首先我们定义在DNN剪枝任务中的评估问题。为了评估经过修剪的网络是否具有提供较高模型准确性的潜力，通常会有评估过程，这样的过程通过应用评估过程进行快速判断来评估子网的潜力，而不是实际训练子网达到其确定的模型准确性，因为这样做可能非常耗时。 

无论评估过程的设计多么复杂，最终目的都是使评估模块具备以下能力：在推理时间内，它可以快速返回子网的潜在能力，而该能力与其最终确定的精度成正比。然后，此评估程序要么用于为后续的比较/选择模块（He等人，2018c）（Yang等人，2018b）产生高潜力的候选人，要么直接将获胜者子网作为其最终交付对象（Yu等人，2018年）。 （2018）（Yu＆Huang，2019b）

如前所述，==我们发现现有的评估方法要么不准确，要么过于复杂以至于无法普遍应用。== 不正确的评估意味着，由于评估设计不正确，未必一定会将选定的获胜子网微调到高精度。==在这项工作中，我们提供了一个定量分析，以表明不正确的评估方法在子网的评估性能与其最终收敛精度之间的相关性较差==。另一方面，复杂的评估意味着一些精心设计的评估方法，强化学习代理（He等人，2018c），生成对抗性学习（Lin等人，2019），知识提炼等组件引入许多超参数（Luo et al。，2017）等。在不仔细处理评估模块的情况下，==重复准确的实验结果可能是一个挑战==。因此，这些方法被认为具有较高的算法复杂度，从而阻止了它们的普遍应用。

为了解决上述问题，我们在提出的修剪算法中提出了一种快速而准确的评估方法。这项工作中存在的主要新颖性描述如下：

1）提出了一种自动剪枝方法。我们的剪枝方法与典型的神经网络训练兼容，并且不会更改原始参数分布。

2）在我们提出的FNNP框架中，关键组件是基于自适应BN（ABN）的评估模块。 它在修剪候选的准确性与其最终收敛的准确性之间具有强相关性。 这种相关性由我们提出的相关系数来描述，该相关系数用于定量显示为什么现有作品中的某些评估方法不准确。

3）我们提出的FNNP在通过不同级别的修剪对相同基本模型进行修剪的情况下实现了更高的准确性，其性能优于许多精心设计的修剪方法。 在Res-Net 50实验中，我们的FNNP优于所有比较方法1.3％至3.9％。 即使在修剪MobileNet V1紧凑型模型这一艰巨的任务中，我们的FNNP修剪的总体操作量（FLOP）也达到了70.6％的最高精确度。 所有准确度数据均为ImageNet分类准确度最高的。

### 相关工作

如前一节所述，现有的DNN剪枝方法可以从两个角度考虑：剪枝搜索空间的修改和不同搜索方法。

对剪枝搜索空间的修改主要是指对原始模型参数分布进行正则化以进行修剪。例如，引入的正则化包括LASSO（Wen等人，2016）等。我们提出的方法与这类技术是正交的，即==我们提出的快速评估方法可用于此类正则搜索空间==。

同时，提出了不同的搜索方法来发现好的修剪候选者。 修剪主要是在早期由手工启发式处理（Li等，2016）。 因此，经过修剪的候选网络是通过人类专业知识获得的，并通过对其进行训练以达到收敛的准确性进行评估，考虑到大量可能的子网，这可能非常耗时。 然后引入了更多的自动化方法，例如贪婪策略，以节省人工（Yang，2018b）。 最近，提出了多种技术来实现自动化和高效的修剪策略，例如强化学习（He et al。，2018c），知识蒸馏（He et al。，2018c），生成对抗性学习机制（Lin et al。，2019）等。

但是，如第1节中简要讨论的那样，这些工作中仍然存在两个问题，即对子网的评估不正确和复杂的评估降低子网的选择速度。

如第1节所述，==评估不正确主要是由于不正确地将全局BN的统计信息用于子网中==。 例如，在（Li et al。，2016）中，将灵敏度分析应用于DNN，这本质上是关于进行逐层修剪如何影响模型准确性的检查。 但是，为了进行快速评估，将使用全局BN统计信息直接通过验证数据集评估子网。 ==我们的实验表明，这种基于BN的全局模型性能与模型的最终收敛精度之间的关联非常弱（图3），因此绘制的敏感度曲线（Li等，2016）不准确==。 在（He et al。2018c）中可以看到相同的问题。

（Yang et al。，2018b）中提出的短期微调区块在一定程度上固定BN统计数据。但是至少需要进行104次的额外迭代训练每个修剪候选对象。因此，考虑到大量可能的修剪候选者，此方法可能会很耗时。元修剪（Liu等人，2019b）分别训练一个“ PruningNet”以生成修剪候选者的权重，因此可以在推理时间内形成子网并快速进行评估。但是“ PruningNet”由完全连接的层组成，并引入了数百万个额外的可训练权重，从而导致大量训练过程。此外，引入了一些需要熟练进行超参数调整的评估组件，例如强化学习代理（He等，2018c），生成对抗性学习（Lin等，2019），知识提炼（Luo等，2017）等。上。如果不仔细初始化和调整评估模块，可能很难重复确切的实验结果。我们认为上述修剪技术具有缓慢或复杂的评估过程。在这项工作中，==我们旨在使用自适应BN来实现简单，快速而有效的子网评估过程，以提供最先进的修剪性能==。

==通过重新计算剪枝子网中BN统计信息，我们的自适应BN可以快速的将整个网络调整为剪枝后的网络==。因此，可以在瞬间完成对剪枝的网络的评估，从而大大加快了网络修剪的过程。自适应BN也已被文献所关注。在经典识别任务中，重新计算统计数据已用于提高识别准确率（Wang，2018）。（ Li et al。，2018）也使用自适应BN进行跨域识别。在网络架构设计中，后BN用于提高网络容量（Luo等人，2019）。最近的神经体系结构搜索还使用自适应BN来快速评估网络体系结构（Guo，2019; Chu，2019b; a）。即使在网络剪枝中（Yu，2018; Liu2019b）也使用BN来提高子网或替代网络推理的准确性。==但是，以上工作均未分析自适应BN的特性，尤其是为什么自适应BN起作用的原因==。相反，在我们的工作中，我们对重新计算的统计数据所使用的网络与经过训练的收敛网络之间的相关性有深入的了解。

我们的工作还与AutoML有关，尤其是神经体系结构搜索（NAS）。 最近，自动化搜索的兴趣日益浓厚。 值得注意的著作是（Zoph＆Le，2017; Real，2018; Liu等，2019a; Cai等，2019a; b; Guo，2019; Chu，2019b; a）。 但是==所有这些方法都没有与预训练的权重相结合，即它们将网络体系结构和网络参数分开==。 这与我们的先验（某些任务必须在imageNat上进行预训练）相矛盾。

### 方法介绍

![1584429260691](D:\Notes\raw_images\1584429260691.png)

图1中概括并可视化了典型的神经网络训练和剪枝流程，模型训练后，将剪枝应用于完整的模型。 在这项工作中，我们将重点放在结构化滤波器修剪上，因为它不会给幸存的卷积核带来更多的稀疏性。 过滤器修剪任务可以表述为：

![1584429200406](D:\Notes\raw_images\1584429200406.png)

其中 ![[公式]](https://www.zhihu.com/equation?tex=L) 是损失函数， ![[公式]](https://www.zhihu.com/equation?tex=A) 是神经网络模型。 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bl%7D) 是应用于第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的剪枝比例。 在给定约束 ![[公式]](https://www.zhihu.com/equation?tex=C) （目标参数量，操作或执行延迟），可以评估一组剪枝候选对象以找出最佳剪枝比率组合（ ![[公式]](https://www.zhihu.com/equation?tex=r_%7B1%7D%EF%BC%8Cr_%7B2%7D...r_%7Bl%7D) ）也称为剪枝策略。在评估过程中， 这样的剪枝候选者或子网络形成一个搜索空间，评估模块会选择潜力最大的获胜者。 最终，优胜者可以有选择地进行微调，以在从管道末端交付之前达到最终的收敛精度。



如第2节所述，现有的评估方法不准确或复杂。例如（Li et al。2016）和He（2018c），我们认为他们的评估不准确的原因是使用了全局BN统计数据。 为了定量地说明这一想法，我们将BN 的符号表示如下： 

![1584429328163](D:\Notes\raw_images\1584429328163.png)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%EF%BC%8C%5Cbeta) 是可训练参数， ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 是一个很小的参数防止分母为0，对于一个batch为N的样本，均值和方差则通过下面两个公式计算

![1584429376789](D:\Notes\raw_images\1584429376789.png)

同时在训练的过程中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%EF%BC%8C%5Csigma) 通过对均值和方差计算滑动平均得到。

![1584429404610](D:\Notes\raw_images\1584429404610.png)

其中 ![[公式]](https://www.zhihu.com/equation?tex=m) 表示滑动平均系数， ![[公式]](https://www.zhihu.com/equation?tex=t) 表示迭代次数，在一般的训练过程中，如果训练的迭代次数是 ![[公式]](https://www.zhihu.com/equation?tex=T) ，则在测试的过程中使用 ![[公式]](https://www.zhihu.com/equation?tex=u_%7BT%7D%EF%BC%8C%5Csigma_%7BT%7D%5E%7B2%7D) 。

==作者认为，快速的分析剪枝模型， 就是需要获得每一个子网络的自适应BN系数==，自适应BN（Adaptive Bach Normalization， ABN)的两个系数则是通过对训练集中采样一小部分数据，然后进行多次迭代得到的。（也就是在一个较小的数据集上进行再训练，而不需要全部数据集）

然而，在（L，2016）和（He，2018c）中， ![[公式]](https://www.zhihu.com/equation?tex=u_%7BT%7D%EF%BC%8C%5Csigma_%7BT%7D%5E%7B2%7D) 在这项工作中称为训练后BN（PBN）直接用于评估不同的修剪候选对象。我们在测试准确性方面比较了评估模型和微调模型之间的相关性。结果分别示于图3。从观察中可以看出，PBN统计数据与ABN揭示的强相关性相比，相关性较差。这解释了为什么在许多现有方法中使用基于PBN的评估是不准确的，（Liu et al。，2019b）中还提出了一个“短期微调块”来解决BN统计数据不准确的问题。 ==一个时期的短期微调明显有助于在评估过程中提供强大的相关性==，如图3所示。训练一个时期需要相对较长的时间。 为了定量描述这种观察，我们在第4节及其相关讨论中引入了相关指数。

我们在图2中展示了（FNNP）框架的总体工作流程。我们的修剪流程包括三个主要部分，修剪策略生成，滤波器（权重）修剪和基于ABN的候选者评估。![1584430297637](D:\Notes\raw_images\1584430297637.png)

**剪枝策略生成：**策略生成以 ![[公式]](https://www.zhihu.com/equation?tex=L) 层模型的分层修剪速率向量 ![[公式]](https://www.zhihu.com/equation?tex=%28r_%7B1%7D%EF%BC%8Cr_%7B2%7D...r_%7Bl%7D%29)的形式输出修剪策略。==生成过程遵循预定义的约束，例如推理延迟，操作（FLOP）或参数方面的减少等==。具体而言，它从给定范围 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2CR%5D) 中随机采样 ![[公式]](https://www.zhihu.com/equation?tex=L) 个实数以形成修剪策略，其中 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bl%7D) 表示第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的修剪率。 ![[公式]](https://www.zhihu.com/equation?tex=R) 是应用于层的最大修剪率，它形成子网搜索空间的边界。在后面的部分中，我们将介绍相关性受R影响的ABN。此外，其他生成方法也可以用于此模块，例如进化算法。在这项工作中，==我们发现简单的随机采样足以使整个管道以最新的精度快速生成子网==。

**滤波器剪枝：**滤波器剪枝块会根据策略生成模块生成的剪枝向量对完整尺寸的训练模型进行修剪。基线模块中的过滤器**根据其** ![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D) **范数进行排序**，较小的 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bl%7D) 滤波器将会被永远剪枝。在此过程之后，可以将来自搜索空间的经过修剪的候选修剪对象传递到下一个评估阶段。（这里的意思应该是对权重按照某个标准进行排序，然后较小的一部分被修剪掉）

**基于ABN的评估模块：**基于ABN的评估模块为从经过修剪的候选模型提供自适应的BN统计信息和快速评估。 ==给定修剪后的网络，它将冻结所有可学习的参数并遍历训练集中的少量数据以计算ABN统计信息（两个参数）。== 实际上，我们在ImageNet实验中对总训练集的1/55进行了30次迭代采样，而在单个Nvidia 1080 Ti GPU中仅花费了10秒。 接下来，该模块在验证集上评估候选网络的性能，并在准确性排名中选择最优先的网络作为获胜者候选。 第4节中提供的相关分析保证了此过程的有效性。 ==经过微调后，经过修剪的优胜者候选人最终将作为输出交付==。（这里也就是通过一部分数据集对子网络进行训练，仅仅训练BN中的两个可学习系数，然后再val数据集上挑选出最优的，并对模型进行微调）。

### 实验分析

**关联性分析：**MobileNetV1在ImageNet上的实验。使用AMC生成的剪枝策略中选择40个。

![1584432153629](D:\Notes\raw_images\1584432153629.png)

Figure 3: Correlation between fine-tuning accuracy and post-evaluation accuracy with three different evalution methods. From left to right: PBN-based, ABN-based and 1 epoch fine-tuning ( MobileNet
V1 on ImageNet classification Top-1 results)

结果表明,它们在PBN与收敛accuarcy之间几乎没有相关性。在我们的ABN 和收敛的accuarcy之间的相关性要强大得多。

![1584432830269](D:\Notes\raw_images\1584432830269.png)

Figure 4: PBN vs. ABN: Correlation between post-evaluation accuracy and fine-tuning accuracy
with different pruning ratios to operations ( MobileNet V1 on ImageNet classification Top-1 results)

基于ABN的评估方法，收敛很快，只需要50个iteration。

![1584433060378](D:\Notes\raw_images\1584433060378.png)

Figure 5. Relation between number of sampling iterations and ABN-based evaluation results.

与其他方法的对比：

![1584433129633](D:\Notes\raw_images\1584433129633.png)



关于代码实现的疑问：这里对权重剪枝这一块有点疑问，开源的源码中也没这一块的内容，首先在某一层的权重的尺度为 ![[公式]](https://www.zhihu.com/equation?tex=%28c_%7Bout%7D%2Cc_%7Bin%7D%2Cw%2Cw+%29) ，则 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bin%7D) 与输入特征图的通道数有关， ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bout%7D) 与输出特征图的通道数有关，也就是与下一层卷积核的 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bin%7D) 有关，也就说如果当前层权重的 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bout%7D) 参数将会影响下一次权重的 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bin%7D) 参数，这里进行删减应该首先计算权重的L1范式，然后根据该参数对权重进行排序，将最后的一部分设置为0（直接删除应该也可以吧，不过这样需要先处理下一层的权重）

#### SOTA对比

**resnet-50 +  ImageNet**

![1584433289570](D:\Notes\raw_images\1584433289570.png)

**MobileNetV1 + ImageNet**

![1584433404378](D:\Notes\raw_images\1584433404378.png)

