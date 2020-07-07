## AMC: AutoML for Model Compression and Acceleration on Mobile Devices

论文地址：https://arxiv.org/abs/1802.03494

作者：Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han

机构：MIT, CMU, Google

发表：ECCV2018

代码：https://github.com/mit-han-lab/amc

代码2：https://github.com/NervanaSystems/distiller

文章地址：https://zhuanlan.zhihu.com/p/63299393



#### 介绍

**模型压缩**是在计算资源有限、能耗预算紧张的移动设备上有效部署神经网络模型的关键技术。

在许多机器学习应用，例如机器人、自动驾驶和广告排名等，深度神经网络经常受到延迟、电力和模型大小预算的限制。已经有许多研究提出通过压缩模型来提高神经网络的硬件效率。

模型压缩技术的核心是确定每个层的压缩策略，因为它们具有不同的冗余，这通常需要手工试验和领域专业知识来探索模型大小、速度和准确性之间的大设计空间。这个设计空间非常大，**人工探索法通常是次优(sub-optimal)的，而且手动进行模型压缩非常耗时**。

为此，韩松团队提出了 **AutoML 模型压缩**（AutoML for Model Compression，简称 AMC），利用**强化学习**来提供模型压缩策略。

负责这项研究的MIT助理教授韩松博士表示： 

> “**算力换算法**”是当今AutoML系列工作的热点话题，AMC则属于“**算力换算力**”：用training时候的算力换取inference时候的算力。模型在完成一次训练之后，可能要在云上或移动端部署成千上万次，所以inference的速度和功耗至关重要。
>
> 我们用AutoML做一次性投入来优化模型的硬件效率，然后在inference的时候可以得到事半功倍的效果。比如AMC将MobileNet inference时的计算量从569M MACs降低到285M MACs，在Pixel-1手机上的速度由8.1fps提高到14.6fps，仅有0.1%的top-1准确率损失。AMC采用了合适的搜索空间，对压缩策略的搜索仅需要4个GPU hours。 
>

研究人员的目标是自动查找任意网络的压缩策略，以实现比人为设计的基于规则的模型压缩方法更好的性能。这项工作的创新性体现在：

1、AMC提出的learning-based model compression优于传统的rule-based model compression。

2、资源有限的搜索。

3、用于细粒度操作的连续行动空间。

4、使用很少的 GPU 进行快速搜索（ImageNet 上 1 个 GPU，花费 4 小时）。



**目标：**自动化压缩流程，完全无需人工。利用 AI 进行模型压缩，自动化，速度更快，而且性能更高。



AutoML for Model Compression（AMC）利用基于学习的策略来自动执行通道的剪裁，而不是依赖于基于规则的策略和工程师，提高了模型压缩质量。 AMC基于强化学习算法（DDPG， 连续压缩比控制策略），通过  layer-by-layer 的方式处理一个预训练网络模型，对每一层 $L_t$，算法的输入为该层的表示向量$S_t$，该向量表示一个层的有用特征；算法的输出是一个压缩率 $a_t$，当层 $L_t$ 以 $a_t$ 压缩率压缩后，该算法处理下一层 $L_{t+1}$。当所有层经过压缩后，模型不经过微调（fine-tune）直接在验证集上测试，作为验证集的精度。当搜索完全结束后，选择最优的模型（Reward值最高）再经过fine-tune，得到最终的模型效果。



#### **AutoML 模型压缩**

以前的研究提出了许多基于规则的模型压缩启发式方法。例如，在第一层中删除较少的参数，该层提取较低层次的特征，参数的装载量最少; 在FC层删除更多的参数，因为FC层有最多的参数; 在对修剪敏感的层中修剪较少的参数, 等等。但是，由于深层神经网络中的层不是独立的，这些基于规则的剪枝策略并非是最优的，而且不能从一个模型转移到另一个模型。随着神经网络结构的快速发展，我们需要一种自动化的方法来压缩它们，以提高工程师的效率。

**CNN Compression and Acceleration.** Extensive works [20,19,34,12,18,17] have been done on accelerating neural networks by compression.

- **Quantization** [55,10,41] and **special convolution implementations** [36,48,29,3] can also speed up the neural networks. 
- **Tensor factorization** [30,15,27,35] decomposes weights into light-weight pieces, for example [51,11,14] proposed to accelerate the fully connected layers with truncated SVD; Jaderberg et al . [26] proposed to factorize layers into 1x3 and 3x1; and Zhang et al . [53] proposed to factorize layers into 3x3 and 1x1. 
- **Channel pruning** [40,24,1,38] removes the redundant channels from feature maps.  

A common problem of these methods is how to determine the sparsity ratio for each layer.

AutoML for Model Compression（AMC）利用强化学习来自动对设计空间进行采样，提高模型压缩质量。图 1 展示了 AMC 引擎的概览。在压缩网络是，ACM 引擎通过基于学习的策略来自动执行这个过程，而不是依赖于基于规则的策略和工程师。

![1582807325395](D:\Notes\raw_images\1582807325395.png)

图 1：AutoML 模型压缩（AMC）引擎的概览。左边：AMC 取代人工，将模型压缩过程完全自动化，同时比人类表现更好。右边：将 AMC 视为一个强化学习问题。我们通过  layer-by-layer 的方式处理一个预训练网络模型 (e.g., MobileNet) . 

我们观察到==压缩模型的精度对每层的稀疏性非常敏感，需要细粒度的动作空间==。因此，我们不是在一个离散的空间上搜索，而是通过 DDPG agent ==提出连续压缩比控制策略==，通过反复试验来学习：**在精度损失时惩罚，在模型缩小和加速时鼓励。**actor-critic 的结构也有助于减少差异，促进更稳定的训练。

> DDPG: Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D.: Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (2015)

Our reinforcement learning agent (DDPG) receives the embedding $s_t$ from a layer t, and outputs a sparsity ratio $a_t$. After the layer is compressed with $a_t$, it moves to the next layer $L_{t+1}$. ==The accuracy of the pruned model with all layers compressed is evaluated without fine-tuning==,  which is an efficient delegate of the fine-tuned accuracy. 这种简单的近似可以提高搜索时间,不需要重新训练模型,并提供高质量的搜索结果。在进行策略搜索后,优化的模型可以进行微调以实现最佳性能。Finally, as a function of accuracy and FLOP, reward R is returned to the reinforcement learning agent.

针对不同的场景，我们提出了两种压缩策略搜索协议:

- 对于 **latency-critical** 的 AI 应用（例如，手机 APP，自动驾驶汽车和广告排名），我们建议采用资源受限的压缩（resource-constrained compression），在最大硬件资源（例如，FLOP，延迟和模型大小）下实现最佳精度 ）；我们实现资源约束的压缩, 通过约束搜索空间, 在此空间中, 操作空间(修剪比)被约束, 这样压缩的模型总是低于资源预算。
- 对于 **quality-critical** 的 AI 应用（例如 Google Photos），我们提出精度保证的压缩（accuracy-guaranteed compression），在实现最小尺寸模型的同时不损失精度。对于精确保证的压缩, 我们定义了一个奖励,它是关于准确性和硬件资源的函数。有了这个奖励函数,  我们就可以在不损害模型准确性的情况下探索压缩的极限。

![1582878420866](D:\Notes\raw_images\1582878420866.png)

Table 1. Comparisons of reinforcement learning approaches for models searching NAS: Neural Architecture Search [57], NT: Network Transformation [6], N2N: Network to Network [2], and AMC: AutoML for Model Compression. AMC distinguishes from other works by getting reward without fine-tuning, continuous search space control, and can produce both accuracy-guaranteed and hardware resource-constrained models. 

**压缩方法研究**

- Fine-grained Pruning: 用于模型压缩的细粒度剪枝, 能够实现在无精度损失的情况下很高的压缩比，但这类算法会导致不规则稀疏结构，而且需要特定的硬件设备如EIE来实现加速。

- Coarse-grained / structured pruning： 粗粒度 / 通道剪枝，目的是规则地修剪整个权重Tensor（比如按通道、行、列、块等），修剪的权重是规则的,可以用现成的硬件和库加快推理速度。

  本文采用粗粒度剪枝的方法来做卷积和FC的压缩。

**状态空间**

For each layer t, we have 11 features that characterize the state $s_t$:

![1582879803435](D:\Notes\raw_images\1582879803435.png)

where t is the layer index, the dimension of the kernel is n x c x k x k, and the input is c x h x w. FLOPs[t] is the FLOPs of layer Lt. Reduced is the total number of reduced FLOPs in previous layers. Rest is the number of remaining FLOPs in the following layers. Before being passed to the agent, they are scaled within [0,1].

**DDPG Agent**

- DDPG Agent （deep deterministic policy gradient） 用于连续动作空间 (0,1]
- 输入每层的状态Embeding，输出稀疏比 $a_t$, 采用压缩算法（比如channel pruning）进行压缩，在验证集上验证压缩后模型准确率，返回准确率。

For the exploration noise process, we use truncated normal distribution, During exploitation, noise $\sigma$ is initialized as 0.5 and is decayed after each episode exponentially :

![1582880793677](D:\Notes\raw_images\1582880793677.png)

Following Block-QNN [54], each transition in an episode is $(s_t, a_t,R, s_{t+1})$, where R is the reward after the network is compressed. During the update, the baseline reward b is subtracted to reduce the variance of gradient estimation, which is an exponential moving average of the previous rewards[56,6]; The discount factor $\gamma$ is set to 1 to avoid over-prioritizing short-term rewards [4].

![1582880975922](D:\Notes\raw_images\1582880975922.png)

**搜索协议**

​	资源受限压缩，以达到理想的压缩比，同时获得尽可能高的性能。精度保证压缩，在保持最小模型尺寸的同时，完全保持原始精度。

​	为了保证压缩的准确性，我们定义了一个精度和硬件资源的奖励函数。有了这个奖励函数，就能在不损害模型精度的情况下探索压缩的极限。

- 对于资源受限的压缩，只需使用 Rerr = -Error
- 对于精度保证的压缩，要考虑精度和资源（如 FLOPs）：RFLOPs = -Error∙log（FLOPs）

![1582890956996](D:\Notes\raw_images\1582890956996.png)



### **实验和结果**

这种基于学习的压缩策略优于传统的基于规则的压缩策略，具有更高的压缩比，在更好地保持准确性的同时节省了人力。在 4×FLOP 降低的情况下，我们在 ImageNet 上对 VGG-16 模型进行压缩，实现了比手工模型压缩策略**高 2.7％**的精度。

![image-20200407151729780](D:\Notes\raw_images_2\image-20200407151729780.png)



![image-20200407150044817](D:\Notes\raw_images_2\image-20200407150044817.png)

为了证明其广泛性和普遍适用性，我们在多个神经网络上评估 AMC 引擎，包括 VGG，ResNet 和 MobileNet，我们还测试了压缩模型从分类到目标检测的泛化能力。



To tackle the problem, we follow the settings in [16] to conduct 4-iteration pruning & ne-tuning experiments, where the overall density of the full model is set to [50%, 35%, 25% and 20%] in each iteration. For each stage, we run AMC to determine the sparsity ratio of each layer given the overall sparsity. The model is then pruned and ne-tuned for 30 epochs following common protocol. With that framework, we are able to push the expert-tuned compression ratio of ResNet-50 on ImageNet from 3.4x to 5x (see Figure 4) without loss of performance on ImageNet (original ResNet50's [top-1, top-5] accuracy=[76.13%, 92.86%]; AMC pruned model's accuracy=[76.11%, 92.89%]). 

大量实验表明，AMC 提供的性能优于手工调优的启发式策略。对于 ResNet-50，我们将专家调优的压缩比从 3.4 倍提高到 5 倍，而没有降低精度。

The density of each layer during each stage is displayed in Figure 3. The peaks and crests show that the RL agent automatically learns to prune 3x3 convolutional layers with larger sparsity, since they generally have larger redundancy; while prunes more compact 1 x 1 convolutions with lower sparsity.

![1582891799683](D:\Notes\raw_images\1582891799683.png)

​														强化学习 agent 对 ResNet-50 的剪枝策略

![1582891815731](D:\Notes\raw_images\1582891815731.png)

ACM 将模型压缩到更低密度而不损失精度（人类专家：ResNet50 压缩 3.4 倍；AMC：ResNet50 压缩 5 倍）



![1582892099094](D:\Notes\raw_images\1582892099094.png)

​																			AMC 对 MobileNet 的加速

此外，我们将 MobileNet 的 FLOP 降低了 2 倍，达到了 70.2％的 Top-1 最高精度，这比 0.75 MobileNet 的 Pareto 曲线要好，并且在 Titan XP 实现了 1.53 倍的加速，在一部 Android 手机实现 1.95 的加速。

![1582892038073](D:\Notes\raw_images\1582892038073.png)

​										AMC 和人类专家对 MobileNet 进行压缩的精度比较和推理时间比较



### **结论**

传统的模型压缩技术使用手工的特征，需要领域专家来探索一个大的设计空间，并在模型的大小、速度和精度之间进行权衡，但结果通常不是最优的，而且很耗费人力。

本文提出AutoML模型压缩（AMC），利用增强学习自动搜索设计空间，大大提高了模型压缩质量。我们还设计了两种新的奖励方案来执行资源受限压缩和精度保证压缩。

在Cifar和ImageNet上采用AMC方法对MobileNet、MobileNet- v2、ResNet和VGG等模型进行压缩，取得了令人信服的结果。压缩模型可以很好滴从分类任务推广到检测任务。在谷歌Pixel 1手机上，我们将MobileNet的推理速度从8.1 fps提升到16.0 fps。AMC促进了移动设备上的高效深度神经网络设计。
