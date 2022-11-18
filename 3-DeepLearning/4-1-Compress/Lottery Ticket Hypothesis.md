## The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

论文地址：https://arxiv.org/abs/1803.03635v5

相关论文: https://arxiv.org/abs/2002.00585  

Proving the Lottery Ticket Hypothesis: Pruning is All You Need

作者: Jonathan Frankle, Michael Carbin

机构：MIT 

发表：ICLR 2019 Best Paper

代码地址： https://github.com/google-research/lottery-ticket-hypothesis



### 摘要

Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.
We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective.
We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy.

神经网络剪枝技术可以将受过训练的网络的参数计数减少90％以上，在不影响准确性的情况下，降低存储要求并提高计算性能。

然而，目前的经验是通过剪枝产生的稀疏架构很难从头训练，也很难提高训练性能。作者发现标准的剪枝技术自然地揭示了子网络，其初始化使它们能够有效地进行训练。

基于这些结果，作者提出了==彩票假设：任何密集、随机初始化的包含子网络（中奖彩票）的前馈网络 ，当隔离训练时，可以在相似的迭代次数内达到与原始网络相当的测试精度==。

并提出了一种算法来==识别中奖彩票和一系列支持彩票假设的实验==。结果中奖彩票的size仅为MNIST和CIFAR10几种全连接和卷积式前馈架构的10-20％，同时比原始网络学得更快，并达到更高的测试精度。

### 介绍

神经网络的参数很多，但其中有些参数对最终的输出结果贡献不大而显得冗余，将这些冗余的参数剪掉的技术称为剪枝。剪枝可以减小模型大小、提升运行速度，同时还可以防止过拟合。

剪枝分为one-shot和iteration剪枝：

1. one-shot剪枝过程：训练模型--> 评估神经元（或者kernel、layer）的重要性-->去掉最不重要的神经元--> fine-tuning-->停止剪枝。
2. iteration剪枝过程：训练模型--> 评估神经元（或者kernel、layer）的重要性-->去掉最不重要的神经元--> fine-tuning-->判断是不是要继续剪枝，如果是回到第二步（评估神经元的重要性），否则停止剪枝。

剪枝还分为结构化剪枝和非结构化剪枝：

1. 结构化剪枝：直接去掉整个kernel的结构化信息；
2. 非结构化剪枝：考虑每个kernel的每个元素，删除kernel中不重要的参数；也称为稀疏剪枝。

考虑几个MNIST的全连接网络和CIFAR10的卷积网络，如下~

![1584361081571](D:\Notes\raw_images\1584361081571.png)

在图1中，从上述网络中随机抽样（非结构化修剪）和训练子网。实线表示作者找到的中奖彩票，虚线表示剪枝后的网络。

![1584361772208](D:\Notes\raw_images\1584361772208.png)

![1584361788340](D:\Notes\raw_images\1584361788340.png)

由图1可知，==中奖彩票能较快的训练，并达到跟原网络相似的精度。同时表明网络越稀疏学习越慢，导致最终的测试精度越低==。

基于此，作者提出了**彩票假设**：**随机初始化的密集神经网络包含一个初始化的子网，当经过隔离训练时，它可以匹配训练后最多相同迭代次数的原始网络的测试精度。** 从公式上来说，彩票假设预测其中j'≤j（相应的训练时间），a≥a（相称的准确度）和|m|«|θ| （参数较少）。

==作者发现标准的剪枝技术会自动从全连接和卷积的前馈网络中发现这种可训练的子网。但当子网的权重被初始化时，他们并不能被有效的训练==。

**识别中奖彩票。**作者通过训练网络并修剪其最小等级的权重来识别获胜的门票。剩余的连接构成了中奖票的架构。作者用iteration剪枝进行，主要实验步骤如下：

1. 随机初始化神经网络 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%3B%5Ctheta_%7B0%7D%29) ；
2. 训练网络j次迭代后，得到参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bj%7D) ;
3. 在参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bj%7D) 中剪掉 $p\%$ 的参数，生成一个mask ![[公式]](https://www.zhihu.com/equation?tex=m) ；
4. 在剩余的结构中用原始初始化参数$\theta_0$进行训练，以产生中奖彩票。

作者方法的独到之处在于每次将为剪掉的连接值初始化为原始网络在训练之前最开始的那个值。

上述步骤中的剪枝是**one-shot**的，也就是网络只被训练一次，裁剪P%的权重，一步到位。但是作者使用的是**iterative pruning，**即重复训练、剪枝、重置网络n次，每次剪掉 ![[公式]](https://www.zhihu.com/equation?tex=P%5E%7B%5Cfrac%7B1%7D%7Bn%7D%7D%5C%25)的参数量。作者发现iterative pruning的效果要好于ont-shot。

![1584410502296](D:\Notes\raw_images\1584410502296.png)



**结果: ** 作者能在在MNIST的全连接架构和CIFAR10的卷积架构中识别中奖彩票。 

- 基于剪枝的寻找中奖彩票的策略对学习率很敏感：需要warmup才能以更高的学习率找到中奖门票。
- 找到的中奖彩票是原始网络size的10-20％（或更小）。
-  在这个size下，它们在最多相同的迭代次数（相应的训练时间）内达到或超过原始网络的测试精度（相称的准确度）。 
- 当随机重新初始化时，中奖彩票表现得更糟，这意味着单独的结构无法解释中奖彩票的成功。

**彩票猜想**。 即SGD寻找并训练一组良好初始化的权重。 密集，随机初始化的网络比修剪产生的稀疏网络更容易训练，因为有更多可能的子网络，训练可以从中恢复中奖彩票。

**作者的贡献。**

1. 证明了剪枝得到的可训练子网络可以达到与原始网络一致的，这些网络是在相同数量的迭代中得出的。
2. 表明了修剪发现中奖彩票比原始网络学得更快，同时达到更高的测试准确度并更好地泛化。
3. 建议将彩票假设作为神经网络组成的新视角来解释这些发现。

**给我们的启示。**在本文中，作者实证研究了彩票假设。可以应用这个假设来：

1. **提高训练性能**。由于中奖彩票可以从一开始就被隔离训练，可以设计出尽可能早地搜索中奖彩票和剪枝的训练方案。

2. **设计更好的网络**。中奖彩票揭示了稀疏架构和初始化的组合，这些组合特别擅长学习。我们可以从获奖门票中获取灵感，设计具有有助于学习的相同属性的新架构和初始化方案。我们甚至可以将为一项任务发现的中奖彩票转移给许多其他任务。

3. **提高我们对神经网络的理论认识**。可以研究为什么随机初始化的前馈网络似乎包含中奖彩票和对优化理论研究的潜在影响和泛化。

   

### WINNING TICKETS IN FULLY-CONNECTED NETWORKS 

在本节中评估应用于在MNIST上训练的全连接网络的彩票假设。 作者使用简单的逐层修剪启发式：删除每层中具有最低重要性的权重的百分比）。 

注意：图中， ![[公式]](https://www.zhihu.com/equation?tex=P_m%3D%5Cfrac%7B%7C%7Cm%7C%7C_0%7D%7B%5Ctheta%7D) 指的是mask m的稀疏度。 ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bm%7D) = 25%意味着有75%的参数被剪掉。

![1584408803058](D:\Notes\raw_images\1584408803058.png)

Figure 3: Test accuracy on ==Lenet== (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are Pm—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.

从图3可以看出，不同剪枝率的子网络性能不一，其中最好的为21.1%，而且重新初始化的结果要原低于原始网络的。从图中可以看出，作者找到的winning tickets比原来的网络要学得更快。为了衡量winning tickets初始化的重要性，作者继续做了对比实验，结果如下：

![1584409398771](D:\Notes\raw_images\1584409398771.png)

Figure 4: Early-stopping iteration and accuracy of Lenet under one-shot and iterative pruning.
Average of five trials; error bars for the minimum and maximum values. At iteration 50,000, training
accuracy $\approx$ 100% for Pm >= 2% for iterative winning tickets (see Appendix D, Figure 12).

由图4可知，==one-shot剪枝也可以在没有重复训练的情况下识别中奖彩票，但iteration剪枝找到的中奖彩票学习能力比one-shot快，并在较小的网络规模下达到更高的测试精度==。4（c）表示==中奖彩票的学习速度和精度都比随机初始化要好==。

### WINNING TICKETS IN CONVOLUTIONAL NETWORKS 

在这里将彩票假设应用于CIFAR10上的卷积网络，增加了学习问题的复杂性和网络的规模。


![1584409370473](D:\Notes\raw_images\1584409370473.png)

Figure 5: Early-stopping iteration and test and training accuracy of the ==Conv-2/4/6== architectures when
iteratively pruned and when randomly reinitialized. ==Each solid line is the average of five trials==; each
==dashed line is the average of fifteen reinitializations== (three per trial). The bottom right graph plots test
accuracy of winning tickets at iterations corresponding to the last iteration of training for the original
network (20,000 for Conv-2, 25,000 for Conv-4, and 30,000 for Conv-6);  at this iteration, training
accuracy $\approx$100% for Pm >= 2% for winning tickets (see Appendix D).

从图5体现了初始化的重要性，==重新初始化的效果会差很多很多，而且训练时间也相对长很多==。


![1584410251583](D:\Notes\raw_images\1584410251583.png)

Figure 6: Early-stopping iteration and test accuracy at early-stopping of Conv-2/4/6 when iteratively pruned and trained with dropout. The dashed lines are the same networks trained without dropout (the solid lines in Figure 5). Learning rates are 0.0003 for Conv-2 and 0.0002 for Conv-4 and Conv-6.

==结论：dropout 比无 dropout的学习慢，但是效果更好==。

### VGG AND RESNET FOR CIFAR10 

![1584411619244](D:\Notes\raw_images\1584411619244.png)

Figure 8: Test accuracy (at 10K, 20K, and 30K iterations) of Resnet-18 when iteratively pruned.

图8. ==体现了学习率的影响，小的学习率达到的精度有限，但大的学习率找不到中奖彩票==。可以看到在稍微复杂点的网络中，需要用warmup来寻找中奖网络。



### DISCUSSION 

1. 中奖彩票初始化的重要性。
2. 中奖彩票结构的重要性，产生中奖票的初始化安排在特定的稀疏架构中。
3. 中奖彩票提高了泛化性，Occam's Hill简单有效原理，但过于简单效果会更差。中奖彩票表明较大的网络可能明确地包含更简单的表示。
4. 神经网络优化的启示，SGD能获得一个比较好的结果是基于一个过参数化的网络，这是因为它们具有更多潜在中奖门票的子网络组合。那SGD是否有必要或足以让神经网络优化到特定的测试精度。



---

https://www.zhihu.com/question/323214798/answer/678706173

作者：刘壮

谢谢之前有回答提到我们的paper。在这里就我们的paper, Rethinking the Value of Network Pruning ([https://arxiv.org/abs/1810.05270](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.05270)) ，中Section 6对Lottery Ticket Hypothesis (以下简称LTH) 的实验展开一些讨论。我们实验观察到的基本现象是：

**在unstructured pruning上，当使用小的learning rate时，相比random initialization，winning ticket有帮助；当使用（标准的且accuracy更高的）大learning rate时，winning ticket没有帮助；在L1-norm structured pruning上，不管大小learning rate, winning ticket都没有帮助。**下面详细介绍。

（背景补充：unstructured pruning是指对于individual weights进行prune；structured pruning是指对于filter/channel/layer的prune）

——————两篇paper的介绍——————

这里先简单总结一下我们的paper的主要结论，方便后面讨论和LTH的异同：对于structured pruning来说，pruned完之后的model是可以train from scratch (from random initialization) 而达到和一般fine-tuning相比同样（甚至有些情况更好）的精度的，所以对于structured pruning来说更重要的是prune完得到的architecture而非weights。我们在6种structured pruning方法，不同的网络architecture，CIFAR/ImageNet数据集上实验验证了这个现象均成立。因此，对于有些structured pruning方法，我们并不需要先训大模型再prune，直接重头训练小模型就可以；另一些structured pruning的方法其实可以看做是architecture search的方法。

我们的paper的主要结论是对structured pruning，然而我们也做了unstructured pruning (magnitude-based [1]) 的实验，发现在CIFAR数据集上，prune完的model还是基本可以from scratch train到和fine-tuning一样好；但在ImageNet数据集上，当prune ratio大到一定程度，from scratch是达不到fine-tuning的performance的。因此，我们paper的主要结论只针对structured pruning。

LTH的pruning只evaluate了unstructured pruning (magnitude-based [1]) 的方法，主要结论是prune完的模型是可以重头训达到competitive的performance，但是前提是必须使用original initialization，也就是“winning ticket”。paper实验说明了使用winning ticket比random initialization要好。主要实验是在一些相对小的model和CIFAR/MNIST上进行的。

好了，两篇paper矛盾的地方来了，那么在unstructured pruning (on CIFAR)重头train小模型时到底需不需要使用winning ticket？使用winning ticket到底相比random initialization有没有提升? 

——————learning rate可能是问题的关键——————

在我们后续的unstructured pruning的实验中发现，learning rate是一个很重要的因素。这里的learning rate既指train大model的也指train小model的。我们在CIFAR-10上实验发现，当使用0.01作为starting learning rate时，winning ticket是有帮助的；当使用0.1作为starting learning rate时，是大概没有的（见下图）。我们的paper中其他实验都使用的是0.1作为starting learning rate，这个learning rate自从ResNet出来后就一直和SGD with momentum一起在classification problem中广为使用，实际上已经是在CIFAR/ImageNet上的默认设置（任何带BN的网络结构），train过这两个数据集的人应该都懂。我们evaluate的6种之前的pruning的方法也都是使用这个learning rate。使用0.1也确实比0.01的accuracy更高，毕竟是前人(可能是ResNet?)调出来的。

![img](https://pic3.zhimg.com/50/v2-b46378e6c6a65761e16232bd4d678783_hd.jpg)![img](https://pic3.zhimg.com/80/v2-b46378e6c6a65761e16232bd4d678783_720w.jpg)

对于structured pruning，我们也在L1-norm filter pruning [2]上做了实验。无论大小learning rate, winning ticket均没有帮助:

![img](https://pic4.zhimg.com/50/v2-c9c1a8b1eb8075f68438b12fc4be0786_hd.jpg)![img](https://pic4.zhimg.com/80/v2-c9c1a8b1eb8075f68438b12fc4be0786_720w.jpg)

可能有人要问了，为什么要相信我们的发现？我宁愿相信best paper! 哈哈，开个玩笑。很多人没注意到的是，人家LTH paper里虽然主实验用的是小learning rate, 但他们也跑了标准大learning rate的实验，然后发现winning ticket并不比random initialization好 (见下两图，分别是VGG和ResNet-18)！(这里LTH还引用了我们paper) 但是人家把这结果放在最后一个实验section里，不是在他们用来得出结论的前两个section....  也就是说，作者是知道大learning rate是不work的，只是没有用这个实验来突出强调LTH的局限性。

![img](https://pic3.zhimg.com/50/v2-2418094884bc9b0f85c8811ec7204146_hd.jpg)![img](https://pic3.zhimg.com/80/v2-2418094884bc9b0f85c8811ec7204146_720w.jpg)

![img](https://pic4.zhimg.com/50/v2-4658a26ed8ed39f3a36e4b58829f8e60_hd.jpg)![img](https://pic4.zhimg.com/80/v2-4658a26ed8ed39f3a36e4b58829f8e60_720w.jpg)

顺便说一句，和LTH进行比较并不是我们paper的原来主旨，我们的原来主旨是evaluate之前的pruning方法；是有个reviewer问起来我们paper和LTH有点矛盾之后我们才进行比较的。不过现在LTH得了best paper, 倒是很多人对和LTH的比较更感兴趣。 我们LTH部分的实验也提供了开源代码，欢迎尝试：[https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/lottery-ticket](https://link.zhihu.com/?target=https%3A//github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/lottery-ticket) 

——————总结——————

这里再总结一下paper关于LTH的实验（也包括LTH原paper的实验），就是：在unstructured pruning上，当使用小的learning rate时，winning ticket有帮助；当使用（标准的且accuracy更高的）大learning rate时，winning ticket没有帮助；在L1-norm structured pruning上，不管大小learning rate, winning ticket都没有帮助。更多讨论见我们paper的section 6。这个回答并没有想说LTH的价值不大，只是想指出可能有些其他情况它并不成立。其他paper中，winning ticket不一定有帮助的实验请参考[3], train from scratch不需要使用winning tickets的实验可以参考[4]。

至于为什么即使对unstructured pruning, 也只有当小learning rate的时候LTH才成立，我的一个naive的猜想是（并没有经过实验验证，轻喷），当learning rate较小时，最终训完时候的weights和original initialization时候的weights距离较小（不一定是L2 distance,可能是更抽象的），所以如果使用original initialization来对小model进行初始化，相当于leak了一些training完后的大model的信息。极端一点的话，甚至可以说，使用了winning ticket的这个小model并不是从scratch训的，而是已经某种程度上based on这个已经train了很久的大model了，所以它能train的相对好。当使用大learning rate时，训完的weights和init的相差较远，就不存在这个原因了。

[1] Learning both Weights and Connections for Efficient Neural Networks. Han et al. 2015.

[2] Pruning filters for Efficient ConvNets. Li et al. 2017. 

[3] The State of Sparsity in Deep Neural Networks. Gale et al. 2019.

[4] Pruning neural networks: is it time to nip it in the bud? Crowley et al. 2019.