## Squeeze-and-Excitation Networks

论文地址：https://arxiv.org/abs/1709.01507

作者：Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

机构：中科院

发表：CVPR 2018 paper, accepted by TPAMI

代码地址：https://github.com/hujie-frank/SENet

文章地址： https://zhuanlan.zhihu.com/p/47494490



SENet的提出动机非常简单，传统的方法是将网络的Feature Map等权重的传到下一层，==SENet的核心思想在于建模通道之间的相互依赖关系，通过网络的全局损失函数自适应的重新矫正通道之间的特征相应强度==。

SENet由一些列SE block组成，一个SE block的过程分为Squeeze（压缩）和Excitation（激发）两个步骤。其中Squeeze通过在Feature Map层上执行Global Average Pooling得到当前Feature Map的全局压缩特征向量，Excitation通过两层全连接得到Feature Map中每个通道的权值，并将加权后的Feature Map作为下一层网络的输入。从上面的分析中我们可以看出SE block只依赖与当前的一组Feature Map，因此可以非常容易的嵌入到几乎现在所有的卷积网络中。论文中给出了在当时state-of-the-art的Inception和残差网络插入SE block后的实验结果，效果提升显著。

SENet虽然引入了更多的操作，但是其带来的性能下降尚在可以接受的范围之内，从GFLOPs，参数数量以及运行时间的实验结果上来看，SENet的损失并不是非常显著。

### 1. SENet详解

#### 1.1. SE Block

一个SE Block的结构如图1所示

![1577089400902](D:\Notes\raw_images\1577089400902.png)

​																		图1：SENet网络结构

网络的左半部分是一个传统的卷积变换，忽略掉这一部分并不会影响我们的SENet的理解。我们直接看一下后半部分，其中 $U$ 是一个 $W\times H \times C$ 的Feature Map， $(W , H)$  是图像的尺寸， $C$ 是图像的通道数。

经过 $F_{sq}(\cdot)$ （Squeeze操作）后，图像变成了一个 $1\times 1 \times C$ 的特征向量，特征向量的值由 $U$ 确定。经过  $F_{ex}(\cdot, W)$ 后，特征向量的维度没有变，但是向量值变成了新的值。这些值会通过和  $U$ 的 $F_{scale}(\cdot,\cdot )$ 得到加权后的 $\tilde{X}$ 。 $\tilde{X}$ 和的 $U$ 维度是相同的。

#### 1.2. Squeeze

Squeeze部分的作用是获得Feature Map $U$ 的每个通道的全局信息嵌入（特征向量）。==在SE block中，这一步通过VGG中引入的Global Average Pooling（GAP）实现的==。也就是通过求每个通道$c, c \in {1, C} $ 的Feature Map的平均值：

![1577089877630](D:\Notes\raw_images\1577089877630.png)

通过GAP得到的特征值是全局的（虽然比较粗糙）。另外, $z_c$ 也可以通过其它方法得到，要求只有一个，得到的特征向量具有全局性。

#### 1.3. Excitation

Excitation部分的作用是通过 ![[公式]](https://www.zhihu.com/equation?tex=z_c) 学习 $C$ 中每个通道的特征权值，要求有三点：

1. ==要足够灵活，这样能保证学习到的权值比较具有价值==；
2. 要足够简单，这样不至于添加SE blocks之后网络的训练速度大幅降低；
3. ==通道之间的关系是non-exclusive的，也就是说学习到的特征能够激励重要的特征，抑制不重要的特征==。

根据上面的要求，SE blocks使用了==两层全连接构成的门机制（gate mechanism）==。门控单元s （即图1中 $1\times 1 \times C$ 的特征向量）的计算法方式表示为：

![1577090452412](D:\Notes\raw_images\1577090452412.png)

==其中 $\delta$ 表示ReLU激活函数， $\sigma$  表示 sigmoid 激活函数==。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_1+%5Cin+%5Cmathbb%7BR%7D%5E%7B%5Cfrac%7BC%7D%7Br%7D%5Ctimes+C%7D) , ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_2+%5Cin+%5Cmathbb%7BR%7D%5E%7BC%5Ctimes%5Cfrac%7BC%7D%7Br%7D%7D) 分别是两个全连接层的权值矩阵。==$r$  则是中间层的隐层节点数，论文中指出这个值是16==。

得到门控单元  $s$  后，==最后的输出  $\tilde{X}$ 表示为  $s$  和 $U$ 的向量积==，即图1中的  $F_{scale}(\cdot,\cdot )$  操作：
$$
\tilde{x_c} = F_{scale}(u_c, s_c) = s_c \cdot u_c
$$
其中 $\tilde{x_c}$ 是 $\tilde{X}$ 的一个特征通道的一个Feature Map，  $s_c$  是门控单元  $s$  （是个向量）中的一个标量值。

以上就是SE blocks算法的全部内容，SE blocks可以从两个角度理解：

1. ==SE blocks学习了每个Feature Map的动态先验==；
2. ==SE blocks可以看做在Feature Map方向的Attention，因为注意力机制的本质也是学习一组权值==。

#### 1.4. SE-Inception 和 SE-ResNet

SE blocks的特性使其能够非常容易的和目前主流的卷及结构结合，例如论文中给出的Inception结构和残差网络结构，如图2。结合方式也非常简单，只需要在Inception blocks或者Residual blocks之后直接接上SE blocks即可。

![1577089809387](D:\Notes\raw_images\1577089809387.png)																			图2：SE-Inception和SE-Resnet

### 2. SENet的复杂性分析

SENet的本质是根据Feature Map的值学习每个Feature Map的权值。 $U$  往往是一个由几万个节点值组成的三维矩阵，但是我们得到的 $s$ 却只有 C 个值，这种 $H \times W$ 程度的压缩是具有非常大的可操作性的。例如将 $U$ 展开成 $(W \times H \times C) \times 1$  的特征向量，然后再通过全连接得到 $s$，这也是目前主流的Feature Map到 FC 的连接方式（`Flatten()`操作）。而且这种方式得到的  $s$ 往往也是效果优于SE blocks的策略的。但是SENet没这么做，==他的原因是SENet是可以添加到网络中的任意一层之后的，而全连接操作往往是整个网络结构的性能瓶颈，尤其是当网络的节点数非常多时==。

论文中主要对比了ResNet-50以及在其中的每一层之后添加了SE blocks之后的在运行性能的各方面的指标：

从计算性能的方向分析：ResNet-50需要约3.86GFLOPS，而SE-ResNet-50仅仅多了0.01个GFLOPS。

从预测速度上来看，运行一个ResNet-50的时间是190ms，SE-ResNet-50的运行时间约209ms，多了10%。

从参数数量上来看，SE-ResNet-50 比 ResNet-50的2500万个参数多了约250万个，约占10%。而且作者发现ResNet-50的最后几层的 SE blocks可以省掉，但是性能影响并不大，这样的网络参数仅多了4%。

### 3. 总结

==SENet的思想非常简单，即通过Feature Map为自身学习一个特征权值，通过单位乘的方式得到一组加权后的新的特征权值。== ==使用的网络结构则是先GAP再接两层全连接的方式得到的权值向量。方法虽然简单，但是非常实用==，并在ImageNet-2017上取得了非常优异的比赛成绩。

第2节对复杂性的分析引发了我们队SE blocks的进一步联想：如何在计算量和性能之间进行权衡？

下面是我的几点思考：

1. 先通过RoI Pooling得到更小的Feature Map（例如 $3 \times 3$ ），在展开作为全连接的输入；
2. 在网络的深度和隐层节点的数目进行权衡，究竟是更深的网络效果更好还是更宽的网络效果更好；
3. 每一层的SE blocks是否要一定相同，比如作者发现浅层更需要SE blocks，那么我们能否给浅层使用一个计算量更大但是性能更好的SE block，而深层的SE blocks更为简单高效，例如单层全连接等。