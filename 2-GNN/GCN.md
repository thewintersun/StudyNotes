# 图卷积GCN

文章来源：https://zhuanlan.zhihu.com/p/89503068

本文的内容包括图卷积的基础知识以及相关辅助理解的知识点，希望有一定深度学习基础的同学能够平滑上手理解GCN。

具体来说，本文包括什么：

- 图网络的有哪些种类，它们之间的区别和联系是什么？
- 图卷积的地位，图卷积怎么理解？
- 有没有一个图卷积的通式，各种不同的公式怎么统一？
- 有没有一个最简单的例子说明网络的训练过程？
- 想快速上手有哪些代码实现和库？

![1577159614555](D:\Notes\raw_images\1577159614555.png)

## 图网络的分类

在最开始，我们先梳理一下经常被提到的几个术语的区别和联系，也就是 ==Graph Embedding，Graph Neural Network 和 Graph Convolutional Network的区别和联系是什么==。

### Graph Embedding

图嵌入（Graph Embedding/Network Embedding，GE），属于表示学习的范畴，也可以叫做网络嵌入，图表示学习，网络表示学习等等。通常有两个层次的含义：

- **将图中的节点表示成低维、实值、稠密的向量形式**，使得得到的向量形式可以在向量空间中具有表示以及推理的能力，这样的向量可以用于下游的具体任务中。例如用户社交网络得到节点表示就是每个用户的表示向量，再用于节点分类等；
- **将整个图表示成低维、实值、稠密的向量形式**，用来对整个图结构进行分类；

图嵌入的方式主要有三种：

- **矩阵分解：**基于矩阵分解的方法是将节点间的关系用矩阵的形式加以表达，然后分解该矩阵以得到嵌入向量。通常用于表示节点关系的矩阵包括邻接矩阵，拉普拉斯矩阵，节点转移概率矩阵，节点属性矩阵等。根据矩阵性质的不同适用于不同的分解策略。
- **DeepWalk：**DeepWalk 是基于 word2vec 词向量提出来的。word2vec 在训练词向量时，将语料作为输入数据，而图嵌入输入的是整张图，两者看似没有任何关联。但是 DeepWalk 的作者发现，预料中词语出现的次数与在图上随机游走节点被访问到底的次数都服从幂律分布。因此 DeepWalk 把节点当做单词，把随机游走得到的节点序列当做句子，然后将其直接作为 word2vec 的输入可以节点的嵌入表示，同时利用节点的嵌入表示作为下游任务的初始化参数可以很好的优化下游任务的效果，也催生了很多相关的工作；
- **Graph Neural Network：**图结合deep learning方法搭建的网络统称为图神经网络GNN，也就是下一小节的主要内容，因此图神经网络GNN可以应用于图嵌入来得到图或图节点的向量表示；

### Graph Neural Network

图神经网络(Graph Neural Network, GNN)是指神经网络在图上应用的模型的统称，根据采用的技术不同和分类方法的不同，又可以分为下图中的不同种类，例如==从传播的方式来看，图神经网络可以分为图卷积神经网络（GCN），图注意力网络（GAT，缩写为了跟GAN区分），Graph LSTM等等==，本质上还是把文本图像的那一套网络结构技巧借鉴过来做了新的尝试。但在这篇文章中并不会细细介绍下面的每一种，作为入门篇，我们着重理解最经典和最有意义的基础模型GCN，这也是理解其他模型的基础。

![1578903237016](D:\Notes\raw_images\1578903237016.png)

![1578903260458](C:\Users\j00496872\AppData\Roaming\Typora\typora-user-images\1578903260458.png)

![1578903300040](D:\Notes\raw_images\1578903300040.png)

图1 图神经网络GNN的分类：分别从图的类型，训练的方式，传播的方式三个方面来对现有的图模型工作进行划分 图片来源：https://arxiv.org/pdf/1812.08434.pdf

### Graph Convolutional Network

图卷积神经网络(Graph Convolutional Network, GCN)正如上面被分类的一样，是一类采用图卷积的神经网络，发展到现在已经有基于最简单的图卷积改进的无数版本，在图网络领域的地位正如同卷积操作在图像处理里的地位。

![img](https://pic1.zhimg.com/80/v2-1767c06f11e8afc95e13d99f36f7ea88_hd.jpg)

​																	图2 GE,GNN,GCN的区别和联系

如图2所示，这三个比较绕的概念可以用一句话来概括：**图卷积神经网络GCN属于图神经网络GNN的一类，是采用卷积操作的图神经网络，可以应用于图嵌入GE。**

## 卷积VS图卷积

要理解图卷积网络的核心操作图卷积，可以类比卷积在CNN的地位。

如下图所示，数字图像是一个二维的离散信号，对数字图像做卷积操作其实就是利用卷积核（卷积模板）在图像上滑动，将图像点上的像素灰度值与对应的卷积核上的数值相乘，然后将所有相乘后的值相加作为卷积核中间像素对应的图像上像素的灰度值，并最终滑动完所有图像的过程。

**用随机的共享的卷积核得到像素点的加权和从而提取到某种特定的特征，然后用反向传播来优化卷积核参数就可以自动的提取特征，是CNN特征提取的基石**。

![1578905487280](D:\Notes\raw_images\1578905487280.png)

然而，现实中**更多重要的数据集都是用图的形式存储的**，例如社交网络信息，知识图谱，蛋白质网络，万维网等等。这些图网络的形式并不像图像，是排列整齐的矩阵形式，而是非结构化的信息，**那有没有类似图像领域的卷积一样，有一个通用的范式来进行图特征的抽取呢**？这就是图卷积在图卷积网络中的意义。

对于大多数图模型，有一种类似通式的存在，这些模型统称GCNs。因此可以说，图卷积是处理非结构化数据的大利器，随着这方面研究的逐步深入，人类对知识领域的处理必将不再局限于结构化数据（ CV，NLP），会有更多的目光转向这一存在范围更加广泛，涵盖意义更为丰富的知识领域。

### 图卷积

![img](https://pic4.zhimg.com/80/v2-4013be2de0ecd8695e30b02b83a5bda3_hd.jpg) 

图4 图结构实例

### 图的定义

对于图，我们有以下特征定义：

对于图 ![[公式]](https://www.zhihu.com/equation?tex=G%3D%28V%2CE%29) ， ![[公式]](https://www.zhihu.com/equation?tex=V) 为节点的集合， ![[公式]](https://www.zhihu.com/equation?tex=E) 为边的集合，对于每个节点 ![[公式]](https://www.zhihu.com/equation?tex=i) ， 均有其特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) ，可以用矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X_%7BN%2AD%7D) 表示。其中 ![[公式]](https://www.zhihu.com/equation?tex=N) 表示节点数， ![[公式]](https://www.zhihu.com/equation?tex=D) 表示每个节点的特征数，也可以说是特征向量的维度。

### 图卷积的形象化理解

在一头扎进图卷积公式之前，我们先从其他的角度理解一下这个操作的物理含义，有一个形象化的理解，我们在试图得到节点表示的时候，容易想到的最方便有效的手段就是利用它周围的节点，也就是它的邻居节点或者邻居的邻居等等，这种思想可以归结为一句话：

==图中的每个结点无时无刻不因为邻居和更远的点的影响而在改变着自己的状态直到最终的平衡，关系越亲近的邻居影响越大。==

实际上从邻居节点获取信息的思想在很多领域都有应用，例如word2vec，例如pagerank。关于这个点展开的内容文章[2]有非常详细的解释。

更加细节的如何从傅立叶变换到拉普拉斯算子到拉普拉斯矩阵的数学推倒可以转向博客[7]，为了避免数学功底没有那么强的初学者（比如我）被绕晕，我们先建立大纲，不要太发散。

### 图相关矩阵的定义

那么有什么东西来度量节点的邻居节点这个关系呢，学过图论的就会自然而然的想到邻接矩阵和拉普拉斯矩阵。举个简单的例子，对于下图中的左图（为了简单起见，举了无向图且边没有权重的例子）而言，它的度矩阵 ![[公式]](https://www.zhihu.com/equation?tex=D) ，邻接矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A) 和拉普拉斯矩阵 ![[公式]](https://www.zhihu.com/equation?tex=L) 分别如下图所示，==度矩阵 ![[公式]](https://www.zhihu.com/equation?tex=D) 只有对角线上有值，为对应节点的度，其余为0==；==邻接矩阵![[公式]](https://www.zhihu.com/equation?tex=A)只有在有边连接的两个节点之间为1，其余地方为0==；==拉普拉斯矩阵 ![[公式]](https://www.zhihu.com/equation?tex=L) 为 ![[公式]](https://www.zhihu.com/equation?tex=D-A)== 。但需要注意的是，这是最简单的一种拉普拉斯矩阵，除了这种定义，还有接下来介绍的几种拉普拉斯矩阵。

![img](https://pic1.zhimg.com/80/v2-bc36fb838241c8f1e34b5d913d9b459c_hd.jpg)

​															图5 一个图的度矩阵，邻接矩阵 和 拉普拉斯矩阵

### 图卷积的通式

任何一个图卷积层都可以写成这样一个非线性函数：

![[公式]](https://www.zhihu.com/equation?tex=H%5E%7Bl%2B1%7D+%3D+f%28H%5E%7Bl%7D%2CA%29)

![[公式]](https://www.zhihu.com/equation?tex=H%5E%7B0%7D%3DX) 为第一层的输入， ![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+R%5E%7BN%2AD%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=N) 为图的节点个数， ![[公式]](https://www.zhihu.com/equation?tex=D) 为每个节点特征向量的维度， ![[公式]](https://www.zhihu.com/equation?tex=A) 为邻接矩阵，不同模型的差异点在于函数 ![[公式]](https://www.zhihu.com/equation?tex=f) 的实现不同。

下面介绍几种具体的实现，但是每一种实现的参数大家都统称拉普拉斯矩阵。

### 实现一

![[公式]](https://www.zhihu.com/equation?tex=H%5E%7Bl%2B1%7D+%3D+%5Csigma+%28AH%5E%7Bl%7DW%5E%7Bl%7D%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=W%5E%7Bl%7D) 为第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的权重参数矩阵， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28%5Ccdot%29) 为非线性激活函数，例如ReLU。

这种思路是基于节点特征与其所有邻居节点有关的思想。邻接矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A) 与特征 ![[公式]](https://www.zhihu.com/equation?tex=H) 相乘，等价于，某节点的邻居节点的特征相加。这样多层隐含层叠加，能利用多层邻居的信息。

但这样存在两个问题：

- **没有考虑节点自身对自己的影响；**
- **邻接矩阵**![[公式]](https://www.zhihu.com/equation?tex=A)**没有被规范化，这在提取图特征时可能存在问题，比如邻居节点多的节点倾向于有更大的影响力。**

因此实现二和实现三针对这两点进行了优化。

### 实现二

![[公式]](https://www.zhihu.com/equation?tex=H%5E%7Bl%2B1%7D+%3D+%5Csigma+%28LH%5E%7Bl%7DW%5E%7Bl%7D%29)

==拉普拉斯矩阵 ![[公式]](https://www.zhihu.com/equation?tex=L%3DD-A) ，学名Combinatorial Laplacian==，是针对实现一的问题1的改进：

- ==引入了度矩阵，从而解决了没有考虑自身节点信息自传递的问题==。

### 实现三

![[公式]](https://www.zhihu.com/equation?tex=H%5E%7Bl%2B1%7D+%3D+%5Csigma+%28D%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Chat%7BA%7DD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7DH%5E%7Bl%7DW%5E%7Bl%7D%29)

==对于这里的拉普拉斯矩阵 ![[公式]](https://www.zhihu.com/equation?tex=L%5E%7Bsym%7D%3DD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Chat%7BA%7DD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D%3DD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D%28D-A%29D%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D%3DI_n-D%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7DAD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D) ，学名Symmetric normalized Laplacian==，也有论文或者博客写 ![[公式]](https://www.zhihu.com/equation?tex=L%3DI_n%2BD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7DAD%5E+%7B-%5Cfrac%7B1%7D%7B2%7D%7D) ， 就是一个符号的差别，但本质上还是实现一的两个问题进行的改进：

- **引入自身度矩阵，解决自传递问题；**
- **对邻接矩阵的归一化操作，通过对邻接矩阵两边乘以节点的度开方然后取逆得到。**具体到每一个节点对 ![[公式]](https://www.zhihu.com/equation?tex=i%EF%BC%8Cj) ，矩阵中的元素由下面的式子给出（对于无向无权图）：

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bi%2C+j%7D%5E%7B%5Cmathrm%7Bsym%7D%7D%3A%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%7B1%7D+%26+%7B%5Ctext+%7B+if+%7D+i%3Dj+%5Ctext+%7B+and+%7D+%5Coperatorname%7Bdeg%7D%5Cleft%28v_%7Bi%7D%5Cright%29+%5Cneq+0%7D+%5C%5C+%7B-%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Coperatorname%7Bdeg%7D%5Cleft%28v_%7Bi%7D%5Cright%29+%5Coperatorname%7Bdeg%7D%5Cleft%28v_%7Bj%7D%5Cright%29%7D%7D%7D+%26+%7B%5Ctext+%7B+if+%7D+i+%5Cneq+j+%5Ctext+%7B+and+%7D+v_%7Bi%7D+%5Ctext+%7B+is+adjacent+to+%7D+v_%7Bj%7D%7D+%5C%5C+%7B0%7D+%26+%7B%5Ctext+%7B+otherwise.+%7D%7D%5Cend%7Barray%7D%5Cright.)

其中 ![[公式]](https://www.zhihu.com/equation?tex=deg%28v_i%29%2Cdeg%28v_j%29) 分别为节点 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 的度，也就是度矩阵在节点 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 处的值。

**可能有一点比较疑惑的是怎么两边乘以一个矩阵的逆就归一化了？**这里需要复习到矩阵取逆的本质是做什么。

我们回顾下矩阵的逆的定义，对于式子 ![[公式]](https://www.zhihu.com/equation?tex=A%2AX%3DB) ，假如我们希望求矩阵X，那么当然是令等式两边都乘以 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B-1%7D) ，然后式子就变成了 ![[公式]](https://www.zhihu.com/equation?tex=X%3DA%5E%7B-1%7D%2AA%2AX%3DA%5E%7B-1%7DB) 。

举个例子对于，单个节点运算来说，做归一化就是除以它节点的度，这样每一条邻接边信息传递的值就被规范化了，不会因为某一个节点有10条边而另一个只有1条边导致前者的影响力比后者大，因为做完归一化后者的权重只有0.1了，==从单个节点上升到二维矩阵的运算，就是对矩阵求逆了，乘以矩阵的逆的本质，就是做矩阵除法完成归一化。但左右分别乘以节点i,j度的开方，就是考虑一条边的两边的点的度。==

常见的拉普拉斯矩阵除了以上举的两种，还有![[公式]](https://www.zhihu.com/equation?tex=L%5E%7Brw%7D%3DD%5E+%7B-1%7D%5Chat%7BA%7D%3DD%5E+%7B-1%7D%28D-A%29%3DI_n-D%5E%7B-1%7DA) 等等，归一化的方式有差别，根据论文[5]的实验，==这些卷积核的形式并没有一种能够在任何场景下比其他的形式效果好，因此在具体使用的时候可以进行多种尝试，但主流的还是实现三，也就是大多数博客提到的==。

### 另一种表述

上面是以矩阵的形式计算，可能会看起来非常让人疑惑，下面从单个节点的角度来重新看下这些个公式（本质是一样的，上文解释过，对于单个节点就是除法，对于矩阵就是乘以度矩阵的逆），对于第 ![[公式]](https://www.zhihu.com/equation?tex=l%2B1) 层的节点的特征 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bl%2B1%7D_i) ，对于它的邻接节点 ![[公式]](https://www.zhihu.com/equation?tex=j%5Cin%7BN%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=N) 是节点 ![[公式]](https://www.zhihu.com/equation?tex=i) 的所有邻居节点的集合，可以通过以下公式计算得到：

![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bl%2B1%7D_%7Bv_i%7D%3D%5Csigma%28%5Csum_j%7B%5Cfrac%7B1%7D%7Bc_%7Bij%7D%7Dh%5El_%7Bv_j%7DW%5E%7Bl%7D%7D%29)

其中， ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bi%2Cj%7D%3D%5Csqrt%7Bd_id_j%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=j%5Cin+N%7Bi%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=N%7Bi%7D) 为 ![[公式]](https://www.zhihu.com/equation?tex=i) 的邻居节点， ![[公式]](https://www.zhihu.com/equation?tex=d_i%2Cd_j) 为 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 的度，这跟上面的公式其实是等价的，所以有些地方的公式是这个，有些的上面那个。

### 代码上手

代码上手有最简单的numpy实现的例子，有pytorch的原生实现，也有利用库pyg的实现，具体可以看了解更多的相关链接。

一个简单的NUMPY例子

[https://mp.weixin.qq.com/s/sg9O761F0KHAmCPOfMW_kQ](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/sg9O761F0KHAmCPOfMW_kQ)

pyg库上手教程，推荐，还有形象化的图来理解向量计算

https://zhuanlan.zhihu.com/p/78452993

PYG的gayhub地址和官方文档地址

[https://github.com/rusty1s/pytorch_geometric](https://link.zhihu.com/?target=https%3A//github.com/rusty1s/pytorch_geometric)

[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://link.zhihu.com/?target=https%3A//pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

用原生pytorch实现GCN

[https://github.com/LYuhang/GNN_Review/blob/master/PyG%E5%92%8CPytorch%E5%AE%9E%E7%8E%B0GNN%E6%A8%A1%E5%9E%8B/GNN_Implement_with_Pytorch.ipynb](https://link.zhihu.com/?target=https%3A//github.com/LYuhang/GNN_Review/blob/master/PyG%E5%92%8CPytorch%E5%AE%9E%E7%8E%B0GNN%E6%A8%A1%E5%9E%8B/GNN_Implement_with_Pytorch.ipynb)



### 参考文献

[1]https://zhuanlan.zhihu.com/p/77729049 图嵌入的分类

[2]https://www.zhihu.com/question/54504471/answer/630639025 关于图卷积的物理学解释

[3]https://www.zhihu.com/question/54504471/answer/332657604 拉普拉斯公式推倒细节，包括谱分解和傅立叶变换

[4] https://link.zhihu.com/?target=http%3A//tkipf.github.io/graph-convolutional-networks/ 两篇论文细讲

[5][https://github.com/conferencesub/ICLR_2020](https://link.zhihu.com/?target=https%3A//github.com/conferencesub/ICLR_2020) 各种图卷积核比较

[6][https://persagen.com/files/misc/scarselli2009graph.pdf](https://link.zhihu.com/?target=https%3A//persagen.com/files/misc/scarselli2009graph.pdf) GNN首次被提出的paper

[7]https://zhuanlan.zhihu.com/p/85287578 拉普拉斯矩阵和拉普拉斯算子的关系

https://link.zhihu.com/?target=https%3A//github.com/benedekrozemberczki/awesome-graph-classification)

编辑于 2019-11-16