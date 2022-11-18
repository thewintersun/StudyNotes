## DARTS: Differentiable Architecture Search

论文地址：https://arxiv.org/abs/1806.09055

作者：Hanxiao Liu, Karen Simonyan, Yiming Yang

机构：CMU, Google

发表： ICLR 2019

代码：https://github.com/quark0/darts



### 摘要

This paper addresses the scalability challenge of architecture search by formulating the task in a differentiable manner. Unlike conventional approaches of applying evolution or reinforcement learning over a discrete and non-differentiable search space, our method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent. 

本文用可微的方法解决网络结构搜索的扩展性问题。与传统的基于进化和强化学习的算法（Over离散不可谓的搜索空间）不同，我们的方法是基于连续的 relaxation of the architecture representation, 这样可以使用梯度下降方法来更高效地进行结构搜索。

Extensive experiments on CIFAR-10, ImageNet, Penn Treebank and WikiText-2 show that our algorithm excels in discovering high-performance convolutional architectures for image classification and recurrent architectures for language modeling, while being orders of magnitude faster than state-of-the-art non-differentiable techniques. 

### Differentiable Architecture Search

设一组有序node： $x^{(1)}, x^{(2)},..., x^{(N)}$ 。其中，操作 o^{(i, j)} 代表从node i 到node j 的操作。在NAS中，node j 是由所有小于它的node经过操作而得到，如论文中公式（1）： 
$$
x^{(j)}=\sum_{\color{red}{i<}j} o^{\color{red}(i, j)} \left( x^{(j)} \right)
$$
在NAS中，操作$o^{(i, j)}$通常是从一组备选candidate中进行选择，如本篇论文中，就是从 $3\times3\ \text{conv}$ 、 $5\times5\ \text{conv}$ 、 $3\times3\ \text{max pool}$ 、 $3\times3\ \text{avg pool}$ 等中进行选择。因此，对于每一组操作 $o^{(i, j)}$ ，我们都定义一组系数 $\alpha=\Big\{\alpha^{(i, j)}\Big\}$ 。从而，在训练过程中，实际上我们使用的是混合操作 $\overline{o}^{(i, j)}$ ，如公式（2）所示：
$$
\overline{o}^{(i, j)}= \sum_{\color{blue}{o \in \mathcal{O}}} {\frac{\exp{(\alpha_{\color{blue}o}^{(i, j)})}} {\sum_{\color{red}{o' \in \mathcal{O}}}{\exp{(\alpha_{\color{red}{o'}}^{(i, j)})}}} \color{blue}o(x)}
$$
 其实也简单，就是利用 $\alpha=\Big\{\alpha^{(i, j)}\Big\}$ 做一个softmax激活，然后对各操作作加权平均罢了。

![1592553736576](D:\Notes\raw_images\1592553736576.png)

图1:DARTS概述:(a)在edges的操作最初是未知的。(b)通过在每条边上混合放置候选操作来连续relaxation搜索空间。(c)通过求解二层优化问题，实现混合概率和网络权值的联合优化。(d)从学到的混合概率中归纳出最终的结构。这里还包括一个特殊的==零操作,表明两个节点之间缺乏连接==。

我们假设上述所说的 $\alpha=\Big\{\alpha^{(i, j)}\Big\}$ 是与NAS有关的参数，另记 $\omega$ 为模型参数（即卷积的weight/bias等可训练参数），那么我们的目标是找到最优结构 $\alpha^*$ ，使得在验证集上得到最优，即最小化 $\color{red}{\mathcal{L}_{val}}(\omega^*, \alpha^*)$ ，并找到最优参数 $\omega^*$ ，能够在特定结构 $\alpha^*$ 上得到最优性能，即最小化训练集 $\omega^*=\text{argmin}_\omega \color{blue}{\mathcal{L}_{train}} (\omega, \alpha^*)$ 。
上述过程可形式化记为（公式（3））：
$$
\begin{align} \min_\alpha & \  \color{red}{\mathcal{L}_{val}} \Big(\omega^*(\alpha), \alpha \Big) \\ \text{s.t.} & \ \omega^*(\alpha)=\text{argmin}_\omega  \color{blue}{\mathcal{L}_{train}}(\omega, \alpha) \end{align}
$$
准备工作到此结束，后面主要推（7，8）两个公式。

![1592558915674](D:\Notes\raw_images\1592558915674.png)

本篇文章最大的贡献是，对于上面的形式化优化目标，进行了一步简单的近似, 公式（5, 6）：

假设在第k步，给定当前网络结构 $α_{k-1}$ ,我们通过 $\mathcal{L}_{train}(w_{k-1},α_{k-1})$ 计算梯度更新得到 $w_k$ 。然后固定 $w_k$, 通过更新网络结构$\alpha_k$来最小化验证集损失值(公式3)，其中 $\xi$ 表示学习率。
$$
\begin{align} & \nabla_\alpha \mathcal{L}_{val} \Big(\color{red}{\omega^*(\alpha)}, \alpha \Big) \\  \approx & \nabla_\alpha \mathcal{L}_{val} \Big(\color{red}{\omega-\xi\nabla_\omega \mathcal{L}_{train}(\omega, \alpha)}, \alpha \Big) \end{align}
$$
这种“一步优化”的骚操作，我在Nesterov，MAML等方法中都见过，所以作者做这一近似也可说是合情合理。正如作者在论文中所讲，当 $\omega$ 已经到达局部极值点时， $\nabla_\omega \mathcal{L}_{train}(\omega, \alpha)=0$ ，此时， $\omega=\omega^*$ 。

那么说了这么一大堆，其实就是为了引出下面2个公式。首先，作者是说使用chain rule可以将上面近似后的式子化简为：
$$
\begin{align} &  \color{red}{\nabla_\alpha} \mathcal{L}_{val} \Big(\omega-\xi\nabla_\omega \mathcal{L}_{train}(\omega, \color{red}{\alpha}), \color{red}{\alpha}\Big) \\ = & \color{blue}{\nabla_\alpha} \mathcal{L}_{val} (\omega', \color{blue}{\alpha}) - \xi \nabla_{\alpha, \omega}^2 \mathcal{L}_{train}(\omega, \alpha) \cdot \color{blue}{\nabla_{\omega'}} \mathcal{L}_{val}(\color{blue}{\omega'}, \alpha) \end{align}
$$
其中： $\omega'= \omega-\xi\nabla_\omega \mathcal{L}_{train}(\omega, \alpha)$ 。

看上去有点晕？其实想明白很简单。

第一行的式子，实际上相当于是一个==关于 $\alpha$ 的复合函数求导==，我们可以将其形式化记为： $\nabla_\alpha f(g_1(\alpha), g_2(\alpha))$  ，其中 $f(\cdot, \cdot)=\mathcal{L}_{val}(\cdot, \cdot) ， g_1(\alpha)= \omega-\xi\nabla_\omega \mathcal{L}_{train}(\omega, \alpha) ， g_2(\alpha)=\alpha$ 。

清楚了吧？这就是一个简单的复合函数求导，别被唬住了。我这里简单的搬运一下维基百科中，有关于这个的一段解释好了：

> 多元函数与多元函数复合的情形
> 若函数 $u=ϕ(x,y)、v=ψ(x,y)$ 都在点 $(x,y)$ 具有对 $x、y$ 的偏导数，函数$z=f(u,v)$在对应点$(u,v)$具有连续偏导数，那么复合函数$z=f[ϕ(x,y),ψ(x,y)]$在点 $(x,y)$ 的两个偏导数都存在，则对应

$$
z=f(u,v),
\begin{cases}
u = \phi(x,y)\\
v = \psi(x,y)
\end{cases}
$$

有
$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} +\frac{\partial z}{\partial v}\frac{\partial v}{\partial x}
$$
OK！我们一步步来：
$$
\begin{align} & \nabla_\alpha f\Big(g_1(\alpha), g_2(\alpha)\Big) \\ = &\color{red}{\nabla_\alpha g_1(\alpha)} \cdot \color{blue}{D_1 f\Big(g_1(\alpha), g_2(\alpha)\Big)} + \color{red}{\nabla_\alpha g_2(\alpha)} \cdot \color{blue}{D_2 f\Big(g_1(\alpha), g_2(\alpha)\Big)} \end{align}
$$
剩下的就容易说了，
$$
\begin{aligned} & \nabla_\alpha g_1(\alpha) = -\xi \nabla_{\alpha, \omega}^2 \mathcal{L}_{train}(\omega, \alpha) \\ & \nabla_\alpha g_2(\alpha) =1 \\ \end{aligned}
$$

$$
\begin{aligned} & D_1 f\Big(g_1(\alpha), g_2(\alpha)\Big) = \nabla_{\omega'} \mathcal{L}_{val}(\omega', \alpha) \\ & D_2 f\Big(g_1(\alpha), g_2(\alpha)\Big) = \nabla_{\alpha} \mathcal{L}_{val}(\omega', \alpha) \end{aligned}
$$

齐活！

这里还要再强调一下，有的同学（比如我），在看到这个公式时会产生疑问：

![1592813210550](D:\Notes\raw_images\1592813210550.png)

您这第一行和第二行第一项一样啊，莫非？打住！我一开始也被蒙蔽了，但后来才反应过来，第一行对 $\alpha$ 求导是一个复合函数求导，也就是要同时兼顾 $\mathcal{L}_{val}$ 函数的第一项和第二项（看我红颜色）。而第二行第一项实际上是 $D_2 f(g_1(\alpha), g_2(\alpha))$ ，也就是说虽然$\mathcal{L}_{val}$里面，第一项也是关于 $\alpha$ 的函数，但我们只对函数里，第二项求导（看我蓝颜色）。

接下来还有一个公式（8）是说，上面的这个二阶导乘以一阶导，这么一个看上去巨复杂的式子，我们可以使用一发差分公式求得：
$$
\nabla_{\alpha, \omega}^2 \mathcal{L}_{train}(\omega, \alpha) \cdot \nabla_{\omega'} \mathcal{L}_{val}(\omega', \alpha) \approx \frac{\nabla_\alpha \mathcal{L}_{train}(\omega^+, \alpha) - \nabla_\alpha \mathcal{L}_{train}(\omega^-, \alpha)}{2\epsilon}
$$
其中， $\omega^\pm = \omega \pm \epsilon \nabla_{\omega'} \mathcal{L}_{val}(\omega', \alpha)$ 。

OK！其实这个化简说神也真神，不过只用到基本的泰勒展开。
$$
f(x_0+h)=f(x_0)+\frac{f'(x_0)}{1!}h + ...
$$
如果我们在 $h$ 这边，动点手脚，用 $hA$ 来替换，那么很容易得到：
$$
\begin{aligned} & f(x_0+hA)=f(x_0)+\frac{f'(x_0)}{1!}hA + ... \\ & f(x_0-hA)=f(x_0)-\frac{f'(x_0)}{1!}hA + ... \\ \end{aligned}
$$
那么，两式相减，很容易就得到：
$$
f'(x_0)\cdot A\approx \frac{f(x_0+hA)-f(x_0-hA)}{2h}
$$
这回熟悉了吗？如果我们把 $h$ 换成 $\epsilon$  ，把 $A$ 换成 $\nabla_{\omega'} \mathcal{L}_{val}(\omega', \alpha)$ ，把 $x_0$ 换成 $\omega$ ，再把 $f$ 换成 $\nabla_\alpha \mathcal{L}_{train}{(\cdot, \cdot)}$ ，就得到我们的公式（8）了。

至此，整篇文章的公式推导完毕。

### EXPERIMENTS AND RESULTS

为了在离散体系结构中形成每个节点，我们从之前所有节点收集的所有非零候选操作中保留前k个最强的操作(来自不同的节点)。对于卷积结构K=2, 对于RNN结构K=1。

我们在CIFAR-10和PTB上的实验包括两个阶段:架构搜索(3.1节)和架构评估(3.2节)。在第一阶段，我们使用DARTS搜索单元架构，并根据其验证的性能来确定最佳Cells。在第二阶段中,我们使用这些Cells来构建更大的架构,我们从头开始训练并Report他们在测试集上的表现。 

Figure: Snapshots of the most likely normal conv, reduction conv, and recurrent cells over time.

![progress_convolutional_normal.gif](https://github.com/quark0/darts/blob/master/img/progress_convolutional_normal.gif?raw=true)![progress_convolutional_reduce.gif](https://github.com/quark0/darts/blob/master/img/progress_convolutional_reduce.gif?raw=true)

![progress_recurrent](https://github.com/quark0/darts/raw/master/img/progress_recurrent.gif)



### 实验结果

CIFAR-10

![1592554043173](D:\Notes\raw_images\1592554043173.png)

CIFAR-10迁移到ImageNet上

![1592554248437](D:\Notes\raw_images\1592554248437.png)

PTB

![1592554201748](D:\Notes\raw_images\1592554201748.png)

