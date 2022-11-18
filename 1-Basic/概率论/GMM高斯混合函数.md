**高斯混合模型（Gaussian Mixture Model, GMM )**

高斯密度函数估计是一种参数化模型。高斯混合模型是单一高斯概率密度函数的延伸，GMM能够平滑地近似任意形状的密度分布。高斯混合模型种类有单高斯模型（Single Gaussian Model, SGM）和高斯混合模型（Gaussian Mixture Model, GMM）两类。类似于聚类，根据高斯概率密度函数（Probability Density Function, PDF）参数不同，每一个高斯模型可以看作一种类别，输入一个样本x，即可通过PDF计算其值，然后通过一个阈值来判断该样本是否属于高斯模型。很明显，SGM适合于仅有两类别问题的划分，而GMM由于具有多个模型，划分更为精细，适用于多类别的划分，可以应用于复杂对象建模。 

**混合模型（Mixture Model）**

混合模型是一个可以用来表示在总体分布（distribution）中含有 K 个子分布的概率模型，换句话说，混合模型表示了观测数据在总体中的概率分布，它是一个由 K 个子分布组成的混合分布。混合模型不要求观测数据提供关于子分布的信息，来计算观测数据在总体分布中的概率。

**单高斯模型**

当样本数据 X 是一维数据（Univariate）时，高斯分布遵从下方概率密度函数（Probability Density Function）：
$$
P(x|\theta) = \frac{1}{\sqrt{2\pi\sigma^{2}}} exp(-\frac{(x-\mu)^2}{2\sigma^{2}})
$$
其中$ \mu $为数据均值（期望），$ \sigma$为数据标准差（Standard Deviation）， $ \sigma^2$ 为方差 （Square Deviation）。

当样本数据 X 是多维数据（Multivariate）时，高斯分布遵从下方概率密度函数：

$$
P(x|\theta) = \frac{1}{(2\pi)^{\frac{D}{2}}\left| \Sigma \right|^{\frac{1}{2}}}exp(-\frac{(x-\mu)^{T}\Sigma^{-1}(x-\mu)}{2})
$$
其中，$ \mu $为数据均值（期望）， $ \Sigma $为协方差（Covariance），D 为数据维度。



**高斯混合模型**

高斯混合模型可以看作是由 K 个单高斯模型组合而成的模型，这 K 个子模型是混合模型的隐变量（Hidden variable）。一般来说，一个混合模型可以使用任何概率分布，这里使用高斯混合模型是因为高斯分布具备很好的数学性质以及良好的计算性能。

举个不是特别稳妥的例子，比如我们现在有一组狗的样本数据，不同种类的狗，体型、颜色、长相各不相同，但都属于狗这个种类，此时单高斯模型可能不能很好的来描述这个分布，因为样本数据分布并不是一个单一的椭圆，所以用混合高斯分布可以更好的描述这个问题，如下图所示：

![1562119132940](D:\Notes\raw_images\1562119132940.png) 

图中每个点都由 K 个子模型中的某一个生成

首先定义如下信息：

$ x_j $表示第$ j $个观测数据，$ j = 1,2,3,..., N $
$ K$ 是混合模型中子高斯模型的数量，$ k = 1,2,3,...,K $
$ \alpha_k $ 是观测数据属于第 $ k $个子模型的概率， $ \alpha_k \ge 0 $ ， $ \sum_{k=1}^K \alpha_k = 1$
$\phi(x|\theta_{k})$ 是第 $ k$ 个子模型的高斯分布密度函数，$\theta_k = (\mu_k, \sigma_k^2) $ 。其展开形式与上面介绍的单高斯模型相同。
$ \gamma_{jk} $ 表示第$ j $个观测数据属于第$ k $个子模型的概率
高斯混合模型的概率分布为：
$$
 P(x|\theta) = \sum_{k=1}^{K}{\alpha_{k}\phi(x|\theta_{k})}  
$$
对于这个模型而言，参数$\theta = (\tilde{\mu_{k}}, \tilde{\sigma_{k}}, \tilde{\alpha_{k}})$ ，也就是每个子模型的期望、方差（或协方差）、在混合模型中发生的概率。



**模型参数学习**

对于单高斯模型，我们可以用最大似然法（Maximum likelihood）估算参数 $ \theta$  的值

$ \theta = argmax_{\theta} L(\theta)$ 

这里我们假设了每个数据点都是独立的（Independent），似然函数由概率密度函数（PDF）给出。

$ L(\theta) = \prod_{j=1}^{N}P(x_{j}|\theta) $ 

由于每个点发生的概率都很小，乘积会变得极其小，不利于计算和观察，因此通常我们用 Maximum Log-Likelihood 来计算（因为 Log 函数具备单调性，不会改变极值的位置，同时在 0-1 之间输入值很小的变化可以引起输出值相对较大的变动）：
$$
 logL(\theta) = \sum_{j=1}^{N}{logP(x_{j}|\theta)}
$$
对于高斯混合模型，Log-Likelihood 函数是
$$
 logL(\theta) = \sum_{j=1}^{N}{logP(x_{j}|\theta)} = \sum_{j=1}^{N}{log(\sum_{k=1}^{K}{\alpha_{k}\phi(x|\theta_{k})})}
$$
如何计算高斯混合模型的参数呢？这里我们无法像单高斯模型那样使用最大似然法来求导求得使 likelihood 最大的参数，因为对于每个观测数据点来说，事先并不知道它是属于哪个子分布的（hidden variable），因此 log 里面还有求和， K 个高斯模型的和不是一个高斯模型，对于每个子模型都有未知的 $ \alpha_{k}, \mu_{k}, \sigma_{k} $ ，直接求导无法计算。需要通过迭代的方法求解。



**EM 算法**

EM 算法是一种迭代算法，1977 年由 Dempster 等人总结提出，用于含有隐变量（Hidden variable）的概率模型参数的最大似然估计。
每次迭代包含两个步骤：
E-step：求期望 $ E(\gamma_{jk} | X, \theta)$, for all $ j = 1,2,...,N $
M-step：求极大，计算新一轮迭代的模型参数。

这里不具体介绍一般性的 EM 算法，只介绍怎么在高斯混合模型里应用从来推算出模型参数。
通过 EM 迭代更新高斯混合模型参数的方法（我们有样本数据$ x_{1}, x_{2}, ...,x_{N} $和一个有$ K $ 个子模型的高斯混合模型，想要推算出这个高斯混合模型的最佳参数）：
首先初始化参数
E-step：依据当前参数，计算每个数据 j 来自子模型 k 的可能性
$$
\gamma_{jk} = \frac{\alpha_{k}\phi(x_{j}|\theta_{k})}{\sum_{k=1}^{K}{\alpha_{k}\phi(x_{j}|\theta_{k})}}, j = 1,2,...,N; k = 1,2,...,K
$$
M-step：计算新一轮迭代的模型参数
$$
\mu_{k} = \frac{\sum_{j}^{N}{(\gamma_{jk}}x_{j})}{\sum_{j}^{N}{\gamma_{jk}}}, k=1,2,...,K
$$

$$
\Sigma_{k} = \frac{\sum_{j}^{N}{\gamma_{jk}}(x_{j}-\mu_{k})(x_{j}-\mu_{k})^{T}}{\sum_{j}^{N}{\gamma_{jk}}}, k = 1,2,...,K   （用这一轮更新后的 \mu_{k} )
$$

$$
\alpha_{k} = \frac{\sum_{j=1}^{N}{\gamma_{jk}}}{N}, k=1,2,...,K
$$

重复计算 E-step 和 M-step 直至收敛 （ $||\theta_{i+1} - \theta_{i}|| < \varepsilon$ , $\varepsilon$ 是一个很小的正数，表示经过一次迭代之后参数变化非常小）。

至此，我们就找到了高斯混合模型的参数。需要注意的是，EM 算法具备收敛性，但并不保证找到全局最大值，有可能找到局部最大值。解决方法是初始化几次不同的参数进行迭代，取结果最好的那次。

