# 模型推理加速技巧：融合BN和Conv层

文章来源：https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3

文章地址：https://zhuanlan.zhihu.com/p/110552861

很多深度模型采用**BN层（Batch Normalization）**被很多深度模型来提升泛化能力。在模型推理时，BN层要从训练状态切换到测试状态，此时采用模型训练中近似的均值和方差。BN层最酷的地方是它可以用一个1x1卷积等效替换，更进一步地，我们可以将BN层合并到前面的卷积层中。

---

其他相关论文： Restructuring Batch Normalization to Accelerate CNN Training

论文地址：https://arxiv.org/abs/1807.01702

https://proceedings.mlsys.org/book/270.pdf

13 pages, 8 figures, to appear in SysML 2019, added ResNet-50 results

Batch Normalization (BN) has become a core design block of modern Convolutional Neural Networks (CNNs). A typical modern CNN has a large number of BN layers in its lean and deep architecture. BN requires mean and variance calculations over each mini-batch during training. Therefore, the existing memory access reduction techniques, such as fusing multiple CONV layers, are not effective for accelerating BN due to their inability to optimize mini-batch related calculations during training. To address this increasingly important problem, we propose to restructure BN layers by first splitting a BN layer into two sub-layers (fission) and then combining the first sub-layer with its preceding CONV layer and the second sub-layer with the following activation and CONV layers (fusion). The proposed solution can significantly reduce main-memory accesses while training the latest CNN models, and the experiments on a chip multiprocessor show that the proposed BN restructuring can improve the performance of DenseNet-121 by 25.7%.



## Batch Normalization

这里假定![[公式]](https://www.zhihu.com/equation?tex=x)是网络的某一激活层特征，我们要对其进行归一化。若模型训练batch中共有![[公式]](https://www.zhihu.com/equation?tex=n)个样例，其特征分别是![[公式]](https://www.zhihu.com/equation?tex=%7Bx_1%2C+x_2%2C+%5Cldots%2C+x_n%7D) ，我们采用下列公式进行归一化： 
$$
\begin{aligned}\begin{gathered}\hat{x}_i = \gamma\frac{x_i - \mu}{\sqrt{\sigma^2 +
        \epsilon}} + \beta\\
        \hat{x}_i = \frac{\gamma x_i}{\sqrt{\sigma^2 +
          \epsilon}} + \beta - \frac{\gamma\mu}{\sqrt{\sigma^2 +
          \epsilon}} \end{gathered}
        \end{aligned}
$$
 这里![[公式]](https://www.zhihu.com/equation?tex=%5Cmu)和![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2)为这个batch上计算得到的均值和方差（在B,H,W维度上计算，每个channel单独计算），而 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 防止除零所设置的一个极小值， ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma)是放缩系数，而![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta)是平移系数。在训练过程中，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) and ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma)在当前batch上计算： ![[公式]](https://www.zhihu.com/equation?tex=+%5Cbegin%7Baligned%7D+%5Cbegin%7Bgather%7D+%5Cmu%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum+x_i%E2%80%8B%5C+%5Csigma%5E2%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum+%28x_i+-%5Cmu%29%5E2%5Cend%7Bgather%7D+%5Cend%7Baligned%7D+%5C%5C)

而参数![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma)和![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta)是和其它模型参数一起通过梯度下降方法训练得到。在测试阶段，我们不太会对一个batch图像进行预测，一般是对单张图像测试。因此，通过前面公式计算![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) and ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma)就不可能。BN的解决方案是采用训练过程中**指数移动平均值** $\hat{\mu}$ 和![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Csigma%7D%5E2)。

目前，BN层大部分用于CNN网络中，此时上面的4个参数就是针对特征图各个通道的参数，这里我们记![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_c), ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2_c), ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma_c)以及![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_c)为第![[公式]](https://www.zhihu.com/equation?tex=c)个通道的参数。

## 融合方案

**首先我们将测试阶段的BN层（一般称为frozen BN）等效替换为一个1x1卷积层：**

对于一个形状为![[公式]](https://www.zhihu.com/equation?tex=C%5Ctimes+H%5Ctimes+W)的特征图![[公式]](https://www.zhihu.com/equation?tex=F)，归一化后的结果![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BF%7D)，可以如下计算：

![1583310947050](D:\Notes\raw_images\1583310947050.png)

这里的![[公式]](https://www.zhihu.com/equation?tex=i%2C+j)是特征图的各个空间位置，我们可以看到上述计算就是![[公式]](https://www.zhihu.com/equation?tex=f%28x%29+%3D+W%2Ax+%2B+b)形式，其可以看成是![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1)卷积，这里W的行向量就是每个对应输出通道的卷积核（![[公式]](https://www.zhihu.com/equation?tex=1%5E2%5Ccdot+C)）。由于BN层常常在Conv层之后，我们可以进行两个操作的合并。

**然后我们将BN层和Conv层融合：** 这里我们记：

- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7BBN%7D%5Cin%5Cmathbb%7BR%7D%5E%7BC%5Ctimes+C%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bb%7D_%7BBN%7D%5Cin%5Cmathbb%7BR%7D%5E%7BC%7D)为BN的参数,
- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7Bconv%7D%5Cin%5Cmathbb%7BR%7D%5E%7BC%5Ctimes%28C_%7Bprev%7D%5Ccdot+k%5E2%29%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bb%7D_%7Bconv%7D%5Cin%5Cmathbb%7BR%7D%5E%7BC%7D)是BN前面Conv层的参数,
- ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bprev%7D)为Conv层的输入，
- ![[公式]](https://www.zhihu.com/equation?tex=C_%7Bprev%7D)为输入层的channel数，
- ![[公式]](https://www.zhihu.com/equation?tex=k)是Conv层的卷积核大小

我们将![[公式]](https://www.zhihu.com/equation?tex=F_%7Bprev%7D)的每个卷积![[公式]](https://www.zhihu.com/equation?tex=k%5Ctimes+k)部分reshape为一个维度为 ![[公式]](https://www.zhihu.com/equation?tex=k%5E2%5Ccdot+C_%7Bprev%7D)的向量![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7Bi%2Cj%7D)，因此Conv层加BN层的操作为：

![[公式]](https://www.zhihu.com/equation?tex=+%5Cmathbf%7B%5Chat%7Bf%7D%7D_%7Bi%2Cj%7D%3D+++++++++%5Cmathbf%7BW%7D_%7BBN%7D%5Ccdot+%28%5Cmathbf%7BW%7D_%7Bconv%7D%5Ccdot%5Cmathbf%7Bf%7D_%7Bi%2Cj%7D+%2B+%5Cmathbf%7Bb%7D_%7Bconv%7D%29+%2B+%5Cmathbf%7Bb%7D_%7BBN%7D+%5C%5C)

显然，我们可以将Conv层和BN层合并成一个新的卷积层，其参数为：

- filter weights: ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D%3D%5Cmathbf%7BW%7D_%7BBN%7D%5Ccdot+%5Cmathbf%7BW%7D_%7Bconv%7D)
- bias: ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bb%7D%3D%5Cmathbf%7BW%7D_%7BBN%7D%5Ccdot%5Cmathbf%7Bb%7D_%7Bconv%7D%2B%5Cmathbf%7Bb%7D_%7BBN%7D)

**最后我们在PyTorch中实现这个融合操作：** nn.Conv2d参数:

- filter weights, ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D) : conv.weight;
- bias, ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bb%7D): conv.bias;

nn.BatchNorm2d参数:

- scaling, ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) : bn.weight;
- shift, ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta): bn.bias;
- mean estimate, ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cmu%7D): bn.running_mean;
- variance estimate, ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Csigma%7D%5E2): bn.running_var;
- ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon)(为了数值稳定性): bn.eps.

具体的实现代码如下（**Google Colab**）:

```python
import torch
    import torchvision
    
    def fuse(conv, bn):
    
        fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
    
        # setting weights
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        fused.weight.copy_( torch.mm(w_bn, w_conv).view(fused.weight.size()) )
        
        # setting bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros( conv.weight.size(0) )
        b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
                              torch.sqrt(bn.running_var + bn.eps)
                            )
        fused.bias.copy_( b_conv + b_bn )
    
        return fused
    
    # Testing
    # we need to turn off gradient calculation because we didn't write it
    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    resnet18 = torchvision.models.resnet18(pretrained=True)
    # removing all learning variables, etc
    resnet18.eval()
    model = torch.nn.Sequential(
        resnet18.conv1,
        resnet18.bn1
    )
    f1 = model.forward(x)
    fused = fuse(model[0], model[1])
    f2 = fused.forward(x)
    d = (f1 - f2).mean().item()
    print("error:",d)
```

运行代码，会发现融合BN和Conv层之后推理结果是一样，所以是等效替换。另外也可以对比前后推理时间的差异，会发现融合后推理时间会减少。

 