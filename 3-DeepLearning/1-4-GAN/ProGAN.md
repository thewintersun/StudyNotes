## Progressive Growing of GANs for Improved Quality, Stability, and Variation

论文地址：https://arxiv.org/abs/1710.10196

作者： Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen

机构： NVIDIA

发表：ICLR 2018 

文章地址：https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2

代码实现：https://github.com/tkarras/progressive_growing_of_gans

讲解视频：https://www.youtube.com/watch?v=ReZiqCybQPA



### 摘要

> We describe a new training methodology for generative adversarial networks. ==The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses==. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented (空前的) quality, e.g., CelebA images at $1024^2$. 
>
> We also ==propose a simple way to increase the variation in generated images==, and achieve a record inception score of 8.80 in unsupervised CIFAR10. 
>
> Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. 
>
> Finally, we ==suggest a new metric for evaluating GAN results==, both in terms of image quality and variation. 
>
> As an additional contribution, we construct a higher-quality version of the CelebA dataset.



## Toward Higher Image Resolutions

正是在这种背景下，NVIDIA的团队展示了由他们的新ProGAN生成的1024x1024的图像，这些图像的细节令人震惊。更好的是，他们不知道为什么他们的技术不能用于合成更高分辨率的图像。它甚至比以前的GANs更有效(在训练时间方面)。

### Growing GANs

与通常的做法不同，该团队并没有尝试一次训练所有层的生成器和鉴别器，而是逐渐增加了他们的GAN的网络层，一次一层，以逐步处理更高分辨率的图像。

![img](https://miro.medium.com/max/928/1*tUhgr3m54Qc80GU2BkaOiQ.gif)

ProGAN开始产生非常低的分辨率图像。当训练稳定时,添加一个新层,分辨率增加一倍。这将持续到输出达到所需的分辨率。通过这种逐步增加网络结构的方法, 首先学习到高层结构, 并且训练稳定。

为了做到这一点,他们首先将他们的训练图像缩小到一个非常小的开始分辨率(只有4x4像素)。他们创建了一个只有几个层的生成器来合成这个低分辨率的图像,以及一个相应的镜像架构的鉴别器。因为这些网络很小,它们的训练速度比较快,只学到了在模糊的图像中可见的大体结构。

当第一个层完成训练时,==他们将另一个层添加到G和D,将输出分辨率增加一倍,达到8x8。在早期的训练权重被保存,但没有被锁住,新层逐渐淡入,以帮助稳定过渡(稍后)。训练恢复到GAN再次合成令人信服的图像, 这次是在新的8x8分辨率==。

通过这种方式,它们继续添加层,加倍分辨率和训练直到达到所需的输出大小。

### The Effectiveness of Growing GANs

通过逐渐提高分辨率，我们不断地要求网络学习整个问题的“更简单的部分”。增量式学习过程极大地稳定了训练。这与我们将在下面讨论的一些训练细节相结合，减少了模式崩溃（mode collapse）的可能。

分辨率由低到高的趋势也迫使逐渐增长的网络，首先关注高层次的结构(在图像最模糊的版本中可以识别的模式)，然后填充细节。这通过减少网络出现高层结构严重错误的可能性来提高最终图像的质量。

与一次性初始化所有层的传统方法相比，逐步增加网络大小的计算效率也更高。更少的层可以更快地进行训练，因为其中只有更少的参数。除了最后的一组训练迭代之外，所有的迭代都是用最终层的子集完成的，这带来了一些不错的效率收益。Karras等人发现，根据输出分辨率的不同，他们的ProGAN通常比相应的传统GAN的训练速度快2-6倍。

## The Architecture

除了逐渐增长的网络，NVIDIA论文的作者还做了其他一些架构上的改变，以促进稳定、有效的训练。

给定分辨率*k*的生成器架构遵循一个熟悉的高层模式:每一组层使表示大小加倍，通道数量减半，直到输出层创建一个只有三个通道对应于RGB的图像。鉴别器做的几乎完全相反，将表示大小减半，并将每一组层的通道数量加倍。在这两个网络中，通过将 filters 的数量限制在一个合理的值(如512)来中断channel倍增模式，以防止参数总数过高。

在这个意义上，ProGAN类似于早期的图像生成GANs。DCGAN使用了类似的结构。

然而，DCGAN使用 *transpose convolutions* 来改变representation大小。相比之下，==ProGAN使用 *nearest neighbors* 进行 upscaling ，使用*average pooling* 进行降级==。这些都是简单的操作，没有需要学习的参数。然后接了两个卷积层。

![img](https://miro.medium.com/max/3022/1*lStHChxfyLB3S7wUW3Quiw.png)

生成器架构的详细视图，当它“增长”到分辨率k时。每一组层通过最近邻上采样操作和随后的两个卷积将分辨率大小翻倍。==为了稳定训练，最近添加的层被“淡入”。这个过程是由α 控制, 一个0和1之间的数值， 通过许多训练迭代线性增加, 直到新层完全到位==。

### “Fading in” New Layers

当网络在一个分辨率下训练完成时，通过添加一组新的层来将分辨率增加一倍，网络也逐步增长。添加新层时,前一层的参数仍然设置为可训练。

为了防止先前存在的较低的层受到突然增加的新顶层的冲击，顶层是线性“渐入”的。这种渐入是由一个参数控制*α*,它是通过许多训练迭代，从0到1线性插值的。如上图所示，最终生成的图像是生成器中最后一层和倒数第二层的加权和。

当增大分辨率时，Generator与Discriminator都会增加卷积层。为了让新加层快速收敛，同时又不对原有层造成过大的影响，ProGAN提出了一种Fade-in的机制。如上图所示，在Fade-in阶段时，旧有的层的输出经过上采样放大两倍，而后通过`toRGB`层转化为RGB图像，与新加层的输出通过`toRGB`层转化后的图像进行加权和，形成最终的输出。这一融合由一个参数α进行控制，$(x'=\alpha x_{i} + (1-\alpha) x_{i-1})$。随着训练的进行，α从1线性减少为0，最终输出也逐渐转为新加层的输出占主导。

```python
# fade-in phase in generator
upsample = F.interpolate(out, scale_factor=2, mode='nearest', align_corners=False)
out = self.progression[i](out)
out = self.to_rgb[i](out)
skip_rgb = self.to_rgb[i - 1](upsample)
out = (1 - alpha) * skip_rgb + alpha * out

# fade-in phase in discriminator
out = self.progression[i](out)
out = F.avg_pool2d(out, 2)
downsample = F.avg_pool2d(input, 2)
skip_rgb = self.from_rgb[i + 1](downsample)
out = (1 - alpha) * skip_rgb + alpha * out
```

### Pixel Normalization

作者采用 *pixel normalization* ，而不是 *batch normalization* 。==这个“pixelnorm”层没有可训练的权重==。它将每个像素的特征向量标准化到单元长度,并在生成器的卷积层之后应用。这是为了防止信号在训练过程中失控。

![img](https://miro.medium.com/max/351/1*8GMCKHbBhps3PT_qBlWqsg.png)

The values of each pixel (*x, y)* across *C* channels are normalized to a fixed length. Here, **a** is the input tensor, **b** is the output tensor, and **ε** is a small value to prevent dividing by zero.

```python
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor / torch.sqrt(
            torch.mean(input_tensor ** 2, dim=1, keepdim=True) + 1e-8)
```

### The Discriminator

生成器和判别器大致互为镜像，始终同步增长。鉴别器接收一个输入图像*x*，它要么是生成器的输出，要么是一个按比例缩小到当前训练分辨率的训练图像。作为典型的GAN鉴别器，它试图区分“真实的”训练集图像和“虚假的”生成的图像。它输出D(x)，这个值表示判别器对输入图像来自训练集的置信度。

![img](https://miro.medium.com/max/2690/1*SoWghLQBfcW5i7tAuqQamQ.png)

A detailed view of the discriminator architecture, when it has “grown” to resolution k. Here, x is the input image (either generated or from the training set), α is the extent to which the last generator layer is “faded in”, and D(x) is the probability the generator has assigned to x being from the training set. ==The representation size is halved at each set of layers by an average pooling operation==.

### Minibatch Standard Deviation

![img](https://www.yuthon.com/images/ProGAN_minibatch_std.png)

一般来说,GANs 倾向于生成的样本的多样性，比在训练集中发现的要少。解决这一问题的一种方法是让鉴别器计算批处理中的统计数据, 并使用这些信息来帮助区分“真实”的训练数据Batch和“假的”生成的数据Batch。这鼓励生成器产生更多的多样性, 这样, 在一个生成的数据Batch处理中计算的统计数据与来自训练数据Batch的统计数据更加相似。

在ProGAN中，这是通过在鉴别器的末端插入一个“minibatch standard deviation”层来实现的。这一层没有可训练的参数。它计算整个批处理中feature map像素的标准偏差，并将它们作为额外的通道追加。

ProGAN为了解决GANs生成的图像多样性较差的问题，在discriminator的最后增加了一个minibatch standard deviation层。 这个层没有需要训练的参数，其作用为求取minibatch内的所有feature maps (N×C×H×W)上各个像素位置对应的标准差(C×H×W)，再求其平均值(scalarscalar)，将其展开为一张新的feature map (N×1×H×W)作为新的通道加入。这有助于统计minibatch内的信息，让discriminator根据这些额外的统计信息来区分真是样本的batch与生成样本的batch。从而让generator需要生成更加多样化、更加接近真实样本分布的样本来“骗过”discriminator，最终达到增强generator生成多样化样本的目的。

```python
out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
mean_std = out_std.mean()
mean_std = mean_std.expand(out.size(0), 1, 4, 4)
out = torch.cat([out, mean_std], 1)
```

### Equalized Learning Rate

The authors found that to ensure healthy competition between the generator and discriminator, it is essential that layers learn at a similar speed. ==To achieve this *equalized learning rate*, they scale the weights of a layer according to how many weights that layer has==. They do this using the same formula as is used in He's initialization, except they do it in *every* forward pass during training, rather than just at initialization.
$$
W_f = W_i * \sqrt {2 \over k*k*c}
$$
==Learning rates can be equalized across layers by scaling the weights before every forward pass==. For example, before performing a convolution with *f* filters of size [k, k, c], we would scale the weights of those filters as shown above.

==Due to this intervention （介入）, no fancy tricks are needed for weight initialization — simply initializing weights with a standard normal distribution works fine==.

```python
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input_tensor):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input_tensor):
        return self.conv(input_tensor)
```

### The Loss Function

作者说，损失函数的选择与它们的贡献是正交的，这意味着上述改进都不依赖于特定的损失函数。使用过去几年出现的任何流行的GAN丢失函数都是合理的。

然而，如果你想要完全遵循这篇论文，==他们使用了改进的Wasserstein loss函数，也称为WGAN-GP==。它是一种常见的损失函数，被证明可以稳定训练和提高收敛的几率。

![img](https://miro.medium.com/max/639/1*dQc8M1TU4u7dqmWideSx4g.png)

WGAN-GP损失方程。这里，x '是生成的图像，x是训练集中的图像，D是鉴别器。GP是一种帮助稳定训练的梯度惩罚。梯度惩罚中的 *a* 项指的是一个由0到1之间的随机数组成的张量，它是均匀随机选择的。通常设置 λ= 10。由于我们通常是分批训练的，所以上述损失通常在小批量上取平均。

值得注意的是, WGAN-GP 损失函数预计D(x)和D(x')是无界实值的数字。换句话说, 判别器的输出不是0到1之间的值。这与传统的GAN函数略有不同, 它认为鉴别器的输出是一个概率。

 