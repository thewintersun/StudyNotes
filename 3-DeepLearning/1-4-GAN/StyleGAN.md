## A Style-Based Generator Architecture for Generative Adversarial Networks

论文地址：https://arxiv.org/abs/1812.04948

作者：Tero Karras, Samuli Laine, Timo Aila

机构：NVIDIA

发表：CVPR2019

代码地址：https://github.com/NVlabs/stylegan

文章地址：https://www.yuthon.com/post/tutorials/from-progan-to-stylegan/

文章地址：https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/

文章地址: https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431



### 摘要

We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an ==automatically learned==, ==unsupervised separation of high-level attributes== (e.g., pose and identity when trained on human faces) and ==stochastic（随机的） variation in the generated images== (e.g., freckles（雀斑）, hair), and it enables intuitive, scale-specific control of the synthesis. 

The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably ==better interpolation properties, and also better disentangles the latent factors of variation==. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. 

Finally, we introduce a new, highly varied and high-quality dataset of human faces.



### 介绍

生成模型的生成图片现在越来越清晰，尤其是基于GAN的方法，发展迅速 [30, 45, 5]。

> T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. CoRR, abs/1710.10196, 2017. 作者的ProGAN
>
> https://arxiv.org/abs/1710.10196
>
> T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. CoRR, abs/1802.05957, 2018. 1, 9
>
> https://arxiv.org/abs/1802.05957
>
> A. Brock, J. Donahue, and K. Simonyan. Large scale GAN training for high fidelity natural image synthesis. CoRR, abs/1809.11096, 2018.
>
> https://arxiv.org/abs/1809.11096

但是生成器仍然是一个黑盒子，除了最近的努力[3]， 理解图片合成的各种因素，随机特征的起源，仍然还很缺失。

> D. Bau, J. Zhu, H. Strobelt, B. Zhou, J. B. Tenenbaum,W. T. Freeman, and A. Torralba. GAN dissection: Visualizing and understanding generative adversarial networks. In Proc. ICLR, 2019.
>
> https://arxiv.org/abs/1811.10597

我们的生成器从一个学习到的常量输入开始，在每一个卷基层，通过“latent code”调整图片的风格，因此可以直接控制图像在各个尺度上的特征。结合注入给网络的噪音，这种网络结构的改变导致一种自动的，非监督的，高层属性（比如，动作，身份）与随机变换属性（比如，皱纹，头发）的分离，使得可以进行直观的比例特定的混合和插值操作。

我们没有修改判别器和Loss函数，我们的工作是与GAN的Loss函数，正则化，和参数等是正交的。



### 本文的工作及贡献

①借鉴风格迁移，提出**基于样式的生成器**（style-based generator）。

- 实现了无监督地分离高级属性(人脸姿势、身份)和随机变化(例如雀斑，头发)
- 实现对生成图像中特定尺度的属性的控制。
- 生成器从一个可学习的常量输入开始，隐码在每个卷积层调整图像的“样式”，从而直接控制不同尺度下图像特征的强度。

②实现了**对隐空间（latent space）较好的解耦**。

- 生成器将输入的隐码z嵌入一个中间的隐空间。因为输入的隐空间Z必须服从训练数据的概率密度，这在一定程度上导致了不可避免的纠缠，而嵌入的中间的隐空间W不受这个限制，因此可以被解耦。

③提出了两个**新的量化隐空间解耦程度**的方法

- 感知路径长度 和 线性可分性。与传统的生成器体系结构相比，新的生成器允许更线性、更解耦地表示不同的变化因素。

④提出了新的高质量的**人脸数据集（FFHQ**，7万张1024×1024的人脸图片）



### ProGAN Drawbacks

ProGAN虽然能够生成高质量高分辨率的图像，但是其本质上还是一种无条件（unconditional）的生成方法。其难以控制所生成图像的属性。==并且就算是修改输入的随机向量，其微小的变化也会引起最终生成图像中的多个属性一起变化。==如何将ProGAN改为有条件（conditional）的生成模型，或者增强其微调单个属性的能力，这是一个可以研究的方向。



## StyleGAN

StyleGAN是NVIDIA继ProGAN之后提出的新的生成网络，其主要通过分别修改每一层级的输入，在不影响其他层级的情况下，来控制该层级所表示的视觉特征。这些特征可以是粗的特征（如姿势、脸型等），也可以是一些细节特征（如瞳色、发色等）。

![img](https://www.yuthon.com/images/StyleGAN_visualization.jpg)

具体地说，StyleGAN提出，如果训练得当，ProGAN的每一个层级都有能力控制不同的视觉特征。层级越低，分辨率越低，其能控制的视觉特征也就越粗糙。因此，StyleGAN将视觉特征划分为三类：

1. 粗糙（初级）特征：分辨率小于8×8，主要影响姿态、大致发型、脸型等；
2. 中级特征：分辨率介于16×16至32×32之间，主要影响更加细节的脸部特征、细节发型、嘴的张闭等；
3. 细节（高级）特征：分辨率介于64×64至1024×1024之间，主要影响整体的色调（发色、肤色以及背景色等）与一些细微的特征。

上图，可视化生成器中样式的效果—（使用训练好的生成器）， 在生成图像时用一个隐码（source）产生的样式覆盖另一个隐码（destination）产生的样式的子集。显示在3个不同的尺度下混合两个隐码得到的合成图像。结果表明样式的子集控制了图像的高级属性，不同尺度的样式控制了图像不同的高级属性的生成。

- 在**粗糙分辨率（4-8）**用source的样式覆盖目标的的样式，产生的图像的高级特征（姿势、头发样式、脸的形状和眼镜）从source图像中复制，而目标图像的颜色（眼睛、头发、光线）和更精细的面部特征被保留下来。说明该尺度的样式控制了生成图像的高级特征。
- 在**中间层（16-32）**用source的样式覆盖目标的样式，合成的图像将会从source中继承较小尺度的面部特征（头发的样式、眼睛的闭合），而目标图像的姿势、脸的形状、眼镜等被保留。说明该尺度的样式控制了生成图像的较小尺度的面部特征。
- 在**精细分辨率(642 - 1024)**用source的样式覆盖目标的样式，主要复制了source中的颜色。说明该尺度的样式控制了生成图像的更低尺度的特征-颜色。

![1579511198631](D:\Notes\raw_images\1579511198631.png)



### Workflow

![img](https://www.yuthon.com/images/StyleGAN_overview.jpg)

StyleGAN网络结构

1. 从先验分布Z中采样一个一个512×1的随机向量z∈Z作为latent code，归一化后经过Mapping Network映射到另外一个中间的 latent space上，得到中间的 latent code表示w∈W。
2. 将上一步得到的w通过==可学习的仿射变换A==输入到Synthesis Network各个层级的AdaIN层中，用以控制style；同时将噪声==通过学习到的缩放参数B==加到AdaIN层之前。
3. 将固定的向量输入Synthesis Network，输出得到生成的图像。

```python
class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, style_dim=512, initial=False):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, 
                                     padding=padding)

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out    
    
class Generator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.progression = nn.ModuleList([StyledConvBlock(512, 512, 3, 1, initial=True),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 256, 3, 1),
                                          StyledConvBlock(256, 128, 3, 1)])
        
        self.to_rgb = nn.ModuleList([EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(256, 3, 1),
                                     EqualConv2d(128, 3, 1)])
        # self.blur = Blur()
    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        pass
```

### Details

#### Mapping Network

==Mapping Network的作用是将输入向量编码为一个中间表示，使得该中间表示的每一个元素都能够控制不同的视觉特征==。

如果像传统的cGANs及其衍生版本那样，只靠输入向量自身控制视觉特征，这种能力是有限的，因为其还要受到训练数据的概率密度分布的影响。训练数据中如果某一类出现得多一些，那么输入向量中的值就更可能被映射到这一类上面。这就导致了模型所控制的特征是耦合的（coupled）或者说是纠缠的（entangled），模型并不能单独控制输入向量的某一部分的映射。==但是通过Mapping Network将输入向量映射为另外的中间表示，则不用服从训练数据集的分布，并且能够在一定程度上减少特征之间的相关性==。

Mapping Network由8层FC层组成，输入为随机向量z∈Z，输出为中间表示w∈W，两者维度均为512×1。

[![img](https://www.yuthon.com/images/StyleGAN_mapping_network.png)](https://www.yuthon.com/images/StyleGAN_mapping_network.png)StyleGAN网络中的Mapping Network (Source: Rani Horev's blog post) 

```python
class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, noise=None, step=0, alpha=-1, mean_style=None, 
                style_weight=0, mixing_range=(-1, -1)):
		pass
```

#### Adaptive Instance Normalization (AdaIN)

Mapping Network编码得到的中间表示w∈W，需要通过AdaIN (Adaptive Instance Normalization)来输入生成网络。AdaIN层在Systhesis Network的每个分辨率层级中都存在，并且用以控制该分辨率层级的视觉特征。
$$
\text{AdaIN}(x_i,y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
$$

1. 对卷积层的输出进行Instance Normalization，==也就是将输出的每个通道都进行归一化==，得到$\frac{x_i - \mu(x_i)}{\sigma(x_i)}$
2. 对输入的中间表示w（维度512）通过一个FC层A转换为针对n个通道的 scale ($y_{s,i}$) 与 bias ($y_{b,i}$)，维度为2n。
3. 通过第2步得到的scale与bias，对于第1步得到的归一化输出的每个通道都进行shift。==这种操作相当于对卷积层的每个滤波器的输出进行加权，而这个权重是可学习的==。通过训练，使得w所代表的权重能够被转化为视觉表示。

[![img](https://www.yuthon.com/images/StyleGAN_adain.png)](https://www.yuthon.com/images/StyleGAN_adain.png)StyleGAN网络中的AdaIN模块 (Source: Rani Horev's blog post) 

```python
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        # 512 -> Nx2
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        # split sytle inyo scale and bias
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
```

#### Removing traditional input

包含之前的ProGAN在内，==传统的GANs都需要用一个随机向量喂给生成网络来生成图像，这个随机向量就决定了生成图像的视觉特征==。而在StyleGAN中，既然生成图像的视觉特征已经交由w与AdaIn来控制，那么再在Synthesis Network的最开始输入一个随机向量就显得有点多余了。==因此这个随机向量输入被替换成了一个定值向量输入，而且这在结果上有益于生成图像的质量。一个可能的解释是这种固定的输入使得网络只需要考虑w那边传过来的视觉属性，而不用再管另外一个输入的变量，从而在一定程度上减少特征之间的纠缠==。

![img](https://www.yuthon.com/images/StyleGAN_constant_input.png)

StyleGAN网络在Synthesis Network上使用了固定的输入：

```python
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
		
        # generate fixed random vector
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        # generate fake batch
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out
```

#### Stochastic variation

==为了增强生成样本的多样性，同时考虑到人脸上还是有许多地方可以看成是随机的（例如雀斑、皱纹、头发纹理等等），通常GANs会在输入向量上增加一层随机噪声来实现这种微小的特征==。StyleGAN也一样，如果只使用w来控制视觉特征，输入Synthesis Network的向量又是固定的，那么一旦w固定，则生成的图像也是一成不变的。

不过，如上文所述，传统方法直接将随机噪声加在输入变量上，这样容易导致特征的纠缠现象，使得其他的特征也受到影响。同样地，与上面的解决方法一致，==StyleGAN将噪声通过FC层B重新编码，然后加在AdaIN之前一层输出的每个通道上，用以轻微改变每一层级所负责的视觉特征==。

![img](https://www.yuthon.com/images/StyleGAN_noise.png)

StyleGAN网络在AdaIN层之前增加了编码后的噪声 (Source: Rani Horev's blog post)

```python
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise
```

![1579513678328](D:\Notes\raw_images\1579513678328.png)

Figure 4. Examples of stochastic variation. (a)生成两幅图像。(b)利用不同的输入噪音的实现进行放大。虽然整体外观几乎是相同的，个别头发放置非常不同。(c)==每个像素超过100种不同实现的标准偏差，突出显示图像的哪些部分受到噪音的影响。主要的区域是头发、轮廓和部分背景，但是在眼睛的反射中也有有趣的随机变化==。全局方面，如身份和姿态不受随机变化的影响。

![1579513921182](D:\Notes\raw_images\1579513921182.png)

Figure 5. 将随机变化应用于不同的层的效果（将噪声输入不同层）。a. 噪声加入了所有层；b. 没有噪声；c. 在精细层（fine layers，64-1024）加入噪声；d. 在粗糙层（coarse layer，4-32）加入噪声。==粗糙的噪声导致头发大规模卷曲，背景特征更大，而细小的噪声则使头发卷曲更细，背景细节更细，皮肤毛孔更细。==

![1579511249260](D:\Notes\raw_images\1579511249260.png)

#### mixing regularization

为了进一步鼓励styles的局部化（减小不同层之间样式的相关性），本文对生成器使用混合正则化。

方法：==对给定比例的训练样本（随机选取）使用样式混合的方式生成图像==。==在训练过程中，使用两个随机隐码z（latent code）而不是一个，生成图像时，在合成网络中随机选择一个点（某层），从一个隐码切换到另一个隐码(称之为样式混合)==。具体来说，通过映射网络运行两个潜码z1、z2，并让对应的w1、w2控制样式，使w1在交点前应用，w2在交点后应用。

这种正则化技术==防止网络假设相邻样式是相关的==，随机切换确保网络不会学习和依赖于层级之间的相关性。

这种方法虽然它并没有改善所有数据集上的模型性能，但这个概念有一个非常有趣的副作用——它能够以一致的方式组合多个图像(如下面的视频所示)。该模型生成两幅图像A和B，然后通过提取A的低层次特征和B的其余特征进行组合。

视频展示：https://youtu.be/kSLJriaOumA 

![1579511123385](D:\Notes\raw_images\1579511123385.png)

![1579512099354](D:\Notes\raw_images\1579512099354.png)

Table 2. FIDs in FFHQ for networks trained by enabling the mixing regularization for different percentage of training examples. Here we stress test the trained networks by randomizing 1 - 4 latents and the crossover points between them. Mixing regularization improves the tolerance to these adverse operations  significantly. Labels E and F refer to the configurations in Table 1.

- 用FFHQ数据集训练网络，对不同比例的训练样本使用混合正则化，得到FID值。
- 0-100%表示训练过程中混合正则化应用的样本的比例。
- 在训练过程中，通过随机化1-4个latent code和它们之间的交点对网络进行压力测试。
- ==使用相同数目的隐码latent code，随着混合正则化应用的样本的比例的增加（同一列从上到下），得到的图像的质量得到了提高，FID值降低。==

#### Truncation trick in W

下图显示了使用生成器F从FFHQ数据集中生成的图片，平均的质量很高，帽子、眼睛等配饰也成功合成了。图中生成人脸时，对中间隐空间W采取了截断 trick（0.7），该trick只在网络的低分辨率（4-32）使用（该trick只在这里使用，在计算FID时没有使用过），生成的图片的分辨率为1024。

截断技巧（截断中间隐空间W）防止生成极值：

- 考虑到训练数据的分布，低密度区域很少被表示，因此生成器很难学习该区域。通过对隐空间进行截断可以提升生成影像的平均质量，虽然会损失一些变化。对中间隐码W进行截断，迫使W接近平均值。

- 具体方法：在训练之后，通过多个输入 z 经过映射网络生成的中间向量W，得到均值$\overline W $ ， 

  $\overline W = \mathbf E_{z \sim P(z)}[f(z)]$

- 在生成新图像时，将输入z的映射w与中心的偏差按比例缩放: 

  $w' = \overline w + \psi ( w - \overline w) , where \ \ \psi < 1$

  ![1579513355645](D:\Notes\raw_images\1579513355645.png)

Figure 2. Uncurated set of images produced by our style-based generator (config F) with the FFHQ dataset. Here we used a variation of the truncation trick [42, 5, 34] with $\psi = 0.7$ for resolutions $4^2 - 32^2$. Please see the accompanying video for more results.

#### Fine-tuning

在ProGAN上额外改进的StyleGAN更新几个网络超参数, 例如训练持续时间和损失函数, ==并将最邻近的up / down缩放替换为双线性插值==。虽然这一步对于模型性能很重要,但它不那么创新,因此不会详细描述(附录C)。

![img](https://miro.medium.com/max/1024/0*ANwSHXJDmwqjNSxi.png)

​															An overview of StyleGAN



### 两种新的量化隐空间耦合度的方法

解耦的目标是使隐空间（latent space）由线性子空间组成，即每个子空间（每个维度）控制一个变异因子（特征）。但是隐空间Z中的各个因子组合的采样概率需要与训练数据中响应的密度匹配，就会产生纠缠。而中间隐藏空间W不需要根据任何固定分布进行采样; 它的采样密度是由可学习的映射网络f(z)得到的，使变化的因素变得更加线性。

本文假设，==生成器基于解耦的表示比基于纠缠的表示应该更容易产生真实的图像==（若在FID变小的同时，隐空间耦合度变小，则可以得证）。因此，我们期望训练在无监督的情况下（即，当不预先知道变异的因素时）产生较少纠缠的W。

最近提出的用于量化解耦的指标需要一个将输入图像映射到隐码的编码器网络。但不适合本文，因为baseline GAN缺乏这样的编码器。所以本文提出了两种新的量化解耦的方法，它们都不需要编码器，所以对于任何数据集和生成器都是可计算的。 

**1. 感知路径长度（Perceptual path length）**

对隐空间向量进行插值会在图像中产生非线性变化。比如，在所有端点中缺失的特征可能会出现在线性插值路径的中间。这表明隐空间是耦合的，变化因子没有被恰当的分开。==所以通过测量当在两个隐空间之间进行插值时图像的剧烈变化程度，可以反映隐空间的纠缠程度（特征分离程度）==。

感知路径长度计算，使用10000个样本计算： 

将两个隐空间之间的插值路径细分为小段，感知总长度定义为每段感知差异的总和。感知路径长度的定义是这个和在无限细的细分下的极限，实际上用一个小的细分$ \epsilon = 10^{-4}$ 来近似它。隐空间Z中所有可能端点（在路径中的位置）的平均感知路径长度，计算如下:
$$
l_Z = \mathbb E[{\frac 1 {\epsilon^2}}\ d(\ G(slerp(z_1,z_2;t)),\ G(slerp(z_1,z_2;t+\epsilon))\ )]
$$
其中$z_1, z_2 \sim P(z) ,\  t \sim U(0,1)$ 。 ==t 服从0,1均匀分布==，==slerp表示球面插值操作==，这是在归一化的输入隐空间中最合适的插值方式。G是生成器，d计算得到生成图像之间的感知距离。因为d是二次的，所以除以 $\epsilon^2$而不是  $\epsilon$ 来消除对细分粒度的依赖。==d的具体计算方式：使用基于感知的成对图像距离，测量连续图像之间的差异==,  (两个VGG16 embeddings之间的差异，利用VGG16提取出图像的特征，在特征层面上计算距离) 

计算隐空间W的感知路径长度与z的唯一不同是采用lerp线性插值，因为w向量没有进行归一化
$$
l_W = \mathbb E[{\frac 1 {\epsilon^2}}\ d(\ g(lerp(f(z_1),f(z_2);t)),\ g(lerp(f(z_1),f(z_2);t+\epsilon))\ )]
$$


**2. 线性可分性（linear separability）**

In order to make the discussion regarding feature separation more quantitative, the paper presents two novel ways to measure feature disentanglement:

1. Perceptual path length — measure the difference between consecutive images (their VGG16 embeddings) when interpolating between two random inputs. Drastic changes mean that multiple features have changed together and that they might be entangled.
2. Linear separability — the ability to classify inputs into binary classes, such as male and female. The better the classification the more separable the features.

By comparing these metrics for the input vector z and the intermediate vector ⱳ, the authors show that features in ⱳ are significantly more separable. These metrics also show the benefit of selecting 8 layers in the Mapping Network in comparison to 1 or 2 layers.

如果一个隐空间是充分解耦的，应该能够找到与每个变化因子对应的方向向量。我们提出了另一种度量方法来量化这种效果，测量通过线性超平面将隐空间点分割成两个不同的集合的程度，使每个集合对应于图像的特定的二元属性（比如男、女）。计算方法：

- ①训练40个辅助分类器，分别对40个二元属性进行区分（每个分类器区分一个属性，比如区分女性男性）。分类器与StyleGAN判别器结构相同，使用CelebA-HQ数据集训练得到（保留原始CelebA的40个属性，150000个训练样本),学习率10-3，批次大小8，Adam优化器。
- ②使用生成器生成200,000个图像，并使用辅助分类器进行分类，根据分类器的置信度对样本进行排序，去掉置信度最低的一半，得到100,000个已知类别的隐空间向量（latent code）
- ③对于每个属性，拟合一个线性SVM来预测标签-基于传统的隐空间点或基于样式的隐空间点w-并且根据这个超平面对这些隐空间点（512维,10000个点）进行分类。
- ④用条件熵H(Y |X)度量超平面将点划分为正确类别的能力，X是SVM预测的类别，Y是预先训练好的辅助分类器确定的类（作为真实类别）；因此，根据SVM确定样本在超平面的哪一边，条件熵告诉我们需要多少额外的信息来确定样本的真实类别。直觉上，如果隐空间中的变化因子是耦合的（非线性的），那么用超平面来分离样本点将会更加困难（需要更多的额外信息），产生高的条件熵。较低的值表示易于分离（可分性好），因此解耦程度更大

可分性计算公式 $exp(\sum_i H(Y_i|X_i)) $，其中i列举了40个属性。取幂是为了将值从对数域变换为线性域，便于进行比较。

 **对隐空间的解耦效果（特征分离）**

![1579514718118](D:\Notes\raw_images\1579514718118.png)

Table 3. Perceptual path lengths and separability scores for various generator architectures in FFHQ (lower is better). We perform the measurements in Z for the traditional network, and in W for stylebased ones. Making the network resistant to style mixing appears to distort the intermediate latent spaceW somewhat. We hypothesize that mixing makes it more difficult forW to efficiently encode factors of variation that span multiple scales.

![1579514759314](D:\Notes\raw_images\1579514759314.png)

Table 4. The effect of a mapping network in FFHQ. The number in method name indicates the depth of the mapping network. We see that FID, separability, and path length all benefit from having a mapping network, and this holds for both style-based and traditional generator architectures. Furthermore, a deeper mapping
network generally performs better than a shallow one.

- 表3说明对于噪声输入的基于样式的生成器E/F（使用FFHQ数据集训练好的），**感知路径长度比传统生成器短，这表明中间隐空间W比隐空间Z更线性**，即W是Z的解耦映射（一定程度上）
- 表3和表4显示，**中间隐空间W始终比输入隐空间Z更易于分离（Separability更小）**，这表明纠缠的表示更少。
- 此外，表4显示**增加映射网络的深度可以同时提高W生成图像的质量（FID）和可分性**，这与综合网络喜欢解耦输入表示的假设(可分性提高，生成图像质量提高)是一致的。



### 实验和结果

**评估指标：FID(Fréchet Inception Distance，越小越好）**

测量不同的生成器结构产生的图像的质量。==FID计算真实样本和生成样本在特征空间之间的距离==。预训练的 Inception V3 来提取全连接层之前的 2048 维向量，作为图片的特征，然后根据均值和协方差来进行距离计算。具体公式如下：
$$
FID = || \mu_r - \mu_g||^2 + Tr(\sum_r + \sum_g -2(\sum_r\sum_g)^{1/2})
$$
公式中： $\mu_r$ 真实图片的特征的均值,  $\mu_g$ 生成的图片的特征的均值, $\sum_r$ 真实图片的特征的协方差矩阵, 生成 $\sum_g$ 图片的特征的协方差矩阵。

FID 只把 Inception V3 作为特征提取器，并不依赖它判断图片的具体类别，因此不必担心 Inception V3 的训练数据和生成模型的训练数据不同。FID的tensorflow实现 https://github.com/bioinf-jku/TTUR

**FID的测量结果**

![1579512243165](D:\Notes\raw_images\1579512243165.png)

Table 1. Fr´echet inception distance (FID) for various generator designs (lower is better). In this paper we calculate the FIDs using 50,000 images drawn randomly from the training set, and report the lowest distance encountered over the course of training.

训练数据集：CelelbA-HQ 和 FFHQ

- FFHQ数据集（本文提出的）Flickr-Faces-HQ(FFHQ)，包含7万张高分辨率的人脸图像（2014×1024）（https://github.com/NVlabs/ffhq-dataset）
- 加入映射网络、样式模块、移除传统输入、增加噪声，都提高了生成图像的质量，降低了FID.
- 基于样式的生成器F相比于B，FID提升了20%，生成的图像质量更好。



### 训练细节和超参数

- StyleGAN使用8个Tesla V100 gpu对CelebA-HQ和FFHQ数据集进行为期一周的训练。在TensorFlow中实现。

- B-F 在两个数据集使用不同的损失函数： 

- - CELEBA-HQ数据集使用WGAN-GP；
  - FFHQ数据集，非线性饱和损失，WGAN-GP+R1正则化（γ=10），使用该损失的FID值比WGAN-GP减小的时间更长，所以使用更长的训练时间。 

- 基于样式的生成器使用leak ReLU，α=0.2，所有卷积层使用相同学习率；

- 特征图（卷积核）数量与ProGAN相同；

- 映射网络的学习率每两层降低一次λ = 0.01λ（映射网络深度变大时，训练不稳定）。

- 权重使用高斯（0，1）随机初始化；除了样式计算ys的偏置初始化1，其余偏置和噪声的尺度因子初始化为0.

- 没有使用 batch normalization、dropout。 



### 结论

StyleGAN是一篇开创性的论文,它不仅能产生高质量和现实的图像,而且还能对生成的图像进行优越的控制和理解,使其比以前更容易产生虚假的图像。在StyleGAN, 特别是映射网络和自适应标准化(AdaIN)中提出的技术,很可能是在GANs 的许多未来创新的基础。

- 基于样式的生成器，能生成更高质量的高分辨率图像。
- 实现了无监督地分离高级属性(人脸姿势、身份)和随机变化(例如雀斑，头发)，实现对生成图像中特定尺度的属性的控制。通过style控制高级属性，通过不同层的style实现对不同尺度的特征的控制；通过噪声控制随机变化。 
- 降低了隐空间的耦合性。通过映射网络生成中间隐空间（intermediate latent space），中间隐空间降低了输入隐空间的纠缠。 
- 提出两种新的量化隐空间耦合性的指标-感知路径长度和线性可分性；
- 对高级属性与随机效应的分离以及中间隐空间的线性的研究，提高了对GAN合成的理解和可控性。 



 