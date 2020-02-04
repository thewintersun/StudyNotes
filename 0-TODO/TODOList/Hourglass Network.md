## Stacked Hourglass Networks for Human Pose Estimation

作者：Alejandro Newell, Kaiyu Yang, Jia Deng

地址：https://arxiv.org/abs/1603.06937

机构：University of Michigan, Ann Arbor

文章地址：https://zhuanlan.zhihu.com/p/45002720



### 摘要

This work introduces a novel convolutional network architecture for the task of human pose estimation. ==Features are processed across all scales and consolidated （合并，巩固，统一）to best capture the various spatial relationships associated with the body.==  We show how repeated bottom-up, top-down processing used in conjunction with intermediate supervision is critical to improving the performance of the network. We refer to the architecture as a "stacked hourglass (沙漏, 水漏)" network based on the successive steps of pooling and upsampling that are done to produce a final set of predictions. State-of-the-art results are achieved on the FLIC and MPII benchmarks outcompeting all recent methods.

Stacked Hourglass Networks for Human Pose Estimation 这篇论文介绍了一种新的网络结构用于人体姿态检测，作者在论文中展现了不断重复bottom-up、top-down过程以及运用intermediate supervison（中间监督）对于网络性能的提升，下面来介绍Stacked Hourglass Networks.

## **简介**

理解人类的姿态对于一些高级的任务比如行为识别来说特别重要，而且也是一些人机交互任务的基础。作者提出了一种新的网络结构Stacked Hourglass Networks来对人体的姿态进行识别，==这个网络结构能够捕获并整合图像所有尺度的信息==。之所以称这种网络为Stacked Hourglass Networks，主要是它长得很像堆叠起来的沙漏，如下图所示：

![img](https://pic1.zhimg.com/80/v2-130933a4b11803fc842897de5a4db698_hd.jpg)

==这种堆叠在一起的Hourglass模块结构是对称的，bottom-up过程将图片从高分辨率降到低分辨率，top-down过程将图片从低分辨率升到高分辨率==，这种网络结构包含了许多pooling和upsampling的步骤，pooling可以将图片降到一个很低的分辨率，upsampling可以结合多个分辨率的特征。

## **Hourglass Module（Hourglass 模块）**

Hourglass模块设计的初衷就是为了捕捉每个尺度下的信息，因为捕捉像脸，手这些部分的时候需要局部的特征，而最后对人体姿态进行预测的时候又需要整体的信息。==为了捕获图片在多个尺度下的特征，通常的做法是使用多个pipeline分别单独处理不同尺度下的信息，然后再网络的后面部分再组合这些特征==，而作者使用的方法就是用带有skip layers的单个pipeline来保存每个尺度下的空间信息。

![img](https://pic1.zhimg.com/80/v2-15106aedb40c0cfcd688a8386e31d320_hd.jpg)

在Hourglass模块中，卷积和max pooling被用来将特征降到一个很低的分辨率，在每一个max pooling步骤中，网络产生分支并在原来提前池化的分辨率下使用更多的卷积，当到达最低的分辨率的时候，网络开始upsample并结合不同尺度下的特征。这里upsample（上采样）采用的方法是[最邻近插值](https://link.zhihu.com/?target=https%3A//blog.csdn.net/ZYTTAE/article/details/42710303)，之后再将两个特征集按元素位置相加。

当到达输出分辨率的时候，再接两个1×1的卷积层来进行最后的预测，网络的输出是一组heatmap，对于给定的heatmap，网络预测在每个像素处存在关节的概率。

## **网络结构**

**Residual Module**

**Fig.3**中的每个方框都由下面这样的残差块组成：

![img](https://pic1.zhimg.com/80/v2-78ccb5db4ad9ad112714d839d615178c_hd.jpg)Residual Module

上图的残差块是论文中的原图，描述的不够详细，自己看了下源代码之后，画出了如下图所示的Residual Module：

![img](https://pic2.zhimg.com/80/v2-d463b532928d3dc6d74e793df0dbd169_hd.jpg)

贴出一段作者提供的关于Residual Module的源代码：

```lua
local conv = nnlib.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end
```

**Hourglass Module**

Hourglass Module由上面的Residual Module组成，由于它是一个递归的结构，所以可以定义一个阶数来表示递归的层数，首先来看一下一阶的Hourglass Module：

![img](https://pic2.zhimg.com/80/v2-8085e99f715a469e227ba07100f46309_hd.jpg)

上图中的Max pool代表下采样，Res代表上面介绍的Residual Module，Up Sample代表上采样。多阶的Hourglass Module就是将上图虚线框中的块递归地替换为一阶Hourglass Module，由于作者在实验中使用的是4阶的Hourglass Moudle，所以我们画出了4阶的Hourglass Module的示意图：

![img](https://pic2.zhimg.com/80/v2-45f12e93b90128e4794aa77061d1b5a5_hd.jpg)



**整体结构**

网络输入的图片分辨率为256×256，在hourglass模块中的最大分辨率为64×64，整个网络最开始要经过一个7×7的步长为2的卷积层，之后再经过一个残差块和Max pooling层使得分辨率从256降到64。下面贴出作者提供的整个网络结构的源代码：

```lua
paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()
    local inp = nn.Identity()()
    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)
        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)
        -- Predicted heatmaps
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,
            									ref.nOutChannels,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)
        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution( 
                				  ref.nOutChannels,opt.nFeats,1,1,1,1,0,0 )(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end
    -- Final model
    local model = nn.gModule({inp}, out)
    return model
end
```

诶...Lua代码看起来是真费劲啊!还是画个图吧：

![img](https://pic3.zhimg.com/80/v2-202a2358835054794c6356b77ddb7a82_hd.jpg)

图中的4阶Hourglass Module就是前面讲的4阶Hourglass Module，可以看到整个网络还是挺庞大的，图中的渐变红色块就是加入了中间监督的地方，即在此处使用loss函数，下面讲一下中间监督。

> 注意，上面的整体网络结构图中中间监督的地方输出的通道数为16是针对于MPII Human Pose这个数据集，因为该数据集将人体划分为16个关节点.

## **Intermediate Supervision（中间监督）**

作者在整个网络结构中堆叠了许多hourglass模块，从而使得网络能够不断重复自底向上和自顶向下的过程，作者提到采用这种结构的关键是要使用中间监督来对每一个hourglass模块进行预测，即对中间的heatmaps计算损失。

关于中间监督的位置，作者在文中也进行了讨论。大多数高阶特征仅在较低的分辨率下出现，除非在上采样最后。如果在网络进行上采样后进行监督，则无法在更大的全局上下文中重新评估这些特征；如果我们希望网络能够进行最佳的预测，那么这些预测就不应该在一个局部范围内进行。

由于hourglass模块整合了局部和全局的信息，若想要网络在早期进行预测，则需要它对图片有一个高层次的理解即使只是整个网络的一部分。最终，作者将中间监督设计在如下图所示位置：

![img](https://pic1.zhimg.com/80/v2-40d110117fc385cc7d0c4a964a7580bc_hd.jpg)

The network splits and produces a set of heatmaps (outlined in blue) where a loss can be applied. A 1x1 convolution remaps the heatmaps to match the number of channels of the intermediate features. These are added together along with the features from the preceding hourglass.

在整个网络中，作者共使用了8个hourglass模块，需要注意的是，这些hourglass模块的权重不是共享的，并且所有的模块都基于相同的ground truth添加了损失函数。下面介绍训练过程的细节。

关于中间监督loss的计算，论文中是这么说的：

> Predictions are generated after passing through each hourglass where the network has had an opportunity to process features at both local and global contexts. Subsequent hourglass modules allow these high level features to be processed again to further evaluate and reassess higher order spatial relationships.

所以，每个Hourglass Module的loss是单独计算的，这样使得后面的Hourglass Module能够更好地再评估。

## **训练过程细节**

作者在FLIC和MPII Human Pose数据集上进行了训练与评估。这篇论文只能用于单人姿态检测，但是在一张图片中经常有多个人，解决办法就是只对图片正中心的人物进行训练。将目标人物裁剪到正中心后再将输入图片resize到256×256。为了进行数据增量，作者将图片进行了旋转（+/-30度）、scaling（.75-1.25）。

网络使用RMSprop进行优化，学习率为2.5e-4. 测试的时候使用原图及其翻转的版本进行预测，结果取平均值。网络对于关节点的预测是heatmap的最大激活值。损失函数使用均方误差（Mean Squared Error,MSE）来比较预测的heatmap与ground truth的heatmap（在节点中心周围使用2D高斯分布，标准差为1） ![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%28%5Chat%7Bx%7D_i-x_i%29%5E2%7D%5C%5C)

为了提高高精度阈值的性能，在转换回图像的原始坐标空间之前，预测在其下一个最高邻居的方向上偏移四分之一像素。

![img](https://pic1.zhimg.com/80/v2-41fd9d7f5d64308856c054f156f2914c_hd.jpg)

## **评测结果**

评测指标采用的是标准的PCK指标（Percentage of Correct Keypoints），这个指标指出了检测结果关键点落在ground truth的标准化距离之内的比例。对于FLIC数据集来说，距离按躯干大小标准化，对于MPII数据集来说，距离按头的大小标准化。

**FLIC上的评测结果**

![img](https://pic3.zhimg.com/80/v2-2e66a304e6ea4af586f8eccf7c1d5aae_hd.jpg)

**MPII上的评测结果**

![img](https://pic2.zhimg.com/80/v2-7d6f00a84e6b9e16eaec57ee71e02f81_hd.jpg)

关于中间监督的位置，作者也进行了对比实验，结果如下图所示：

![img](https://pic3.zhimg.com/80/v2-ace912cbd7e96c3e84bbab46c8873a3a_hd.jpg)

可以看到结果最好的是HG-Int，即在最终输出分辨率之前的两个最高分辨率上进行上采样后应用中间监督。

关于hourglass模块使用的个数，作者也进行了对比实验，分别采用2、4、8个堆叠的hourglass模块进行对比实验，结果如下所示：

![img](https://pic4.zhimg.com/80/v2-2a31271aee27080b8f13f6f2b65abecb_hd.jpg)