## SSH: Single Stage Headless Face Detector

论文地址：https://arxiv.org/abs/1708.03979

作者: Mahyar Najibi, Pouya Samangouei, Rama Chellappa, Larry Davis

发表: ICCV2017

机构：University of Maryland

官方代码地址： https://github.com/mahyarnajibi/SSH



#### 摘要

介绍了Single Stage Headless (SSH) 人脸检测技术。与两个阶段的 proposal-classification检测器不同，SSH直接从主干网络的早期卷积层做单阶段的人脸检测。==SSH是headless的==。也就是说，它能够在去除分类网络“头”的同时，获得最先进的结果——即VGG-16中包含大量参数的所有全连接层。另外，==SSH在设计上是尺度不变的==，而不是依靠图像金字塔来检测各种尺度的人脸。我们==在网络的单次前向通道中同时检测不同尺度的人脸，但是来自不同的层==。这些属性使SSH快速且轻量级。令人惊讶的是，使用无头的VGG-16, SSH在更大的数据集上击败了基于resnet 101的最新技术。尽管如此，SSH不使用FPN速度快了5倍。此外，如果部署了图像金字塔，我们的轻量级网络在更广泛的数据集的所有子集上都达到了最先进的水平，将AP提高了2.5%。SSH还可以在FDDB和pasca - faces数据集上获得最先进的结果，即使使用较小的输入大小，使得在GPU上的运行时间达到 50 ms/image。

总结：Proposal Free， 无全连接，无特征金字塔



#### 介绍

SSH是一阶段的检测器，和两阶段的检测器相比，相似的是都采用Anchor机制来回归BBOX，不同的是分类和回归Anchor操作是同时的。

```python
rpn_cls_score = conv_act_layer(rpn_relu, 'rpn_cls_score_stride%d' % stride, 2 * num_anchors, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='')
rpn_bbox_pred = conv_act_layer(rpn_relu, 'rpn_bbox_pred_stride%d' % stride, 4 * num_anchors, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='')
```

SSH 相对于其他的检测器，是没有头部的。也就是说没有FC层，这去除了大量的参数与计算, 使得SSH轻量化。也就是说特征层，直接接估计层。

SSH设计上将是Scale-invariant的，类似于SSD或FPN，在不同Stride特征处做Predict。

```python
m1 = ssh_detection_module(m1_conv, F2, 'ssh_m1_det')
m2 = ssh_detection_module(relu5_3, F1, 'ssh_m2_det')
m3 = ssh_detection_module(m3_pool, F1, 'ssh_m3_det')
return {8: m1, 16: m2, 32: m3}
```

 ![1574415903216](C:\Users\j00496872\Desktop\Notes\raw_images\1574415903216.png)

![1574664585515](C:\Users\j00496872\Desktop\Notes\raw_images\1574664585515.png)

```python
def ssh_context_module(body, num_filters, name):
    conv_dimred = conv_act_layer(body, name + '_conv1', num_filters, kernel=(3, 3), 
                                 pad=(1, 1), stride=(1, 1), act_type='relu', dcn=False)
    conv5x5 = conv_act_layer(conv_dimred, name + '_conv2', num_filters, kernel=(3, 3), 
                             pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
    conv7x7_1 = conv_act_layer(conv_dimred, name + '_conv3_1', num_filters, 
                               kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
                               act_type='relu', dcn=False)
    conv7x7 = conv_act_layer(conv7x7_1, name + '_conv3_2', num_filters, kernel=(3, 3), 
                             pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
    return (conv5x5, conv7x7)


def ssh_detection_module(body, num_filters, name):
    conv3x3 = conv_act_layer(body, name + '_conv1', num_filters, kernel=(3, 3), 
                             pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
    conv5x5, conv7x7 = ssh_context_module(body, num_filters // 2, name + '_context')
    ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name=name + '_concat')
    ret = mx.symbol.Activation(data=ret, act_type='relu', name=name + '_concat_relu')
    return ret
```

具体网络结构：





More formally, if the feature map connected to the detection module Mi has a size of Wi  Hi, there would
be Wi  Hi  Ki anchors with aspect ratio one and scales fS1i ; S2i ; : : : SKi i g.



#### 实验结果

**在WiderFace上的测试结果:**

![1574666994201](C:\Users\j00496872\Desktop\Notes\raw_images\1574666994201.png)

**Ablation Study:**

![1574666833037](C:\Users\j00496872\Desktop\Notes\raw_images\1574666833037.png) 

![1574666854073](C:\Users\j00496872\Desktop\Notes\raw_images\1574666854073.png)

![1574666886344](C:\Users\j00496872\Desktop\Notes\raw_images\1574666886344.png) 

Timing are performed on a NVIDIA Quadro P6000 GPU.

![1574666817582](C:\Users\j00496872\Desktop\Notes\raw_images\1574666817582.png)