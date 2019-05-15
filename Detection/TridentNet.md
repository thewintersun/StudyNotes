Scale-Aware Trident Networks for Object Detection
- 论文地址:Scale-Aware Trident Networks for Object Detection
- 作者：Yanghao Li, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang
- 机构：TuSimple
- 作者知乎：知乎用户
- 代码地址：TuSimple/simpledet
- 作者知乎解析：Naiyan Wang：TridentNet：处理目标检测中尺度变化新思路
- 代码论文： Naiyan Wang：SimpleDet: 一套简单通用的目标检测与物体识别框架

https://arxiv.org/abs/1903.05831
SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition

With the help of dilated convolutions [43], different branches of trident blocks have the same network structure and share the same parameter weights yet have different receptive fields.

Furthermore, to avoid training objects with extreme scales, we leverage a scale-aware training scheme to make each branch specific to a given scale range matching its receptive field.

Finally, thanks to weight sharing through the whole multibranch network, we could approximate the full TridentNet with only one major branch during inference.

There are several design factors of the backbone network that may affect the performance of object detectors including downsample rate, network depth and receptive field.

