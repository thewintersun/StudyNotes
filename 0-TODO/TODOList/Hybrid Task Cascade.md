### Hybrid Task Cascade for Instance Segmentation

论文讲解：https://mp.weixin.qq.com/s/n-7XBvWRuRIZoCXcvJ21oA

论文地址： https://arxiv.org/abs/1901.07518 

代码地址： https://github.com/open-mmlab/mmdetection





商研丨实例分割的进阶三级跳：从Mask R-CNN到Hybrid Task Cascade

商汤科技SenseTime *3月21日*

![img](https://mmbiz.qpic.cn/mmbiz_jpg/JQz7S69vP7FqtpVhnOYYAAalmXYt5km2juxKxBUIyu5UbTnqnGiacXjGdXZkCYSgia9GRsw40PbyiaDibSbnzeW0mw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**编者按**：早在2018年的COCO比赛中，商汤科技研究员和香港中文大学-商汤联合实验室（MMLab）同学组成的团队，在最核心的物体检测（Detection）项目中夺得冠军。商汤团队创造性地开发了全新的技术，尤其是提出了新的多任务混合级联架构（Hybrid Task Cascade），通过把不同子任务进行逐级混合，有效地改善了整个检测过程中的信息流动。



近日，基于 COCO 2018比赛团队合作的成果，MMLab团队又取得新突破，发表了论文《Hybrid Task Cascade for Instance Segmentation》，该论文已入选CVPR 2019。论文提出了一种新的实例分割框架，设计了多任务多阶段的混合级联结构，并且融合了一个语义分割的分支来增强Spatial Context。这种框架取得了明显优于Mask R-CNN和Cascade MaskR-CNN的结果。本文是对《Hybrid Task Cascade for Instance Segmentation》的论文解读。





##   背景

实例分割（Instance Segmentation）是一个和物体检测非常相关但是更难的问题，在物体检测的基础上，还要求分割出物体的像素。

实例分割这个问题近几年的发展在很大程度上是由COCO数据集和比赛推动的。从MNC，FCIS到PANet，都是在COCO Instance Segmentation Track 上拿第一名的方法。Mask R-CNN是个例外，因为Paper公开得比较早，所以是2017年前几名队伍的基本方法。同理可知，Hybrid Task Cascade（HTC）在 COCO 2018 的比赛中也取得了第一名。



##   概述



级联是一种比较经典的结构，在很多任务中都有用到，比如物体检测中的CC-Net，CascadeR-CNN，语义分割中的Deep Layer Cascade等等。然而将这种结构或者思想引入到实例分割中并不是一件直接而容易的事情，如果直接将Mask R-CNN和Cascade R-CNN结合起来，获得的提升是有限的，因此我们需要更多地探索检测和分割任务的关联。



在本篇论文中，我们提出了一种新的实例分割框架，设计了多任务多阶段的混合级联结构，并且融合了一个语义分割的分支来增强Spatial Context。这种框架取得了明显优于Mask R-CNN和Cascade Mask R-CNN的结果。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGCL4mRmib3EFlGOFUq3Zgp6eqpZGKbFpSRegZvQHVpHXGIXQZaYIsjqA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



##   方法



整个框架的演进可以用四张图来表示，其中M表示Mask分支，B表示Box分支，数字表示Stage，M1即为第一个Stage的Mask分支。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGYhNaISicDq2nq6HCoicJ4UmAWJ6CnxOiaW0VjGMYiaGyniaPzf1SuQ8R4hA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



##   进阶准备：Cascade Mask R-CNN



由于Cascade R-CNN在物体检测上的结果非常好，我们首先尝试将Cascade R-CNN和Mask R-CNN直接进行杂交，得到子代Cascade Mask R-CNN，如上图（a）所示。在这种实现里，每一个Stage和Mask R-CNN 相似，都有一个Mask分支和Box分支。当前Stage会接受RPN或者上一个Stage回归过的框作为输入，然后预测新的框和Mask。这也是实验中我们所比较的Baseline，从实验表格可以看到其实这个Baseline已经很强了，但是仍然存在明显的问题，主要在于Cascade MaskR-CNN相比Mask R-CNN在Box AP上提高了3.5个点，但是在Mask AP上只提高了1.2个点。

##  

##   进阶第一步：Interleaved Execution



Cascade R-CNN虽然强行在每一个Stage里面塞下了两个分支，但是这两个分支之间在训练过程中没有任何交互，它们是并行执行的。所以我们提出Interleaved Execution，也即在每个Stage里，先执行Box分支，将回归过的框再交由Mask分支来预测Mask，如上图（b）所示。这样既增加了每个Stage内不同分支之间的交互，也消除了训练和测试流程的Gap。我们发现这种设计对Mask R-CNN和Cascade Mask R-CNN 的Mask分支都有一定提升。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGzOujReSicY448mcCZLxDe4xjsfDgGEk7UgYehGHX7jgSDtQTuvK4Wxw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

##  

##   进阶第二步：Mask Information Flow

##  

这一步起到了很重要的作用，对一般Cascade结构的设计和改进也具有借鉴意义。我们首先回顾原始Cascade R-CNN的结构，每个Stage只有Box分支。当前Stage对下一Stage产生影响的途径有两条：（1）![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGCY45ZkQbHxe8CHv1C1O1RXggzOVS5wIMGftNyqrevyWAQcwicia4hjaQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的输入特征是![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkG0xwsLLvMe0CxnXL2TWghIrficDy3ujlSRYvA0mnxjNZTjECUgItEDAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)预测出回归后的框通RoI Align获得的；（2）![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGCY45ZkQbHxe8CHv1C1O1RXggzOVS5wIMGftNyqrevyWAQcwicia4hjaQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的回归目标是依赖![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkG0xwsLLvMe0CxnXL2TWghIrficDy3ujlSRYvA0mnxjNZTjECUgItEDAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的框的预测的。这就是Box分支的信息流，让下一个Stage的特征和学习目标和当前Stage有关。在Cascade的结构中这种信息流是很重要的，让不同Stage之间在逐渐调整而不是类似于一种Ensemble。



然而在Cascade Mask R-CNN 中，不同Stage之间的Mask分支是没有任何直接的信息流的，![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGZlQUDc5k4bkYznegia07t7hryYEBSnx9WcicKAPSOlCicFmfmUwsVDWcQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)只和当前![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkG0xwsLLvMe0CxnXL2TWghIrficDy3ujlSRYvA0mnxjNZTjECUgItEDAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)通过RoI Align有关联而与![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGHl2OQuvtxMR7icmQ6jkZzJtpUSGvV3NRJ5DnAFP83iatwJxzOosU73hA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)没有任何联系。多个Stage的Mask分支更像用不同分布的数据进行训练然后在测试的时候进行Ensemble，而没有起到Stage间逐渐调整和增强的作用。为了解决这一问题，我们在相邻的Stage的Mask分支之间增加一条连接，提供Mask分支的信息流，让能![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGZlQUDc5k4bkYznegia07t7hryYEBSnx9WcicKAPSOlCicFmfmUwsVDWcQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)知道![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGHl2OQuvtxMR7icmQ6jkZzJtpUSGvV3NRJ5DnAFP83iatwJxzOosU73hA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的特征。具体实现上如下图中红色部分所示，我们将![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGHl2OQuvtxMR7icmQ6jkZzJtpUSGvV3NRJ5DnAFP83iatwJxzOosU73hA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的特征经过一个1x1的卷积做Feature embedding，然后输入到![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGZlQUDc5k4bkYznegia07t7hryYEBSnx9WcicKAPSOlCicFmfmUwsVDWcQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，这样![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGZlQUDc5k4bkYznegia07t7hryYEBSnx9WcicKAPSOlCicFmfmUwsVDWcQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)既能得到Backbone的特征，也能得到上一个Stage的特征。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGz4Mia9A2KibTib1iacSrIcVVaSaDvevmxibFQUxGUoicJW6YBHTkqTp3C9qA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



##   进阶第三步：Semantic Feature Fusion



这一步是我们尝试将语义分割引入到实例分割框架中，以获得更好的Spatial Context。因为语义分割需要对全图进行精细的像素级的分类，所以它的特征是具有很强的空间位置信息，同时对前景和背景有很强的辨别能力。通过将这个分支的语义信息再融合到Box和Mask分支中，这两个分支的性能可以得到较大提升。



在具体设计上，为了最大限度和实例分割模型复用Backbone，减少额外参数，我们在原始的FPN的基础上增加了一个简单的全卷积网络用来做语义分割。首先将FPN的5个level的特征图Resize到相同大小并相加，然后经过一系列卷积，再分别预测出语义分割结果和语义分割特征。这里我们使用COCO-Stuff的标注来监督语义分割分支的训练。红色的特征将和原来的Box和Mask分支进行融合（在下图中没有画出），融合的方法我们也是采用简单的相加。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGyDIMR7WWSvXGIECVniaPXkKCy28X48EOHrxYvYrnF4eunQ1H2gGjrFA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



##   进阶结果



通过上面的几步，在使用ResNet-50的Backbone下，相对Cascade Mask R-CNN可以有1.5个点的Mask AP提升，相对Mask R-CNN可以有2.9个点的提升。在COCO 2017 val子集上的逐步对比试验如下表所示。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGDRGpaib7ia3avztaUGGxVUnRzb9UXC5neyfWFYbjq5zcuwoLTcJmrq8g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



除了纯净版HTC之外，在Paper里我们还给出了在COCO Challenge里面用到的所有步骤和技巧的涨点情况（良心买卖有木有）。



![img](https://mmbiz.qpic.cn/mmbiz_png/JQz7S69vP7GONOO7V425eQO5oSDHKvkGpoGmmu4uQF7qDpXhaIglqich7Dicf55sRKia41ib9MSu9RLvoou1vjq5PQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



##   总结

1、多任务多阶段的混合级联结构。

2、训练时每个Stage内Box和Mask分支采用交替执行。

3、在不同Stage的Mask分支之间引入直接的信息流。

4、语义分割的特征和原始的Box/Mask分支融合，增强Spatial Context。



 

