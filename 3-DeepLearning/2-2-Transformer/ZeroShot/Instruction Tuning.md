## Finetuned Language Models Are Zero-Shot Learners

论文地址：https://arxiv.org/abs/2109.01652

作者：Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le

机构：Google Research

发表：ICLR 2022

其他文章：https://amitness.com/2020/05/zero-shot-text-classification/



### 摘要

This paper explores a simple method for improving the zero-shot learning abilities of language models. We show that instruction tuning -- finetuning language models on a collection of tasks described via instructions -- substantially improves zero-shot performance on unseen tasks.

We take a 137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call **FLAN**, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 20 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of finetuning datasets, model scale, and natural language instructions are key to the success of instruction tuning.



这个文章提出了一个**Instruction tuning**的概念，用这种方式精调大模型之后可以显著提升大模型在NLI和阅读理解的表现：

<img src="D:\Notes\raw_images\image-20221008162148864.png" alt="image-20221008162148864" style="zoom:67%;" />

从官网给的几个例子，可以看到Instruct版本的模型相比于GPT-3可以更好的完成**问答、分类和生成任务**，是**更符合实际应用需求的模型**。

而看到这里想必大家和我一样疑惑，**这个Instruction怎么和前阵子的网红Prompt有些像？**感觉傻傻分不清楚？

不急，我们这就来好好看看。



### 二、什么是Instruction Tuning

让我们先抛开脑子里的一切概念，把自己当成一个模型。我给你两个任务：

1. 带女朋友去了一家餐厅，她吃的很开心，这家餐厅太__了！

2. 判断这句话的情感：带女朋友去了一家餐厅，她吃的很开心。选项：A=好，B=一般，C=差

你觉得哪个任务简单？请把序号打在公屏上。做判别是不是比做生成要容易？**Prompt就是第一种模式，Instruction就是第二种。**

Instruction Tuning和Prompt的核心一样，就是去发掘语言模型本身具备的知识。而他们的不同点就在于，Prompt是去激发语言模型的**补全能力**，比如给出上半句生成下半句、或者做完形填空，**都还是像在做language model任务**，它的模版是这样的：

![图片](D:\Notes\raw_images\weixin16318645440466.png)

而Instruction Tuning则是激发语言模型的**理解能力**，通过给出更明显的指令/指示，让模型去理解并做出正确的action。比如NLI/分类任务：

<img src="D:\Notes\raw_images\image-20221008162210680.png" alt="image-20221008162210680" style="zoom: 80%;" />

还有一个不同点，就是==Prompt在没精调的模型上也能有一定效果，而Instruction Tuning则必须对模型精调==，让模型知道这种指令模式。

但是，Prompt也有精调，经过Prompt tuning之后，模型也就学习到了这个Prompt模式，精调之后跟Instruction Tuning有啥区别呢？

这就是Instruction Tuning巧妙的地方了，（我看到的）**Prompt tuning都是针对一个任务的**，比如做个情感分析任务的prompt tuning，精调完的模型只能用于情感分析任务，**而经过Instruction Tuning多任务精调后，可以用于其他任务的zero-shot！**

<img src="D:\Notes\raw_images\image-20221008162235722.png" alt="image-20221008162235722" style="zoom:80%;" />

我知道上面的解释有些绕，请深深体会一下。



### 三、怎么做Instruction Tuning

理解了Instruction Tuning的概念之后，再看实验方法就清晰多了。作者把62个NLP任务分成了12个类，训练时在11个上面精调，在1个上面测试zero-shot效果，这样可以保证模型真的没见过那类任务，看模型是不是真的能理解「指令」：

![图片](D:\Notes\raw_images\weixin16318645440469.png)

像Prompt一样，作者也会为每个任务设计10个指令模版，测试时看平均和最好的表现：

<img src="D:\Notes\raw_images\weixin163186454404610.png" alt="图片" style="zoom:80%;" />

### 四、效果

<img src="D:\Notes\raw_images\image-20221008162325268.png" alt="image-20221008162325268" style="zoom:80%;" />

通过上述多任务指令精调的FLAN模型在大部分情况可以超过GPT-3的zero-shot（绿色箭头）甚至是few-shot（绿色三角）表现，其中有监督模型a=T5 11B，b=BERT-large：

![图片](D:\Notes\raw_images\weixin163186454404611.png)

![图片](D:\Notes\raw_images\weixin163186454404612.png)

![图片](D:\Notes\raw_images\weixin163186454404613.png)

![图片](D:\Notes\raw_images\weixin163186454404614.png)

同时也可以和Prompt相结合，会有更大提升：

![图片](D:\Notes\raw_images\weixin163186454404615.png)

但遗憾的是，这个方法只在大模型上有效果，小模型上甚至会降低表现。作者认为是小模型容量有限，只学一个任务的知识就很不容易了：

![图片](D:\Notes\raw_images\weixin163186454404616.png)

### 五、总结

当时看这篇文章的第一反应，是觉得这个idea难得没有很多人做过吗？Prompt、Instruction，从GPT-2开始就有了吧。然而仔细想，却发现之前研究主要是针对单任务的少样本情况，**并没有研究这种多任务的Prompt、指令泛化**。

这个研究的应用潜力显然更大，而且谷歌和OpenAI居然不谋而合都在做，同时在应用时使用者还可以对任务进行一定的精调：

![图片](D:\Notes\raw_images\weixin163186454404617.png)

再往深想，Instruction和Prompt一样存在手工设计模版的问题，怎样把模版参数化、或者自动挖掘大量模版从而提升指令精调的效果，也是一个方向。