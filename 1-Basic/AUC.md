## AUC

 AUC（Area Under Curve）被定义为ROC曲线下与坐标轴围成的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。AUC越接近1.0，检测方法真实性越高; 等于0.5时，则真实性最低，无应用价值。

AUC（Area Under Curve）被定义为ROC曲线下的面积。我们往往使用AUC值作为模型的评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

其中，ROC曲线全称为受试者工作特征曲线 （receiver operating characteristic curve），它是根据一系列不同的二分类方式（分界值或决定阈），以真阳性率（敏感性）为纵坐标，假阳性率（1-特异性）为横坐标绘制的曲线。

AUC就是衡量学习器优劣的一种性能指标。从定义可知，AUC可通过对ROC曲线下各部分的面积求和而得。

### ROC 

ROC曲线的横坐标是**伪阳性率**（也叫假正类率，False Positive Rate），纵坐标是**真阳性率**（真正类率，True Positive Rate），相应的还有**真阴性率**（真负类率，True Negative Rate）和**伪阴性率**（假负类率，False Negative Rate）。这四类指标的计算方法如下：

（1）伪阳性率（[FPR](https://baike.baidu.com/item/FPR/6343296)）：判定为正例却不是真正例的概率，即真负例中判为正例的概率

（2）真阳性率（[TPR](https://baike.baidu.com/item/TPR/5548598)）：判定为正例也是真正例的概率，即真正例中判为正例的概率（也即正例召回率）

（3）伪阴性率（[FNR](https://baike.baidu.com/item/FNR/5609400)）：判定为负例却不是真负例的概率，即真正例中判为负例的概率。

（4）真阴性率（[TNR](https://baike.baidu.com/item/TNR/4663071)）：判定为负例也是真负例的概率，即真负例中判为负例的概率。

ROC（Receiver Operating Characteristic）曲线，又称[接受者操作特征曲线](https://baike.baidu.com/item/接受者操作特征曲线/2075302)。该曲线最早应用于雷达信号检测领域，用于区分信号与噪声。后来人们将其用于评价模型的预测能力，ROC曲线是基于[混淆矩阵](https://baike.baidu.com/item/混淆矩阵/10087822)得出的。一个二分类模型的阈值可能设定为高或低，每种阈值的设定会得出不同的 FPR 和 TPR ，将同一模型每个阈值的 (FPR, TPR) 坐标都画在 ROC 空间里，就成为特定模型的ROC曲线。ROC曲线横坐标为假正率（FPR），纵坐标为真正率（TPR）。

AUC就是曲线下面积，在比较不同的分类模型时，可以将每个模型的ROC曲线都画出来，比较曲线下面积做为模型优劣的指标。ROC 曲线下方的面积(Area under the Curve)，其意义是：

（1）因为是在1x1的方格里求面积，AUC必在0~1之间。

（2）假设阈值以上是阳性，以下是阴性；

（3）若随机抽取一个阳性样本和一个阴性样本，分类器正确判断阳性样本的值高于阴性样本的概率 = AUC 。

（4）简单说：AUC值越大的分类器，正确率越高。

**从AUC 判断分类器（预测模型）优劣的标准：**

- AUC = 1，是完美分类器。
- AUC = [0.85, 0.95], 效果很好
- AUC = [0.7, 0.85], 效果一般
- AUC = [0.5, 0.7],效果较低，但用于预测股票已经很不错了
- AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

使用：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

```python
sklearn.metrics.auc(x, y) 

import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)
# 0.75
```

