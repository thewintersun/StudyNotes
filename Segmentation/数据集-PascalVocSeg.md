#### PascalVOC 2012

官网地址：http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html 

数据下载地址：http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 

==20 classes==. The train/val data has ==11,530 images== containing ==27,450 ROI== annotated objects and ==6,929 segmentations==.	

The twenty object classes that have been selected are:

- *Person:* person
- *Animal:* bird, cat, cow, dog, horse, sheep
- *Vehicle:* aeroplane, bicycle, boat, bus, car, motorbike, train
- *Indoor:* bottle, chair, dining table, potted plant, sofa, tv/monitor



#### 任务类型

1. **Classification**: For each of the twenty classes, predicting presence/absence of an example of that class in the test image.
2. **Detection**: Predicting the bounding box and label of each object from the twenty target classes in the test image.
3. **Segmentation**: Generating pixel-wise segmentations giving the class of the object visible at each pixel, or "background" otherwise.

![1562812768439](C:\Users\j00496872\Desktop\Notes\raw_images\1562812768439.png)

4. **Action Classification:** Predicting the action(s) being performed by a person in a still image. 10 action classes + "other"

![1562812818944](C:\Users\j00496872\Desktop\Notes\raw_images\1562812818944.png)

5. **Person Layout**: Predicting the bounding box and label of each part of a person (head, hands, feet).

![1562812842778](C:\Users\j00496872\Desktop\Notes\raw_images\1562812842778.png)



#### 语义分割LeaderBoard

所有榜单的LeaderBoard链接：http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php 

语义分割 - 自训练数据集 - 成绩排名:

 http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6

| Title                                                        | mAP  | 机构                                                       | 方法说明                                                     | submit date |
| ------------------------------------------------------------ | :--: | ---------------------------------------------------------- | :----------------------------------------------------------- | :---------- |
| [DeepLabv3+_JFT](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=15347)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_DeepLabv3+_JFT) | 89.0 | Google Inc.                                                | DeepLabv3+ by fine-tuning from the model pretrained on==JFT-300M dataset==. For details, please refer to https://arxiv.org/abs/1802.02611. | 09-Feb-2018 |
| [DeepLabv3+_AASPP](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=17681)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_DeepLabv3+_AASPP) | 88.5 | Tsinghua University                                        | DeepLabv3+ with ==Attention==Atrous Spatial Pyramid Pooling. | 22-May-2018 |
| [SRC-B-MachineLearningLab](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=17060)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_SRC-B-MachineLearningLab) | 88.5 | Samsung R&D Institue China - Beijing, Machine Learning Lab | The model is ==pre-trained on ImageNet==, and fine-turned on ==COCO VOC SBD==. The result is tested by ==multi scale and flip==. The paper is in preparing. | 19-Apr-2018 |
| [MSCI](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=18202)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_MSCI) | 88.0 | Shenzhen University                                        | We propose a novel scheme for aggregating features from different scales, which we refer to as ==Multi-Scale Context Intertwining (MSCI)==. Please see our paper http://vcc.szu.edu.cn/Di_Lin/papers/MSCI_eccv2018.pdf | 08-Jul-2018 |
| [ExFuse](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=17673)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_ExFuse) | 87.9 | Fudan University, Megvii Inc.                              | For more details, please refer to https://arxiv.org/abs/1804.03821. | 22-May-2018 |
| [DeepLabv3+](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=15346)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_DeepLabv3+) | 87.8 | Google Inc.                                                | Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task.  For details, please refer to https://arxiv.org/abs/1802.02611. | 09-Feb-2018 |
| [** CFNet **](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=25540)[ [?\]](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_CFNet) | 87.2 | Amazon                                                     | including increasing the receptive field and aggregating pyramid feature representations.  We go beyond global context and explore the fine-grained representation using co-occurrent features by introducing Co-occurrent Feature Model, which predicts the distribution of co-occurrent features for a given target. To leverage the semantic context in the co-occurrent features, we build an Aggregated Co-occurrent Feature (ACF) Module by aggregating the probability of the co-occurrent feature within the co-occurrent context. | 12-Jun-2019 |