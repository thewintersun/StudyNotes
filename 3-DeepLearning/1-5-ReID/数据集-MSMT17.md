### **MSMT17**

官网地址：https://www.pkuvmc.com/publications/msmt17.html

**Description to MSMT17**

![1564019193311](D:\Notes\raw_images\1564019193311.png)To collect a large-scale person re-identification dataset-MSMT17, we utilize an ==15-camera== network deployed in campus. This camera network contains ==12 outdoor cameras== and ==3 indoor cameras==. We select 4 days with ==different weather conditions== in a month for video collection. For each day, 3 hours of videos taken in the morning, noon, and afternoon, respectively, are selected for pedestrian detection and annotation. Our final raw video set contains ==180 hours of videos==, 12 outdoor cameras, 3 indoor cameras, and 12 time slots. ==Faster RCNN== is utilized for pedestrian bounding box detection. Three labelers go through the detected bounding boxes and annotate ID label for 2 months. Finally, ==126,441 bounding boxes== of ==4,101 identities== are annotated. Some statistics on MSMT17 are shown in above. Compared with existing datasets, we summarize the new features in MSMT17 into the following aspects:

(1) Larger number of identities, bounding boxes, and cameras.
(2) Complex scenes and backgrounds.
(3) Multiple time slots result in severe lighting changes.
(4) More reliable bounding box detector.

| Dataset    | MSMT17         | Duke [1] [8] | Market [2] | CUHK03 [3] | CUHK01 [4] | VIPeR [5] | PRID [6] | CAVIAR [7] |
| ---------- | -------------- | ------------ | ---------- | ---------- | ---------- | --------- | -------- | ---------- |
| BBoxes     | 126,441        | 36,411       | 32,668     | 28,192     | 3,884      | 1,264     | 1,134    | 610        |
| Identities | 4,101          | 1,812        | 1,501      | 1,467      | 971        | 632       | 934      | 72         |
| Cameras    | 15             | 8            | 6          | 2          | 10         | 2         | 2        | 2          |
| Detector   | Faster RCNN    | hand         | DPM        | DPM,hand   | hand       | hand      | hand     | hand       |
| Scene      | outdoor,indoor | outdoor      | outdoor    | indoor     | indoor     | outdoor   | outdoor  | indoor     |

**Reference**

[1] Z. Zheng et al. Unlabeled samples generated by gan improve the person re-identification baseline in vitro. In ICCV, 2017.
[2] L. Zheng et al. Scalable person re-identification: A benchmark. In ICCV, 2015.
[3] W. Li et al. Deepreid: Deep filter 918 pairing neural network for person re-identification. In CVPR, 2014.
[4] W. Li et al. Human reidentification with transferred metric learning. In ACCV, 2012.
[5] D. Gray et al. Viewpoint invariant pedestrian recogni- tion with an ensemble of localized features. In ECCV, 2008.
[6] M. Hirzer et al. Person re-identification by descriptive and discriminative classifica- tion. In SCIA, 2011.
[7] D. S. Cheng et al. Custom pictorial structures for re-identification. In BMVC, 2011.
[8] E. Ristani et al. Performance measures and a data set for multi-target, multi-camera tracking. In ECCV Workshop, 2016.

The [dataset](https://docs.google.com/forms/d/e/1FAIpQLScIGhLvB2GzIXjX1oFW0tNUWxkbK2l0fYG5Q9vX93ls2BVsQw/viewform?usp=sf_link) on MSMT17 has been released. 

The [evaluation code](https://github.com/JoinWei-PKU/MSMT17_Evaluation) on MSMT17 has been released.  

0000_008_01_0303morining_0019_2.jpg represents the image was captured from camera1 
0000_009_05_0303morining_0029_1.jpg represents the image was captured from camera5

#### LeaderBoard:

state of the art methods: https://www.pkuvmc.com/publications/state_of_the_art.html

| **Reference**                                                | **MSMT17** |         |         |      |      | **Notes**                        |
| ------------------------------------------------------------ | ---------- | ------- | ------- | ---- | ---- | -------------------------------- |
| rank-1                                                       | rank-5     | rank-10 | rank-20 | mAP  |      |                                  |
| L. Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. In CVPR, 2018. | 47.6       | 65.0    | 71.8    | 78.2 | 23.0 | GoogleNet[1], euclidean distance |
|                                                              | 58.0       | 73.6    | 79.4    | 84.5 | 29.7 | PDC[2], euclidean distance       |
|                                                              | 61.4       | 76.8    | 81.6    | 85.9 | 34.0 | GLAD[3], euclidean distance      |

[1] C. Szegedy et al. Going deeper with convolutions. In CVPR, 2015.
[2] C. Su et al. Pose-driven deep convolutional model for person re-identification. In ICCV, 2017.
[3] L. Wei et al. Glad: Global-local-alignment descriptor for pedestrian retrieval. In ACM MM, 2017.

![1564022948200](D:\Notes\raw_images\1564022948200.png)

 