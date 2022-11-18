## YouTube-VOS

A Large-Scale Benchmark for Video Object Segmentation

官方地址：https://youtube-vos.org/

组织者：Adobe, Snap, UIUC, 中科院, 字节跳动

与ICCV2019联合举办Workshop，看榜单竞争激烈，难度挺大。



## What is YouTube-VOS

YouTube-VOS is the first large-scale benchmark that supports multiple video object segmentation tasks.

- Semi-supervised Video Object Segmentation
- Video Instance Segmentation

It also has the following features.

- ==4000+ high-resolution== YouTube videos  **是DAVIS的30倍大**
- ==90+ semantic categories==  **94 object categories**
- 7800+ unique objects
- 190k+ high-quality manual annotations
- 340+ minutes duration



数据集样例：

![1568881442918](D:\Notes\raw_images\1568881442918.png)



相关文档：

Semi-supervised video object segmentation

- [Download the paper that describes the initial YouTube-VOS dataset.](https://arxiv.org/abs/1809.03327)
- [Download the paper that describes a RNN method trained on YouTube-VOS.](https://arxiv.org/abs/1809.00461)

Video instance segmentation

- [Download the paper that describes the video instance segmentation task, dataset and baselines.](https://arxiv.org/abs/1905.04804)



### Tasks

**The first track** targets at **semi-supervised video object segmentation**, which is the same setting as in [the first workshop](https://youtube-vos.org/challenge/2018/). 

**The second track** will be a new task named **video instance segmentation**, which targets at automatically segmenting all object instances of pre-defined object categories from videos.



## Track 1: Video Object Segmentation

### Leaderboard

| Team Name    | Overall       | J_seen     | J_unseen  | F_seen     | F_unseen  | Ranking |
| :----------- | :------------ | :--------- | :-------- | :--------- | :-------- | :------ |
| zszhou       | ==0.818 (1)== | 0.807 (1)  | 0.773 (2) | 0.847 (1)  | 0.847 (2) | 1       |
| theodoruszq  | 0.817 (2)     | 0.800 (2)  | 0.779 (1) | 0.833 (2)  | 0.855 (1) | 2       |
| zxyang1996   | 0.804 (3)     | 0.794 (3)  | 0.759 (4) | 0.833 (3)  | 0.831 (4) | 3       |
| swoh         | 0.802 (4)     | 0.788 (4)  | 0.759 (3) | 0.825 (4)  | 0.835 (3) | 4       |
| youtube_test | 0.791 (5)     | 0.779 (5)  | 0.747 (5) | 0.815 (5)  | 0.822 (5) | 5       |
| Jono         | 0.714 (7)     | 0.703 (10) | 0.680 (7) | 0.736 (10) | 0.740 (8) | 6       |
| andr345      | 0.710 (8)     | 0.699 (11) | 0.667 (8) | 0.732 (11) | 0.740 (7) | 7       |

### Team Information

| Team Name    | Team Members                                                 | Organization                                                 |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| zszhou       | Zhishan Zhou [1], Lejian Ren [2], Pengfei Xiong [3], Yifei Ji [4], Peisen Wang [3], Haoqiang Fan [3], Si Liu [5] | [1] Beijing University of Posts and Telecommunications, [2] Institute of Information Engineering, Chinese Academy of Sciences, [3] Megvii Inc, [4] Tsinghua University, [5] Beihang University |
| theodoruszq  | Qiang Zhou[1], Zilong Huang[1], Lichao Huang[2], Yongchao Gong[2], Han Shen[2], Wenyu Liu[1] and Xinggang Wang[1] | [1] MCLAB, Huazhong University of Science and Technology, China [2] Horizon Robotics, China |
| zxyang1996   | Zongxin Yang, Peike Li, Yunchao Wei and Yi Yang              | ReLER lab, Centre for Artificial Intelligence, University of Technology Sydney |
| swoh         | Seoung Wug Oh [1], Joon-Young Lee [2], Ning Xu [2], Seon Joo Kim [1] | [1] Yonsei University, [2] Adobe                             |
| youtube_test | Dongdong Yu*, Kai Su*, Hengkai Guo, Jian Wang, Kaihui Zhou, Yuanyuan Huang, Minghui Dong, Jie Shao, Changhu Wang (* denotes equal contribution) | ByteDance AI Lab                                             |
| Jono         | Jonathon Luiten, Paul Voigtlaender, Bastian Leibe            | RWTH Aachen University                                       |
| andr345      | andreas.robinson@liu.se                                      | -                                                            |



## Track 2: Video Instance Segmentation

### Leaderboard

| Team Name    | mAP           | AP50       | AP75       | AR1        | AR10       | Ranking |
| :----------- | :------------ | :--------- | :--------- | :--------- | :--------- | :------ |
| Jono         | ==0.467 (1)== | 0.697 (1)  | 0.509 (1)  | 0.462 (1)  | 0.537 (2)  | 1       |
| foolwood     | 0.457 (2)     | 0.674 (3)  | 0.490 (3)  | 0.435 (5)  | 0.507 (4)  | 2       |
| bellejuillet | 0.450 (3)     | 0.636 (6)  | 0.502 (2)  | 0.447 (3)  | 0.503 (5)  | 3       |
| linhj        | 0.449 (4)     | 0.665 (4)  | 0.486 (5)  | 0.453 (2)  | 0.538 (1)  | 4       |
| mingmingdiii | 0.444 (5)     | 0.684 (2)  | 0.487 (4)  | 0.436 (4)  | 0.508 (3)  | 5       |
| xiAaonice    | 0.400 (7)     | 0.578 (10) | 0.449 (7)  | 0.396 (10) | 0.452 (10) | 6       |
| guwop        | 0.400 (8)     | 0.608 (8)  | 0.439 (9)  | 0.412 (8)  | 0.491 (6)  | 7       |
| exing        | 0.397 (9)     | 0.621 (7)  | 0.426 (10) | 0.414 (6)  | 0.461 (9)  | 8       |

### Team Information

| Team Name    | Team Members                                                 | Organization                                                 |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Jono         | Jonathon Luiten [1][2], Phillip Torr[2], Bastian Leibe [1]   | [1] RWTH Aachen University, [2] University of Oxford         |
| foolwood     | Qiang Wang [1], Yi He [2], Xiaoyun Yang [2], Zhao Yang [3], Philip H.S. Torr [3] | [1] Institute of Automation, Chinese Academy of Sciences, [2] Intellimind Ltd, [3] University of Oxford |
| bellejuillet | Qianyu Feng, Zongxin Yang, Peike Li, Yunchao Wei, Yi Yang    | ReLER lab, Centre for Artificial Intelligence, University of Technology Sydney |
| linhj        | Huaijia Lin [1], Shu Liu [2], Mengdan Zhang [2], Xiaojuan Qi [3] | [1] The Chinese University of Hong Kong, [2] Tencent YouTu X-Lab, [3] University of Oxford |
| mingmingdiii | Minghui Dong, Jian Wang, Yuanyuan Huang,Dongdong Yu, Kai Su, Kaihui Zhou, Jie Shao, Shiping Wen, Changhu Wang | ByteDance AI Lab && Huazhong University of Science and Technology |
| xiAaonice    | Xiaoyu Liu, Haibin Ren and Tingmeng Ye                       | Alibaba                                                      |
| guwop        | Patrick Poirson, Alexander C. Berg                           | UNC Chapel Hill                                              |
| exing        | Eliot Xing                                                   | Georgia Institute of Technology & Wave Computing             |