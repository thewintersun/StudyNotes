### DAVIS: Densely Annotated VIdeo Segmentation

In-depth analysis of the state-of-the-art in video object segmentation

官方地址：https://davischallenge.org/

组织机构：ETH, Google

目前为止每年与CVPR组织Workshop



## Datasets

- [DAVIS 2016:](https://davischallenge.org/davis2016/code.html) In each video sequence a single instance is annotated.
- [DAVIS 2017 Semi-supervised:](https://davischallenge.org/davis2017/code.html#semisupervised) In each video sequence multiple instances are annotated.
- [DAVIS 2017 Unsupervised:](https://davischallenge.org/davis2017/code.html#unsupervised) In each video sequence multiple instances are annotated.

一共150个小视频，90个训练和验证集，30个Test-Dev, 30个Challenge.

- Train + Val: ==90 sequences== from DAVIS 2017.

- Test-Dev 2017: ==30 sequences== from DAVIS 2017 Semi-supervised. Ground truth not publicly available.

- Test-Challenge 2017: ==30 sequences== from DAVIS 2017 Semi-supervised. Ground truth not publicly available.
  Feel free to train or pre-train your algorithms on any other dataset apart from DAVIS (Youtube-VOS, MS COCO, Pascal, etc.) or use the full resolution DAVIS annotations and images.

  

## Tasks 

数据集下载地址： https://davischallenge.org/davis2017/code.html

*The official metrics will be computed using the images and annotations at 480p resolution, but feel free to use the full resolution ones (4k, 1080p, etc.) in any step of your research.*

### Semi-supervised 

The semi-supervised scenario assumes the user inputs a full mask of the object of interest in the first frame of a video sequence. ==Methods have to produce the segmentation mask for that object in the subsequent frames==.

评测代码：https://github.com/davisvideochallenge/davis2017-evaluation 

- The per-object measures are those described in the original DAVIS CVPR 2016 paper: Region Jaccard (J) and Boundary F measure (F).

- The overall ranking measures are computed as the mean between J and F, both averaged over all objects. Precise definitions available in the papers [DAVIS 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf) and [DAVIS 2017](https://arxiv.org/abs/1704.00675) publications.



### Unsupervised

The unsupervised scenario assumes that the user does not interact with the algorithm to obtain the segmentation masks. Methods should provide a set of object candidates with no overlapping pixels that span through the whole video sequence. ==This set of objects should contain at least the objects that capture human attention when watching the whole video sequence== i.e objects that are more likely to be followed by human gaze.
More information in the [DAVIS 2019 publication](https://arxiv.org/abs/1905.00737) and the [Codalab submission site.](https://competitions.codalab.org/competitions/20515).

*The **TrainVal** video sequences are the same as in Semi-supervised, but the annotations are different.*

评测代码：https://github.com/davisvideochallenge/davis2017-evaluation



### Interactive

The interactive scenario assumes the user gives iterative refinement inputs to the algorithm, in our case in the form of a scribble, to segment the objects of interest. ==Methods have to produce a segmentation mask for that object in all the frames of a video sequence taking into account all the user interactions==.

In the classical DAVIS Semi-supervised Challenge track, the task is to segment an object in a *semi-supervised* manner, i.e. the given input is the ground truth mask of the first frame. In the **DAVIS Interactive Challenge**, in contrast, the user input is scribbles, which can be drawn much faster by humans and thus are a more realistic type of input.

![1568880469267](C:\Users\j00496872\Desktop\Notes\raw_images\1568880469267.png)

More information in the [DAVIS 2018 publication](https://arxiv.org/abs/1803.00557) and the [Interactive Python package documentation](https://interactive.davischallenge.org/).

评测代码：https://github.com/albertomontesg/davis-interactive



### Prizes

- The **winner** of the competition will get an **NVIDIA GEFORCE RTX 2080Ti**.
- All the other participants invited to the workshop will get a subscription to Adobe CC for 1 year.



### Papers

- Right after the Challenge closes (24th May) we will invite all participants to submit a **short abstract**(400 words maximum) of their method (Deadline 29th of May, 23:59 UTC).

- Together with the results obtained, we will decide which teams are accepted at the workshop. Date of notification June 3rd.

- Accepted teams will be able to submit a paper describing their approach (Deadline 12th June, 23:59 UTC). The template of the paper is the same as CVPR, but length will be limited to **4 pages including references**.

- Papers will also be invited to the workshop in form of **oral presentation or poster**.

- Accepted papers will be self-published in the web of the challenge (==not in the official proceedings==, although they have the same *value*).

  

## LeaderBoard - 2019

 The DAVIS Challenge 2019 finished with ==28 teams==. Thanks to everyone that participated!

#### Leaderboard Semi-Supervised track

| Participant    | Finn_zhang | Jono | Tmtriet | Swoh | Theodoruszq | Pandayf_jjj | Andr345 | BytedanceCV |
| -------------- | ---------: | ---: | ------: | ---: | ----------: | ----------: | ------: | ----------: |
| Rank           |        1st |  2nd |     3rd |  4th |         5th |         6th |     7th |         8th |
| Mean J And F ↑ |   ==76.7== | 76.2 |    75.4 | 75.2 |        73.1 |        71.3 |    70.6 |        69.2 |
| J Mean ↑       |   ==72.7== | 72.9 |    72.4 | 72.6 |        70.1 |        67.7 |    68.5 |        66.0 |
| J Recall ↑     |       81.5 | 81.7 |    81.7 | 81.0 |        77.3 |        74.8 |    78.1 |        73.4 |
| J Decay ↓      |       19.5 | 16.3 |    11.0 | 21.2 |        24.8 |        24.7 |    20.3 |        28.5 |
| F Mean ↑       |   ==80.6== | 79.4 |    78.4 | 77.7 |        76.1 |        75.0 |    72.8 |        72.3 |
| F Recall ↑     |       87.3 | 86.7 |    87.6 | 84.9 |        84.0 |        81.2 |    84.2 |        80.4 |
| F Decay ↓      |       22.0 | 19.5 |    12.9 | 24.5 |        28.3 |        27.5 |    24.0 |        31.1 |

#### Leaderboard Interactive track

| Position | Participant                         | Session ID | AUC  | J&F@60s |
| -------- | ----------------------------------- | ---------- | ---- | ------- |
| 1        | Seoung Wug Oh (Yonsei University)   | 069a9f2c   | 78.3 | 79.1    |
| 2        | Yuk Heo (Korea University)          | 19b36910   | 64.7 | 60.9    |
| 3        | Zihang Lin (Sun Yat-sen University) | 07af0840   | 62.1 | 60.1    |
| 4        | YK_CL (Youku Company)               | 98d2b5a5   | 58.9 | 49.1    |

#### Leaderboard Unsupervised track

| Participant    | Idilesenzulfikar |     JayY | Don_don | Davis_try |
| -------------- | ---------------: | -------: | ------: | --------: |
| Rank           |              1st |      2nd |     3rd |       4th |
| Mean J And F ↑ |         ==56.4== |     56.2 |    51.6 |      50.4 |
| J Mean ↑       |             53.4 | ==53.5== |    48.7 |      47.5 |
| J Recall ↑     |             60.9 |     61.3 |    55.1 |      54.2 |
| J Decay ↓      |              1.5 |     -2.1 |     4.0 |       3.2 |
| F Mean ↑       |         ==59.4== |     59.0 |    54.5 |      53.3 |
| F Recall ↑     |             64.1 |     63.2 |    59.4 |      56.9 |
| F Decay ↓      |              5.8 |      0.1 |     7.7 |       5.5 |



## Publications Semi-Supervised track

1. Object-based Spatial Similarity for Semi-supervised Video Object Segmentation
   B. Wang, C. Zheng, N. Wang , S. Wang, X. Zhang, S. Liu,
   S. Gao, K. Lu, D. Zhang, L. Shen, Y. Wang, Y. Xu
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-1st-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-1st'))]
2. Combining PReMVOS with Box-Level Tracking for the 2019 DAVIS Challenge
   J. Luiten, P. Voigtlaender, B. Leibe
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-2nd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-2nd'))]
3. Guided Instance Segmentation Framework for Semi-supervised Video Instance Segmentation
   M. Tran, T. Le, T. V. Nguyen, T. Ton, T. Hoang, N. Bui, T. Do,
   Q. Luong, V. Nguyen, D. A. Duong, M. N. Do
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-3rd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-3rd'))]
4. A Unified Model for Semi-supervised and Interactive Video Object Segmentation using
   Space-time Memory Networks
   S. W. Oh, J. Lee, N. Xu, S. J. Kim
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-4th-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-4th'))]
5. *Q. Zhou, S. Chen, L. Huang, X. Wang
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
6. Discriminative Learning and Target Attention for the 2019 DAVIS Challenge
   on Video Object Segmentation
   A. Robinson, F. J. Lawin, M. Danelljan, M. Felsberg
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-7th-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-7th'))]
7. An Empirical Study of Propagation-based Methods for Video Object Segmentation
   H. Guo, W. Wang, G. Guo, H. Li, J. Liu, Q. He, X. Xiao
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Semisupervised-Challenge-8th-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Semi-8th'))]

## Publications Interactive track

1. A Unified Model for Semi-supervised and Interactive Video Object Segmentation using
   Space-time Memory Networks
   S. W. Oh, J. Lee, N. Xu, S. J. Kim
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Interactive-Challenge-1st-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Int-1st'))]
2. Interactive Video Object Segmentation Using Sparse-to-Dense Networks
   Y. Heo, Y. J. Koh, C. Kim
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Interactive-Challenge-2nd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Int-2nd'))]
3. Interactive Video Object Segmentation via Spatio-temporal Context Aggregation
   and Online Learning
   Z. Lin, J. Xie, C. Zhou, J. Hu, W. Zheng
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Interactive-Challenge-3rd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Int-3rd'))]
4. Robust Multiple Object Mask Propagation with Efficient Object Tracking
   H. Ren, Y. Yang, X. Liu
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Interactive-Challenge-4th-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Int-4th'))]

## Publications Unsupervised track

1. UnOVOST: Unsupervised Offline Video Object Segmentation and Tracking for
   the 2019 Unsupervised DAVIS Challenge
   I. E. Zulfikar, J. Luiten, B. Leibe
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Unsupervised-Challenge-1st-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Unsup-1st'))]
2. Video Segmentation by Detection for the 2019 Unsupervised DAVIS Challenge
   Z. Yang, Q. Wang, S. Bai, W. Hu, P.H.S. Torr
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Unsupervised-Challenge-2nd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Unsup-2nd'))]
3. Key Instance Selection for Unsupervised Video Object Segmentation
   D. Cho, S. Hong, S. Kang, J. Kim
   *The 2019 DAVIS Challenge on Video Object Segmentation - CVPR Workshops, 2019*
   [[PDF](https://davischallenge.org/challenge2019/papers/DAVIS-Unsupervised-Challenge-3rd-Team.pdf)] [[BibTex](javascript:toggleBibtex('DAVIS2019-Unsup-3rd'))]