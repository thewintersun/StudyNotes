## YouTube-Objects dataset

A large-scale database of object videos from YouTube

官方地址： https://data.vision.ee.ethz.ch/cvl/youtube-objects/

组织机构： University of Edinburgh, CALVIN； ETH Zurich, CALVIN； INRIA Grenoble, LEAR



### 介绍

![Annotated example](https://data.vision.ee.ethz.ch/cvl/youtube-objects/youtube-objects_files/prest12cvpr.jpg)

YouTube- objects数据集由通过查询10个 object classes的名称从YouTube收集的视频组成。每个类包含9到24个视频。每个视频的时长从30秒到3分钟不等。这些视频是弱注释的，也就是说，我们确保每个视频包含对应类的一个对象。

In addition to the videos, this release also includes several materials from our paper [1]

- **Bounding-boxes annotations.** For evaluation purposes we annotated the object location in a few hundred video frames for each class (see sec. 6.1 [1]).
- **Point tracks and motion segments.** As produced by [2].
- **Tubes.** Spatio-temporal bounding-boxes as described in section 3.2 [1]. We include all candidate tubes (yellow in the fig. above) as well as the tube automatically selected by our method (blue).

### Dataset release download

![Annotated example](https://data.vision.ee.ethz.ch/cvl/youtube-objects/youtube-objects_files/dataset.png)

As the total download size amounts to 89 GB, we have partitioned the dataset by object class. The following table contains the URLs of the different archives and MATLAB code to access the data. 

### Related publications and software

[1] *A. Prest, C. Leistner, J. Civera, C. Schmid and V. Ferrari.*
     **Learning Object Class Detectors fromWeakly Annotated Video**
     Computer Vision and Pattern Recognition (CVPR), 2012.

[2] *T. Brox, J. Malik.*
     **Object segmentation by long term analysis of point trajectories**
     European Conference on Computer Vision (ECCV), 2010.

