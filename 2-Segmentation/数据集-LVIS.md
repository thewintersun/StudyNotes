### LVIS: A Dataset for Large Vocabulary Instance Segmentation  

###### A new dataset for long tail object recognition.

官网地址：https://www.lvisdataset.org/

论文地址： https://arxiv.org/abs/1908.03195



![1572858940097](D:\Notes\raw_images\1572858940097.png)

##### Overview

Today, rigorous evaluation of general purpose object detectors is mostly performed in the few category regime (e.g. 80) or when there are a large number of training examples per category (e.g. 100 to 1000+). LVIS provides an opportunity to enable research in the setting where there are a large number of categories and where per-category data is sometimes scarce.

Given that state-of-the-art deep learning methods for object detection perform poorly in the low-sample regime, we believe that our dataset poses an important and exciting new scientific challenge.

##### Rules and Awards

The LVIS Challenge follows the same rules and awards as the[ COCO Challenge](http://cocodataset.org/#detection-2019). Please take note of the requirement to submit a technical report describing your entry. Participants must submit a technical report that includes a detailed ablation study of their submission (suggested length 1-4 pages). The reports will be made public. Please, use this [latex template ](http://cocodataset.org/files/tech_report_template.zip)for the report and send it to [coco.iccv19@gmail.com](mailto:coco.iccv19@gmail.com).

##### External Data Policy

The use of external data is permitted under two conditions: It does not contain any images in the LVIS v0.5 test set; andThe submission technical report must include: A description of the external data (including at least: the name of the dataset, the number of images in it, how the images were collected if it is not a public dataset, the annotation type such as image-level label or instance-level boxes or polygons, and a list of the categories that are annotated);  andAn ablation study showing the impact of the external data on the LVIS v0.5 validation set while holding all other design factors constant. Failure to include (a) and (b) will invalidate the submission.

##### Evaluation Server

The challenge is hosted on [EvalAI](https://evalai.cloudcv.org/). The evaluation server is now live at [LVIS Challenge 2019](https://evalai.cloudcv.org/web/challenges/challenge-page/442/overview).

##### Baselines

The [LVIS paper on arXiv](https://arxiv.org/abs/1908.03195) now includes two new appendix sections that will be of great interest to anyone who's participating in the challenge. Appendix B shows two Mask R-CNN baselines on the LVIS v0.5 val set and Appendix C is an analysis of how results transfer from the val set to the test set that will be used in the challenge.

  ![1572858412747](D:\Notes\raw_images\1572858412747.png)

**Note**: LVIS uses the COCO 2017 train, validation, and test image sets. If you have already downloaded the COCO images, ==you only need to download the LVIS annotations==. Please refer this [link](https://groups.google.com/forum/#!msg/lvis-dataset/pxFYMCyi6MY/G8vwKLwxAAAJ) if you are having trouble with image file paths for val json.

#### Data Format

LVIS has annotations for instance segmentations in a format similar to [COCO](http://cocodataset.org/#home). The annotations are stored using JSON. The[ LVIS API](https://github.com/lvis-dataset/lvis-api) can be used to access and manipulate annotations. The JSON file has the following format:

![1572858513872](D:\Notes\raw_images\1572858513872.png)