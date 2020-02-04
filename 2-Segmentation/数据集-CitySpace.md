官网地址：https://www.cityscapes-dataset.com/ 

## The Cityscapes Dataset

The Cityscapes Dataset focuses on ==semantic understanding of urban street scenes==. In the following, we give an overview on the design choices that were made to target the dataset’s focus.

Type of annotations

- Semantic （语义）
- Instance-wise （实例）
- Dense pixel annotations （密集像素点标注）

Complexity

- 30 classes  （30 类）
- See [Class Definitions](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions) for a list of all classes and have a look at the applied [labeling policy](https://www.cityscapes-dataset.com/dataset-overview/#labeling-policy).

Diversity

- 50 cities （50个城市，同一个国家）
- Several months (spring, summer, fall)
- Daytime
- Good/medium weather conditions
- Manually selected frames
  - Large number of dynamic objects
  - Varying scene layout
  - Varying background

Volume

- 5 000 annotated images with fine annotations ([examples](https://www.cityscapes-dataset.com/examples/#fine-annotations))
- 20 000 annotated images with coarse annotations ([examples](https://www.cityscapes-dataset.com/examples/#coarse-annotations))

Metadata

- Preceding and trailing video frames. Each annotated image is the 20th image from a 30 frame video snippets (1.8s)
- Corresponding right stereo views
- GPS coordinates
- Ego-motion data from vehicle odometry
- Outside temperature from vehicle sensor

Benchmark suite and evaluation server

- Pixel-level semantic labeling
- Instance-level semantic labeling

----------------

#### 语义分割榜单

具体榜单地址：https://www.cityscapes-dataset.com/benchmarks/

可参看论文和代码：https://github.com/HRNet/HRNet-Semantic-Segmentation

| name                                             | IoU class | iIoU class | IoU category | iIoU category | code                                                         |
| :----------------------------------------------- | :-------- | :--------- | :----------- | :------------ | :----------------------------------------------------------- |
| iFLYTEK-CV （科大讯飞）                          | 83.6      | 64.7       | 92.1         | 82.3          | no                                                           |
| GALD-Net （北京大学）                            | 83.3      | 64.5       | 92.3         | 81.9          | [yes](https://github.com/lxtGH/GALD-Net)                     |
| HRNetV2 + OCR （微软）                           | 83.3      | 62.0       | 92.1         | 81.7          | [yes](https://github.com/HRNet)                              |
| NV-ADLR （英伟达）                               | 83.2      | 64.2       | 92.1         | 82.2          | no                                                           |
| GGCF                                             | 83.2      | 63.0       | 92.0         | 81.3          | no                                                           |
| GALD-net  （北京大学）                           | 83.1      | 63.5       | 92.2         | 81.4          | [yes](https://github.com/lxtGH/GALD_net)                     |
| Tencent AI Lab （腾讯）                          | 82.9      | 63.9       | 91.8         | 80.4          | no                                                           |
| DRN_CRL_Coarse （ICIP）                          | 82.8      | 61.1       | 91.8         | 80.7          | [yes](https://github.com/zhuangyqin/DRN.git)                 |
| NAVINFO_DLR                                      | 82.8      | 63.1       | 91.9         | 82.2          | no                                                           |
| Valeo DAR Germany                                | 82.8      | 62.9       | 92.0         | 82.2          | no                                                           |
| DPC                                              | 82.7      | 63.3       | 92.0         | 82.5          | [yes](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| SRC-B-MachineLearningLab (三星 )                 | 82.5      | 60.7       | 91.8         | 81.5          | no                                                           |
| RelationNet_Coarse                               | 82.4      | 61.9       | 91.8         | 81.4          | no                                                           |
| SSMA                                             | 82.3      | 62.3       | 91.5         | 81.7          | [yes](http://deepscene.cs.uni-freiburg.de/)                  |
| GFF-Net                                          | 82.3      | 62.1       | 92.0         | 81.4          | no                                                           |
| DDAR                                             | 82.2      | 62.7       | 91.9         | 81.5          | no                                                           |
| DeepLabv3+ （Google）                            | 82.1      | 62.4       | 92.0         | 81.9          | [yes](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| Auto-DeepLab-L                                   | 82.1      | 61.0       | 91.9         | 82.0          | [yes](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| Fast OCNet                                       | 82.1      | 61.0       | 91.7         | 80.7          | no                                                           |
| Mapillary Research: In-Place Activated BatchNorm | 82.0      | 65.9       | 91.2         | 81.7          | [yes](https://github.com/mapillary/inplace_abn)              |

---

#### 实例分割榜单

| name                                                         | AP   | AP 50% | AP 100m | AP 50m | code                                                  |
| :----------------------------------------------------------- | :--- | :----- | :------ | :----- | :---------------------------------------------------- |
| iFLYTEK-CV                                                   | 38.0 | 65.4   | 51.6    | 55.0   | no                                                    |
| Sogou_MM                                                     | 37.2 | 64.5   | 51.1    | 54.5   | no                                                    |
| PANet [COCO]                                                 | 36.4 | 63.1   | 49.2    | 51.8   | [yes](https://github.com/ShuLiu1993/PANet)            |
| NV-ADLR                                                      | 35.3 | 61.5   | 49.3    | 53.5   | no                                                    |
| UPSNet                                                       | 33.0 | 59.6   | 46.8    | 50.7   | [yes](https://github.com/uber-research/UPSNet)        |
| BshapeNet+ [COCO]                                            | 32.9 | 58.8   | 47.3    | 50.7   | no                                                    |
| TCnet                                                        | 32.6 | 59.0   | 45.0    | 47.8   | no                                                    |
| AdaptIS                                                      | 32.5 | 52.5   | 48.2    | 52.1   | no                                                    |
| RUSH_ROB                                                     | 32.1 | 55.5   | 45.2    | 46.3   | no                                                    |
| Mask R-CNN [COCO]                                            | 32.0 | 58.1   | 45.8    | 49.5   | no                                                    |
| PANet [fine-only]                                            | 31.8 | 57.1   | 44.2    | 46.0   | [yes](https://github.com/ShuLiu1993/PANet)            |
| SegNet                                                       | 29.5 | 55.6   | 43.2    | 45.8   | no                                                    |
| Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth | 27.7 | 50.9   | 37.8    | 37.3   | [yes](https://github.com/davyneven/SpatialEmbeddings) |
| GMIS: Graph Merge for Instance Segmentation                  | 27.6 | 44.6   | 42.7    | 47.9   | no                                                    |
| BshapeNet+ [fine-only]                                       | 27.3 | 50.4   | 40.5    | 43.1   | no                                                    |
| Mask R-CNN [fine-only]                                       | 26.2 | 49.9   | 37.6    | 40.1   | no                                                    |
| PolygonRNN++                                                 | 25.5 | 45.5   | 39.3    | 43.4   | [yes](http://www.cs.toronto.edu/polyrnn/)             |
| SGN                                                          | 25.0 | 44.9   | 38.9    | 44.5   | no                                                    |
| Deep Coloring                                                | 24.9 | 46.2   | 39.0    | 44.0   | no                                                    |
| NL_ROI_ROB                                                   | 24.0 | 45.8   | 36.1    | 40.8   | no                                                    |