### Caltech Pedestrian Detection Benchmark

地址:http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

![1565168166918](C:\Users\j00496872\Desktop\Notes\raw_images\1565168166918.png)

格式转换方法：

将caltech数据集转换成VOC格式

https://blog.csdn.net/Michelexie/article/details/83957652 

## Description

The Caltech Pedestrian Dataset consists of approximately 10 hours of 640x480 30Hz video taken from a vehicle driving through regular traffic in an urban environment. About 250,000 frames (in 137 approximately minute long segments) with a total of 350,000 bounding boxes and 2300 unique pedestrians were annotated. The annotation includes temporal correspondence between bounding boxes and detailed occlusion labels. More information can be found in our [PAMI 2012](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf) and [CVPR 2009](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/CVPR09pedestrians.pdf)benchmarking papers.

## Download

- **Caltech Pedestrian Dataset**. The training data (set00-set05) consists of six training sets (~1GB each), each with 6-13 one-minute long seq files, along with all annotation information (see the paper for details). The testing data (set06-set10) consists of five sets, again ~1GB each. New: annotations for the entire dataset are now also provided. Output files containing detection results for all evaluated algorithms are also available.
- **Seq video format**. An seq file is a series of concatenated image frames with a fixed size header. Matlab routines for reading/writing/manipulating seq files can be found in [Piotr's Matlab Toolbox](https://pdollar.github.io/toolbox/index.html)(version 3.20 or later required). These routines can also be used to extract an seq file to a directory of images.
- **Matlab evaluation/labeling code (3.2.1)**. The annotations use a custom "video bounding box" (vbb) file format. The code also contains utilities to view seq files with annotations overlaid, evaluation routines used to generate all the ROC plots in the paper, and also the vbb labeling tool used to create the dataset (see also this somewhat outdated [video tutorial](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/vbbLabelerTutorial-divx.avi)).
- **Additional datasets in standardized format**. For convenience we are posting full images/annotations in seq/vbb format as well as detection results for all evaluated algorithms on a number of additional datasets. This facilitates training/testing on these additional datasets and exact reproduction of all ROC curves. Full copyright remains with the original authors, please see the respective website for additional information including how to cite evaluation results on these datasets. [INRIA pedestrian dataset ](http://pascal.inrialpes.fr/data/human/)[[converted](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/INRIA/)], [ETH pedestrian dataset](http://www.vision.ee.ethz.ch/~aess/dataset/) [[converted](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/ETH/)], [TUD-Brussels pedestrian dataset](http://www.d2.mpi-inf.mpg.de/tud-brussels) [[converted](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/TudBrussels/)], [Daimler pedestrian dataset](http://www.gavrila.net/Research/Pedestrian_Detection/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Detection_Be/daimler_mono_ped__detection_be.html) [[converted](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/Daimler/)].

## Benchmark Results

### [Algorithm Details and References](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/algorithms.pdf) | [Algorithm Runtime vs. Performance](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/timing.pdf)

For details on the evaluation scheme please see our [PAMI 2012](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf) paper. 
**Note:** The evaluation scheme has evolved since our [CVPR 2009](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/CVPR09pedestrians.pdf) paper. 
**Note:** We render at most 15 top results per plot (but always include the VJ and HOG baselines).

1. **Caltech Pedestrian Testing Dataset**: We give two set of results: on 50-pixel or taller, unoccluded or partially occluded pedestrians ([reasonable](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/UsaTestRocReasonable.pdf)), and a more detailed breakdown of performance as in the paper ([detailed](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/UsaTestRocs.pdf)).
2. **Caltech Pedestrian Training Dataset**: Results on the Caltech training data: [reasonable](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/UsaTrainRocReasonable.pdf), [detailed](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/UsaTrainRocs.pdf).
3. **Caltech Pedestrian Japan Dataset**: Similar to the Caltech Pedestrian Dataset (both in magnitude and annotation), except video was collected in Japan. We cannot release this data, however, we will benchmark results to give a secondary evaluation of various detectors. Results: [reasonable](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/JapanRocReasonable.pdf), [detailed](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/JapanRocs.pdf).
4. **INRIA Pedestrian Test Dataset**: [Full image results](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/InriaTestRocReasonable.pdf) on the INRIA Pedestrian dataset ([evaluation details](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/INRIA/readme.txt)).
5. **ETH Pedestrian Dataset**: [Results](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/ETHRocReasonable.pdf) on the ETH Pedestrian dataset ([evaluation details](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/ETH/readme.txt)).
6. **TUD-Brussels Pedestrian Dataset**: [Results](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/TudBrusselsRocReasonable.pdf) on the TUD-Brussels Pedestrian dataset ([evaluation details](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/TudBrussels/readme.txt)).
7. **Daimler Pedestrian Dataset**: [Results](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/rocs/DaimlerRocReasonable.pdf) on the Daimler Pedestrian dataset ([evaluation details](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/Daimler/readme.txt)).

## Submitting Results

Please contact us to include your detector results on this site. We perform the evaluation on every 30th frame, starting with the 30th frame. For each video, the results for each frame should be a text file, with naming as follows: "I00029.txt, I00059.txt, ...". Each text file should contain 1 row per detected bounding box, in the format "[left, top, width, height, score]". If no detections are found the text file should be empty (but must still be present). The directory structure should mimic the directory structure containing the videos: "set00/V000, set00/V001...". Please see the output files for the evaluated algorithms (available in the download section) if the above description is unclear. Note that during evaluation all detections for a given video are concatenated into a single text file, thus avoiding having tens of thousands of text files per detector (see provided detector files for details).

## Related Datasets

Below we list other pedestrian datasets, roughly in order of relevance and similarity to the Caltech Pedestrian dataset. A more detailed comparison of the datasets (except the first two) can be found in the paper.

- [GM-ATCI](https://sites.google.com/site/rearviewpeds1/): Rear-View Pedestrians Dataset captured from a fisheye-lens camera.
- [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html): Also captured in an urban setting, update of the older DaimlerChrysler dataset. Contains tracking information and a large number of labeled bounding boxes.
- [NICTA](http://www.nicta.com.au/category/research/computer-vision/tools/automap-datasets/): A large scale urban dataset collected in multiple cities/countries. No motion/tracking information, but significant number of unique pedestrians.
- [ETH](http://www.vision.ee.ethz.ch/~aess/dataset/): Urban dataset captured from a stereo rig mounted on a stroller.
- [TUD-Brussels](http://www.d2.mpi-inf.mpg.de/tud-brussels): Dataset with image pairs recorded in an crowded urban setting with an onboard camera.
- [INRIA](http://pascal.inrialpes.fr/data/human/): Currently one of the most popular static pedestrian detection datasets.
- [PASCAL](http://pascallin.ecs.soton.ac.uk/challenges/VOC/databases.html): Static object dataset with diverse object views and poses.
- [CVC-ADAS](http://www.cvc.uab.es/adas/site/?q=node/7): collection of pedestrian datasets including pedestrian videos acquired on-board, virtual-world pedestrians (with part annotations), and occluded pedestrians.
- [USC](http://iris.usc.edu/Vision-Users/OldUsers/bowu/DatasetWebpage/dataset.html): A number of fairly small pedestrian datasets taken largely from surveillance video.
- [MIT](http://cbcl.mit.edu/software-datasets/PedestrianData.html): One of the first pedestrian datasets, fairly small and relatively well solved at this point.