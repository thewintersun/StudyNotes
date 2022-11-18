### CrowdHuman

A Benchmark for Detecting Human in a Crowd

地址：http://www.crowdhuman.org/#page-top

下载地址：http://www.crowdhuman.org/download.html

论文地址：https://arxiv.org/abs/1805.00123

CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

![1565085254251](D:\Notes\raw_images\1565085254251.png)

![1565232769831](D:\Notes\raw_images\1565232769831.png)



We support `annotation_train.odgt` and `annotation_val.odgt` which contains the annotations of our dataset.

### What is odgt?

`odgt` is a file format that **each line of it is a JSON**, this JSON contains the whole annotations for the relative image. We prefer using this format since it is reader-friendly.

### Annotation format

```json
JSON{
    "ID" : image_filename,
    "gtboxes" : [gtbox], 
}

gtbox{
    "tag" : "person" or "mask", 
    "vbox": [x, y, w, h],
    "fbox": [x, y, w, h],
    "hbox": [x, y, w, h],
    "extra" : extra, 
    "head_attr" : head_attr, 
}

extra{
    "ignore": 0 or 1,
    "box_id": int,
    "occ": int,
}

head_attr{
    "ignore": 0 or 1,
    "unsure": int,
    "occ": int,
}
```

- `Keys` in `extra` and `head_attr` are **optional**, it means some of them may not exist
- `extra/head_attr` contains attributes for `person/head`
- `tag` is `mask` means that this box is `crowd/reflection/something like person/...` and need to be `ignore`(the `ignore` in `extra` is `1`)
- `vbox, fbox, hbox` means `visible box, full box, head box` respectively

![1565259240224](D:\Notes\raw_images\1565259240224.png)

![1565259271337](D:\Notes\raw_images\1565259271337.png)



### Objects365 Crowd Human Track Leaderboard

地址: 

https://www.objects365.org/crowd_human_track.html 

https://www.objects365.org/workshop2019.html

Challenge2019 in conjection with CVPR2019

| Rank | Team Name   | Institution                                                  | AP     |
| ---- | ----------- | ------------------------------------------------------------ | ------ |
| 1    | zack0704    | Tencent AI Lab                                               | 0.7746 |
| 2    | boke        | Sun Yat-Sen University                                       | 0.7525 |
| 3    | ZNuanyang   | Zhejiang University                                          | 0.7446 |
| 4    | HardenMVP   | ShanghaiTech University                                      | 0.7215 |
| 5    | balalamagic | Sichuan Yixun Information Technology Co.,Ltd.                | 0.7124 |
| 6    | sdu         | vsislab Shandong Universtiy                                  | 0.7101 |
| 7    | sail        | winsense                                                     | 0.7028 |
| 8    | NJUST       | PCALab Nanjing University of Science and Technology          | 0.7018 |
| 9    |             | Individual                                                   | 0.6878 |
| 10   | x           | complete Xilinx                                              | 0.6822 |
| 11   | liuiyang    | Beijing Tongfang Software                                    | 0.6798 |
| 12   | Iron        | Man University of Chinese Academy of Sciences & Huawei       | 0.6683 |
| 13   | py          | hitsz Harbin Institute of Technology                         | 0.6566 |
| 14   | gyc1036     | Nanjing University of Aeronautics and Astronautics           | 0.6522 |
| 15   | shenqian    | Huazhong University of Science and Technology                | 0.6367 |
| 16   | TJU         | VILAB Tianjin University                                     | 0.5412 |
| 17   | wxxcn       | Southeast University                                         | 0.5045 |
| 18   | zuolong     | l Guangdong University of Technology                         | 0.4923 |
| 19   | dereyly     | NtechLab                                                     | 0.4609 |
| 20   | weidi1024   | Xidian University                                            | 0.4402 |
| 21   | black       | box South China Agricultural University                      | 0.3764 |
| 22   | shenty      | Advanced Institute of Information Technology, Peking University | 0.3646 |
| 23   | wxy         | Anhui University of Technology                               | 0.0581 |