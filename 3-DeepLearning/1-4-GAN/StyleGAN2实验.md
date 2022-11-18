## StyleGAN2 实验总结

#### 代码运行问题：

代码地址: https://github.com/NVlabs/stylegan2

问题：This file requires compiler and library support for the ISO C++ 2011 standard

https://stackoverflow.com/questions/59342888/tensorflow-error-this-file-requires-compiler-and-library-support-for-the-iso-c

问题：Status: CUDA driver version is insufficient for CUDA runtime version

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

问题： 重新加载NVIDIA Driver 而不用重启机器。

Failed to initialize NVML: Driver/library version mismatch

https://comzyh.com/blog/archives/967/

问题：undefined symbol: _ZN10tensorflow12OpDefBuilder6OutputESs

https://blog.csdn.net/zaf0516/article/details/103618601



#### 网络结构：

#### ![1581390679472](D:\Notes\raw_images\1581390679472.png)

![1581390733279](D:\Notes\raw_images\1581390733279.png) 

D网络：

   ![1581390570703](D:\Notes\raw_images\1581390570703.png)



#### 实验结果

```bash
# 生成人脸
python run_generator.py generate-images --network=./results/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5 
```

教堂训练

```bash
python run_training.py --num-gpus=8 --data-dir=./datasets --config=config-f --dataset=church --total-kimg 88000 --gamma=100 
```

![1582008743301](D:\Notes\raw_images\1582008743301.png)

真实车牌训练

```bash
python run_training.py --num-gpus=8 --data-dir=./datasets --config=config-f --dataset=france_plate --total-kimg 19863 --gamma=100 
```

![1582008812787](D:\Notes\raw_images\1582008812787.png)

风格混合 - 人脸

```bash
 python run_generator.py style-mixing-example --network=./results/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
```

![1582008458652](D:\Notes\raw_images\1582008458652.png)

风格混合 - 车牌

```bash
python run_generator.py style-mixing-example --network=./results/network-snapshot-004032.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
```

![1582008521243](D:\Notes\raw_images\1582008521243.png)

### 总结与结论

1. StyleGAN 的生成，输入均为随机数字，输出亦是随机。生成数据无标签可用。
2. 输出受输入影响较大：比如教堂输出，会有水印，会有分格等，由输入图像的不正规导致。教堂生成有不合理地方：直线不值，部分区域不清等问题。
3. 车牌生成问题：在双行车牌上生成效果不佳，或因数据过少导致，部分车牌甚至生成错误。车牌风格混合无法控制，结果无法预测。