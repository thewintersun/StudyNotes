## Roofline Model on NVIDIA GPUs

项目地址：https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020

This repo demonstrates the use of Roofline analysis on NVIDIA GPUs especially on V100s and architectures after V100s.
The Roofline performance model provides an intuitive and insightful way to understand application performance, identify bottlenecks and perform optimization for HPC applications.
For more details on Roofline, please visit [this page](https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/).

The methodology for Roofline data collection on NVIDIA GPUs has evolved from using nvprof ([github:nersc-roofline](https://github.com/cyanguwa/nersc-roofline)), to Nsight Compute in CUDA 10 ([tag:cuda10.2.89-ncu](https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tags/cuda10.2.89-ncu)) to Nsight Compute in CUDA 11 ([tag:cuda11.0.167-ncu](https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tags/cuda11.0.167-ncu)). A few papers have been published to validate the efficacy of the methodology:

- C. Yang, T. Kurth, and S. Williams, Hierarchical Roofline analysis for GPUs: Accelerating performance optimization for the NERSC‐9 Perlmutter system, Concurrency and Computation: Practice and Experience, e5547, 2019. https://doi.org/10.1002/cpe.5547 
- Y. Wang, C. Yang, Y. Zhang, T. Kurth and S. Williams, Roofline Performance Analysis of Operator Fusion in Deep Learning Applications, Cray User Group (CUG), 2020. (accepted)
- C. Yang, S. Williams, and Y. Wang, Roofline Performance Model for HPC and Deep-Learning Applications, GPU Technology Conference (GTC), 2020. https://developer.nvidia.com/gtc/2020/video/s21565 

The application in question in this repo is the GPP (General Plasmon Pole) kernel from the Sigma module in [BerkeleyGW](https://berkeleygw.org). This is a mini-app that abstracts a key part of the self-energy calculation in GW workflows in Material Science.

### Customized ncu-based Roofline Workflow

For tigher integration with users' other workflow, [/custom-scripts]() provides a set of scripts for manual metric collection and Roofline visualization.

- `run.gpp.customized`
-  `postprocess.py` and `roofline.py` 

The `run.gpp.customized` script uses GPP as an example to show a list of Nsight Compute metrics required for Roofline analaysis. These metrics are collected using the command-line utility of Nsight Compute `ncu` (or `nv-nsight-cu-cli`) and are written into a `.csv` file in `/custom-scripts`. The results are then post processed by `postprocess.py` using Pandas to calculate the Arithmetic Intensity (AI) and FLOP/s throughput for each kernel being profiled. When processing is done, `postprocess.py` will call `roofline.py` which is based on Matplotlib, to plot Roofline charts and then save the charts into `.png` files.

The data collection methodology used in these scripts is detailed below. It is new from Nsight Compute in CUDA 11 so please give it a try and let us know if there is any problem.

- Time: 
  - sm\_\_cycles_elapsed.avg / sm__cycles_elapsed.avg.per_second
  
- FLOPs: 
  -  `DP`: sm\_\_sass_thread_inst_executed_op_dadd_pred_on.sum + 2 x sm\_\_sass_thread_inst_executed_op_dfma_pred_on.sum + sm\_\_sass_thread_inst_executed_op_dmul_pred_on.sum
  -  `SP`: sm\_\_sass_thread_inst_executed_op_fadd_pred_on.sum + 2 x sm\_\_sass_thread_inst_executed_op_ffma_pred_on.sum + sm\_\_sass_thread_inst_executed_op_fmul_pred_on.sum
-  `HP`: sm\_\_sass_thread_inst_executed_op_hadd_pred_on.sum + 2 x sm\_\_sass_thread_inst_executed_op_hfma_pred_on.sum + sm\_\_sass_thread_inst_executed_op_hmul_pred_on.sum
  -  `Tensor Core`: 512 x sm__inst_executed_pipe_tensor.sum
  
- Bytes: 
  -  `DRAM`: dram__bytes.sum
  -  `L2`: lts__t_bytes.sum
-  `L1`: l1tex__t_bytes.sum

Also, please bear in mind that these scripts are written with GPP in mind, so please modify certain parameters as necessary, such as location of files, name of the output, kernels to profile, type of Roofline to use (e.g. HBM or hierarchical, and double/single/half precision), min/max of the axes of the chart, and the size/colors of the markers. Happy hacking :)