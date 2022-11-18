**卷积计算流程**

http://3ms.huawei.com/hi/group/3554771/wiki_5596624.html

摘要：CONV计算流程 CONV计算流程 本文以test_conv_cce.py文件中的test_FP16(bias)函数为切入点，主要剖析了conv_compute和conv_schedule流程，整个代码梳理过程也可以作为分析算子流程的参考，是一个完整的过程。 基本方法： 通过python llt/tensor_engine/ut/testcase_python/conv/test_conv_cce.py运行ut用例，并通过在代码中import pdb并调用pdb.set_trace()方法，进入断点，通过断点命中来查看函数运行的堆栈信息，进而知晓函数调用关系。

### CONV计算流程

本文以test_conv_cce.py文件中的test_FP16(bias)函数为切入点，主要剖析了conv_compute和conv_schedule流程，整个代码梳理过程也可以作为分析算子流程的参考，是一个完整的过程。

1. 基本方法：

   通过python llt/tensor_engine/ut/testcase_python/conv/test_conv_cce.py运行ut用例，并通过在代码中import pdb并调用pdb.set_trace()方法，进入断点，通过断点命中来查看函数运行的堆栈信息，进而知晓函数调用关系。

   ```python
   def test_FP16(bias):
       shape_in = (1, 64, 8, 8)
       shape_w = (128, 64, 1, 1)
       in_dtype = "float16"
       w_dtype = "float16"
       res_dtype = "float16"
       padh = 0
       padw = 0
       strideh = 1
       stridew = 1
   
       import pdb
       pdb.set_trace()
       topi.cce.conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                               padh, padw, strideh, stridew, bias=bias,
                               kernel_name="test23_TBE_FP1622", need_build=True, 
                               need_print=False)
   ```

    

2. 调用流程图：

![img](http://image.huawei.com/tiny-lts/v1/images/84d3a26395e913fc95dc_1023x615.png@900-0-90-f.png)



```python
te.lang.cce.con：#卷积计算描述
generic.auto_schedule: # 调用卷积模板
te.lang.cce.cce_build_code： # 生成cce代码
```

**conv计算**

代码路径：tensor_engine/python/te/lang/cce/te_compute/conv_compute.py

![img](http://image.huawei.com/tiny-lts/v1/images/5fb6826395ea75353731_1000x603.png@900-0-90-f.png)

![img](http://image.huawei.com/tiny-lts/v1/images/12c3326395eb58ce07fa_1056x788.png@900-0-90-f.png)



卷积操作实质上就是等效于通过im2col+GEMM实现，相关指令为Load3D和MAD，前者分为set_fmatrix和img2col_cbuf_to_ca，对应实现函数如上图所示。

实现过程中会把相关数据封装到conv_param中，以供schedule使用。

![img](http://image.huawei.com/tiny-lts/v1/images/c163e26395ec0beb1494_361x113.png@900-0-90-f.png)

**conv_schedule：**

代码路径：tensor_engine/python/te/lang/cce/te_schedule/conv_schedule.py

![img](http://image.huawei.com/tiny-lts/v1/images/2f6e026395ecae3ea5ba_576x379.png@900-0-90-f.png)

关键调用入口：tensor_engine/python/te/lang/cce/te_schedule/cce_schedule.py中的global_core_schedule函数：

![img](http://image.huawei.com/tiny-lts/v1/images/3465326395ed6a2857c4_1023x64.png@900-0-90-f.png)

```
AutoScheduleOp---analyze_data_dependence---tiling_fetch（关键步骤，获取到tiling）---double buffer---tile C---tile CUB---tile c_col---AL1 and BL1 slice---intrin mapping---cube schedule end
```

tiling_fetch函数调用tiling_query获取到tiling方案，然后再check tiling，如果错误，则使用默认的tiling方案，否则使用获取到的tiling方案进行schedule操作。tiling_query位于tensor_engine/python/te/domain/tiling/tiling_query.py

```python
params_encode = encode(params)
shape_encode = np.array([params_encode[0], params_encode[1], params_encode[2], params_encode[3],    params_encode[4], params_encode[5], params_encode[6], params_encode[7]], dtype="uint32")
shape_encode_array = tvm.nd.array(shape_encode)
fun = tvm.get_global_func("_tiling_query")
ret = fun(shape_encode_array,1)
res = list(ret.asnumpy())
tiling = decode(res)
```
