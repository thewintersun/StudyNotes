## Conv2d 算子实现 TIK方式

本样例使用TIK API进行了Conv2d算子的实现，将输入的张量和一个权重张量执行卷积2-D操作，输出结果张量。

### 算子分析  

使用TIK API开发Conv2d算子前，我们需要确定算子功能、输入、输出，算子开发方式、算子类型以及算子实现函数名称等。

1. 明确算子的功能。

   Conv2d算子计算过程是：将输入的张量和一个权重张量执行卷积2-D操作，输出结果张量，如图1

   ![img](https://support.huaweicloud.com/odevg-Inference-cann/figure/zh-cn_image_0274896054.png)

   

2. 明确输入和输出。

   - Conv2d算子有2个输入x和filter，1个输出y，3个属性。
   - 算子输入的数据类型为float16，算子输出的数据类型为float16。
   - 算子输入支持固定shape，输出shape与输入shape需要满足算子的数学表达式。
   - 算子输入支持的format：NCHW。
   - 算子的三个属性为strides，pads，dilations，属性值分别为[1,1,1,1]。

3. 确定算子开发方式及使用的计算接口。

   1. 计算过程主要涉及到二维卷积运算，初步分析可使用conv2d()接口实现二维卷积运算功能。

      conv2d接口在处理strides和dilations两个属性时，在NC两个维度的值必须设定为1，同时当前样例中HW两个维度上的值指定为1。

   2. 由于在整个conv2d计算过程中会涉及到数据搬运操作，可使用data_move()接口实现从Global Memory搬运数据到L1 Buffer中。

   3. 计算完成后，可使用fixpipe()接口把数据从L1OUT Buffer搬运数据到Global Memory中。

4. 明确算子实现文件名称、算子实现函数名称以及算子的类型（OpType）。

   - 算子类型需要采用大驼峰的命名方式，即采用大写字符区分不同的语义。

   - 算子文件名称和算子函数名称，可选用以下任意一种命名规则：

     - 用户自定义，此时需要在[算子信息定义](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0101.html)中配置**opFile.value**与**opInterface.value**。

     - 不配置算子信息定义中的opFile.value与opInterface.value，FE会将OpType按照如下方式进行转换后进行算子文件名和算子函数名的匹配。

       转换规则如下：

       - 首字符的大写字符转换为小写字符。

         例如：Abc -> abc

       - 小写字符后的大写字符转换为下划线+小写字符。

         例如：AbcDef -> abc_def

       - 紧跟数字以及大写字符后的大写字符，作为同一语义字符串，查找此字符串后的第一个小写字符，并将此小写字符的前一个大写字符转换为下划线+小写字符，其余大写字符转换为小写字符。若此字符串后不存在小写字符，则直接将此字符串中的大写字符转换为小写字符。

         例如：ABCDef -> abc_def；Abc2DEf -> abc2d_ef；Abc2DEF -> abc2def；ABC2dEF -> abc2d_ef。

   本样例中，为不影响内置的Conv2D算子，算子类型定义为Conv2DTik；算子的实现文件名称及实现函数名称定义为conv2d_tik。

   通过以上分析，得到Conv2DTik算子的设计规格如下：

   | 算子类型（OpType）        | Conv2DTik                    |                        |                    |              |                  |
   | ------------------------- | ---------------------------- | ---------------------- | ------------------ | ------------ | ---------------- |
   | 算子输入                  | name：x                      | shape：（8,512,7,7）   | data type：float16 | format：NCHW | -                |
   |                           | name：filter                 | shape：（512,512,3,3） | data type：float16 | format：NCHW | -                |
   | 算子属性                  | name：strides                | -                      | data type：listInt | -            | value：[1,1,1,1] |
   |                           | name：pads                   | -                      | data type：listInt | -            | value：[1,1,1,1] |
   |                           | name：dilations              | -                      | data type：listInt | -            | value：[1,1,1,1] |
   | 算子输出                  | name：y                      | shape：（8,512,7,7）   | data type：float16 | format：NCHW | -                |
   | 算子实现使用主要TIK接口   | data_move()conv2d()fixpipe() |                        |                    |              |                  |
   | 算子实现文件/实现函数名称 | conv2d_tik                   |                        |                    |              |                  |

#### 算子代码实现

样例中Conv2DTik算子接收的数据类型为"float16"，首先需要对==算子类型进行校验==，然后==对参数进行设置==，并调用算子计算函数。

```python
def conv2d_tik(inputs, weights, outputs, strides, pads, dilations, kernel_name="conv2d_tik"):
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    res_dtype = outputs.get("dtype")
    in_shape = inputs.get("shape")
    wori_shape = weights.get("ori_shape")

    if len(strides) != 4:
        raise RuntimeError("strides shape should be 4d.")
    if len(dilations) != 4:
        raise RuntimeError("dilations shape should be 4d.")
    if len(pads) != 4:
        raise RuntimeError("pads shape should be 4d.")
    if in_dtype!="float16" or w_dtype!="float16" or res_dtype!="float16":
        raise RuntimeError("dtype shape should be float16.")
    if weights.get("ori_format")!="NCHW":
        raise RuntimeError("format should be NCHW.")
    loc_dtype = "float32"
    quantize_params = {"mode":"fp322fp16", "mode_param":None}
    strideList = [strides[2], strides[3]]
    dilationList = [dilations[2], dilations[3]]
    # NCHW => NC1HWC0
    w_shape = [wori_shape[1]//16, wori_shape[2], wori_shape[3], wori_shape[0], 16]

    params = {
        "fm_shape": in_shape,
        "weight_shape": w_shape,
        "fm_dtype": in_dtype,
        "weight_type": w_dtype,
        "dst_l0c_type": loc_dtype,
        "dst_gm_type": res_dtype,
        "quantize_params": quantize_params,
        "pad_list": pads,
        "pad_value": 0,
        "stride_list": strideList,
        "dilation_list": dilationList,
        "cout_split_factor": 64,
        "kernel_name": kernel_name}

    conv2d_tik_compute(params)
```

算子计算函数的实现逻辑如下所示。

根据参数对输入和输出tensor进行==shape计算和占位==。

```python
def conv2d_tik_compute(params):
    tik_instance = tik.Tik()
    te_set_l2_mode(1)
    n, c1, h, w, c0 = params["fm_shape"]
    c1, kh, kw, cout, c0 = params["weight_shape"]
    stride_h, stride_w = params["stride_list"]
    dilation_h, dilation_w = params["dilation_list"]
    pad_top, pad_bot, pad_left, pad_right = params["pad_list"]
    kh_dilation = (kh - 1) * dilation_h + 1
    kw_dilation = (kw - 1) * dilation_w + 1
    ho = int(np.ceil((h + pad_top + pad_bot - kh_dilation + 1) / stride_h)) 
    wo = int(np.ceil((w + pad_right + pad_left - kw_dilation + 1) / stride_w))
    round_howo = ceil_div(ho * wo, 16) * 16 
    fm_gm = tik_instance.Tensor(params['fm_dtype'], (n, c1, h, w, c0),
                                name='fm_gm', scope=tik.scope_gm)
    weight_gm = tik_instance.Tensor(params['weight_type'],
                                    (c1, kh, kw, cout, c0), name='weight_gm',
                                    scope=tik.scope_gm)
    dst_gm = tik_instance.Tensor(params['dst_gm_type'],
                                 [n, cout // 16, ho, wo, 16],
                                 name='dst_gm', scope=tik.scope_gm)
    core_num = 2 #AScend 310
    pre_core_cout = cout // core_num
    cout_iter_num = pre_core_cout // params["cout_split_factor"]
    Cin_blocks = c1
```

通过for_range( )循环==开启double buffer和多核==，对==输入数据进行切分==，实现卷积的高效运算。

```python
    with tik_instance.for_range(0, core_num, block_num=core_num) as cout_o:
        with tik_instance.for_range(0, cout_iter_num, thread_num=1) as cout_i:
            weight_L1 = tik_instance.Tensor(
                params['weight_type'], 
                (Cin_blocks, kh, kw, params["cout_split_factor"], c0),
                name='weight_l1', 
                scope=tik.scope_cbuf)
            tik_instance.data_move(
                weight_L1,
                weight_gm.flatten()[cout_o * pre_core_cout * c0 +
                                    params["cout_split_factor"] * cout_i * c0],
                0, Cin_blocks * kh * kw, params["cout_split_factor"],
                (cout - params["cout_split_factor"]), 0)

            with tik_instance.for_range(0, n, thread_num=2) as n_index:
                feature_map_l1 = tik_instance.Tensor(params['fm_dtype'],
                                                     (c1, h, w, c0),
                                                     name='feature_map_l1',
                                                     scope=tik.scope_cbuf)
                tik_instance.data_move(feature_map_l1,
                                       fm_gm[n_index, :, :, :, :],
                                       0, 1, c1 * h * w, 0, 0)
                dst_l0c = tik_instance.Tensor(
                                    params['dst_l0c_type'], 
                                    [params["cout_split_factor"]//16, round_howo, 16],
                                    name='dst_l0c', scope=tik.scope_cbuf_out)
```

调用conv2d()实现二维卷积计算。

```python
                tik_instance.conv2d(dst_l0c, feature_map_l1,
                                    weight_L1, (c1, h, w, c0),
                                    (Cin_blocks, kh, kw,
                                    params["cout_split_factor"], c0),
                                    params['stride_list'],
                                    [pad_left, pad_right, pad_top, pad_bot],
                                    params['dilation_list'],
                                    params['pad_value'])
```

调用fixpipe()实现计算结果数据的搬运。

```python
                tik_instance.fixpipe(
                    dst_gm[n_index, (cout_o*pre_core_cout +
                                     params["cout_split_factor"]*cout_i) //
                           (32//DTYPE_SIZE[params['dst_gm_type']]), 0, 0, 0],
                    dst_l0c, params["cout_split_factor"]//16,
                    ho * wo * 16 * DTYPE_SIZE[params['dst_l0c_type']] // 32, 0, 0,
                    extend_params={"bias": None,
                                   "quantize_params": params["quantize_params"]})
```

调用BuildCCE()进行编译。

```python
    tik_instance.BuildCCE(kernel_name=params["kernel_name"],
                          inputs=[fm_gm, weight_gm], outputs=[dst_gm])
```

#### 算子适配插件实现

开发者需要自定义实现ParseParamsConv2D函数，实现原始Caffe中Type为ConvolutionTik算子到适配昇腾AI处理器的Conv2DTik算子的属性映射。

ParseParamsConv2D函数的实现如下所示：

```python
// Get covolution pad params from caffe proto and convert to tbe conv2d ir
// pad flag [pads]
static bool SetPads(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const int kDefaultPad = 0;
    int64_t pad[2] = {kDefaultPad, kDefaultPad};
    std::vector<int64_t> pad_attr;
    int pad_h;
    int pad_w;
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(PAD, pad_attr)){
        return false;
    }
    const int pSize = pad_attr.size();
    if (op_src.GetAttr(PAD_H, pad_h) || op_src.GetAttr(PAD, pad_w)){
        if (pSize != 0) {
            return false;
        }
        pad[0] = pad_h;
        pad[1] = pad_w;
    }else{
        if (pSize == 1 || pSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (pSize == 1) ? 0 : i;
                pad[i] = pad_attr[index];
            }
        } else if (pSize != 0) {
            return false;
        }
    }
    std::vector<int64_t> pList;
    pList.push_back(pad[0]);
    pList.push_back(pad[0]);
    pList.push_back(pad[1]);
    pList.push_back(pad[1]);
    op_dest.SetAttr(PADS, (pList));

}
// Get covolution stride params from caffe proto and convert to tbe conv2d
// ir [strides]
static bool SetStrides(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const int kDefaultStride = 1;
    int64_t stride[2] = {kDefaultStride, kDefaultStride};
    std::vector<int64_t> stride_attr;
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(STRIDE, stride_attr)){
        return false;
    }
    const int sSize= stride_attr.size();
    int stride_h;
    int stride_w;
    if (op_src.GetAttr(STRIDE_H, stride_h) || op_src.GetAttr(STRIDE_W, stride_w)){
        if (sSize != 0) {
            return false;
        }
        stride[0] = stride_h;
        stride[1] = stride_w;
    }else {
        if (sSize == 1 || sSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (sSize == 1) ? 0 : i;
                stride[i] = stride_attr[index];
            }
        } else if (sSize != 0) {
            return false;
        }
    }
    std::vector<int64_t> sList;
    sList.push_back(1);
    sList.push_back(1);
    sList.push_back(stride[0]);
    sList.push_back(stride[1]);
    op_dest.SetAttr(STRIDES, (sList));

    return true;
}

// Get covolution dilation params from caffe proto and convert to tbe conv2d
// ir [dilations]
static bool SetDilations(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const int kDefaultDilation = 1;
    std::vector<int64_t> dilation_attr;
    int64_t dilation[2] = {kDefaultDilation, kDefaultDilation};
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(DILATION, dilation_attr)){
        return false;
    }
    const int dSize = dilation_attr.size();
    if (dSize == 1 || dSize == 2) {
        for (size_t i = 0; i < 2; i++) {
            int index = (dSize == 1) ? 0 : i;
            dilation[i] = dilation_attr[index];
        }
    } else if (dSize != 0) {
        return false;
    }
    std::vector<int64_t> dList;
    dList.push_back(1);
    dList.push_back(1);
    dList.push_back(dilation[0]);
    dList.push_back(dilation[1]);
    op_dest.SetAttr(DILATIONS, (dList));

    return true;

}
// Check input parameters that are illegal or not applicable to 2D convolution
static bool ProcSpecParams(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int num_output;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(NUM_OUTPUT, num_output)){
        if (num_output < 1) {
            return false;
        }
    }
    int group;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(GROUP, group)){
        if (group < 1 || num_output % group != 0) {
            return false;
        }
    }
    op_dest.SetAttr(GROUP, (int64_t)group);

    vector<int64_t> kernel_size;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(KERNEL_SIZE, kernel_size)){
        return false;
    }
    int kSize = kernel_size.size();
    int kernel[2] = {0, 0};
    int kernel_h;
    int kernel_w;
    if (op_src.GetAttr(KERNEL_H, kernel_h) || op_src.GetAttr(KERNEL_W, kernel_w)){
        if (kSize != 0) {
            return false;
        }
        kernel[0] = kernel_h;
        kernel[1] = kernel_w;
    }else{
        if (kSize == 1 || kSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (kSize == 1) ? 0 : i;
                kernel[i] = kernel_size[index];
            }
        } else {
            return false;
        }
    }
    for (size_t i = 0; i < 2; i++) {
        if (kernel[i] < 1) {
            return false;
        }
    }
    int channel_axis;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(AXiS, channel_axis)){
        if ((channel_axis + 4) % 4 != 1) {
            return false;
       }
    }
    bool force_nd_im2col;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(FORCE_ND_IM2COL, force_nd_im2col)){
        if (force_nd_im2col) {
            return false;
        }
    }
    return true;
}

// Replace GE ParseParams function to process graph conv2d node attrs
Status ParseParamsConv2D(const ge::Operator& op_src, ge::Operator& op_dest)
{

    if (!(ProcSpecParams(op_src, op_dest) && SetPads(op_src, op_dest) &&
          SetStrides(op_src, op_dest) && SetDilations(op_src, op_dest))) {
        return FAILED;
    }

    return SUCCESS;
}
```

#### 算子原型定义

conv2d_tik.h对Conv2DTik算子进行原型定义。

```python
REG_OP(Conv2DTik)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DTik)
}
```

原型定义的关键点是推理输出Tensor的shape及dtype，如下所示，conv2d_tik.cpp推理算子输出tensor的shape并对基本参数进行校验。

```python
static bool GetPadConv2D(ge::Operator& op,
                         int32_t ih, int32_t iw,
                         int32_t kh, int32_t kw,
                         int32_t strh, int32_t strw,
                         int32_t dilh, int32_t dilw,
                         int32_t& padt, int32_t& padb,
                         int32_t& padl, int32_t& padr) {
    std::string padStr;
    std::vector<int32_t> padList;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)){
        if (padStr.compare("SAME") == 0){
            int32_t tails_h = ih % strh;
            int32_t tails_w = iw % strw;
            int32_t dkh = dilh*(kh - 1) + 1;
            int32_t dkw = dilw*(kw - 1) + 1;
            int32_t pad_h = \
                    std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
            int32_t pad_w = \
                    std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_w / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
        } else if (padStr.compare("VALID") == 0) {
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
        } else {
            return false;
        }
        op.SetAttr("pads", padList);
    }
    std::vector<int32_t> padVec;
    op.GetAttr("pads", padVec);
    auto pSize = padVec.size();
    if (pSize != 4) {
        return false;
    }
    padt = padVec[0];
    padb = padVec[1];
    padl = padVec[2];
    padr = padVec[3];
    if (padt < 0 || padb < 0 || padl < 0 || padr < 0) {
        return false;
    }

    return true;
}

/*
 * Get 2D(H/W) stride and dilation params to infershape output
 *   [strides]: 4D list, format sensitive, according to first input
 *              tensor format
 *   [dilations]: 4D list, format sensitive
*/
static bool GetAttrsConv2D(ge::Operator& op, Format refer,
                           int32_t& strh, int32_t& strw,
                           int32_t& dilh, int32_t& dilw) {
    std::vector<int32_t> strideList;
    op.GetAttr("strides", strideList);
    auto sSize = strideList.size();
    if (sSize != 4) {
        return false;
    }
    std::vector<int32_t> dilationList;
    op.GetAttr("dilations", dilationList);
    auto dSize = dilationList.size();
    if (dSize != 4) {
        return false;
    }

    if (refer == FORMAT_NCHW) {
        strh = strideList[2];
        strw = strideList[3];
        dilh = dilationList[2];
        dilw = dilationList[3];
    } else if (refer == FORMAT_NHWC) {
        strh = strideList[1];
        strw = strideList[2];
        dilh = dilationList[1];
        dilw = dilationList[2];
    }
    if (strh <= 0 || strw <= 0) {
        return false;
    }
    if (dilh <= 0 || dilw <= 0) {
        return false;
    }

    return true;
}

/*
* Infer output shape and dtype, dtype is same to first input tensor
* Output format is set by ge parser process already
*/
IMPLEMT_INFERFUNC(Conv2DTik, Conv2DInfer) {

    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    auto xFormat = xTensor.GetFormat();
    auto wFormat  = wTensor.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(wFormat);

    int32_t in = 0;
    int32_t ic = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    if (xFormat == FORMAT_NCHW) {
        in = xShape[0];
        ic = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else if (xFormat == FORMAT_NHWC) {
        in = xShape[0];
        ic = xShape[3];
        ih = xShape[1];
        iw = xShape[2];
    } else {
        return GRAPH_FAILED;
    }

    if (wFormat == FORMAT_NCHW) {
        kn = wShape[0];
        kc = wShape[1];
        kh = wShape[2];
        kw = wShape[3];
    } else if (wFormat == FORMAT_NHWC) {
        kn = wShape[0];
        kc = wShape[3];
        kh = wShape[1];
        kw = wShape[2];
    } else if (wFormat == FORMAT_HWCN) {
        kn = wShape[3];
        kc = wShape[2];
        kh = wShape[0];
        kw = wShape[1];
    } else {
        return GRAPH_FAILED;
    }

    int64_t groups = 1;

    if (ic != kc*groups) {
        return GRAPH_FAILED;
    }

    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (false == GetAttrsConv2D(op, xFormat, strh, strw, dilh, dilw)) {
        return GRAPH_FAILED;
    }
    if (false == GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw,
                              padt, padb, padl, padr)) {
        return GRAPH_FAILED;
    }

    int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
    int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

    vector<int64_t> yShape;
    auto yTensor = op.get_output_desc_y();
    auto yFormat = yTensor.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(in);
        yShape.push_back(kn);
        yShape.push_back(oh);
        yShape.push_back(ow);
    } else if (yFormat == FORMAT_NHWC) {
        yShape.push_back(in);
        yShape.push_back(oh);
        yShape.push_back(ow);
        yShape.push_back(kn);
    } else {
        return GRAPH_FAILED;
    }
    yTensor.SetShape(Shape(yShape));
    auto xDtype = xTensor.GetDataType();
    if (xDtype == ge::DT_INT8){
        yTensor.SetDataType(ge::DT_INT32);
    }else{
        yTensor.SetDataType(xDtype);
    }
    if (GRAPH_SUCCESS != op.update_output_desc_y(yTensor)) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
```

#### 算子信息定义

Conv2DTik算子的信息定义文件请参见“tbe/op_info_cfg/ai_core/*{soc_version}*/conv2d_tik.ini”。

