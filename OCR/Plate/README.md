
Automatic License Plate Recognition （ALPR）

<h2>商业车牌</h2>

<b>Face++</b>
https://www.faceplusplus.com.cn/license-plate-recognition/ 
检测图片中的车牌，并返回车牌边框坐标、车牌号码、车牌颜色等信息。支持识别各种位置、白天以及夜间的车牌识别。
- 精准识别
- 适应不同光照场景
- 适应倾斜/模糊等场景

<b>openalpr</b>
https://www.openalpr.com/index.html
国外车牌识别，数量少，但有GroundTruth
https://www.openalpr.com/benchmarks.html 
https://www.openalpr.com/benchmarks/opensource_us.html
https://www.openalpr.com/benchmarks/commercial_us.html

<b>翔云OCR服务</b>
https://www.netocr.com/


<h2>BenchMark</h2>

暂无

<h2>车牌识别存在的问题</h2>

检测存在的问题：
- 光线，白天与黑夜。
- 角度。
- 不同尺度，远近问题。
- 模糊。
- 遮挡。
- 检测不完整，截断字符。 

识别问题：
- 模糊
- 首尾断字
- 字符重叠识别

中国车牌类别：
- 单行蓝牌
- 单行黄牌
- 新能源车牌
- 白色警用车牌
- 使馆/港澳车牌
- 教练车牌
- 武警车牌
- 民航车牌
- 双层黄牌
- 双层武警
- 双层军牌
- 双层农用车牌
- 双层个性化车牌

中文列表：

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
