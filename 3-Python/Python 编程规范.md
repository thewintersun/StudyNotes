# 华为Python语言通用编程规范 V2.1

华为技术有限公司 版权所有 侵权必究

http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md

# 目录

| **章节**                                                     | **内容**                                                     |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [0 前言](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#前言) | [背景](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#背景) [使用对象](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#使用对象) [适用范围](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#适用范围) [术语定义](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#术语定义) |
| [1 排版](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#排版) | [缩进](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C1_1) [语句](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C1_2) [空格](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C1_3) [导入](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C1_4) [解释器](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C1_5) |
| [2 注释](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#注释) | [类、接口和函数](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C2_1) [属性](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C2_2) [格式](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C2_3) |
| [3 命名](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#命名) | [包和模块](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C3_1) [类](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C3_2) [函数](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C3_3) [变量](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C3_4) [命名规范推荐表](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C3_5) |
| [4 编码](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#编码) | [编码](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#编码) |
| [5 异常](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#异常) | [异常处理](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C5_1) [断言](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C5_2) |
| [6 并发与并行](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#并发与并行) | [线程](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#C6_1) |
| [7 性能](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#性能) | [性能](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#性能) |
| [8 编程实践](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#编程实践) | [编程实践](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#编程实践) |
| [附录](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#附录) | [参考资料](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#参考资料) [贡献者](http://rnd-isourceb.huawei.com/coding-guide/python-coding-style-guide/tree/master/huawei-python-coding-style-guide.md#贡献者) |

# 前言

## 背景

《Python语言编程规范》针对Python语言编程中代码风格要求，编码、异常处理、并发与并行、性能、其他等方面，描述可能导致程序错误、低效、难以理解维护的常见编程问题。

## 使用对象

本规范的读者及使用对象主要为使用Python语言的研发人员和测试人员等。

## 适用范围

该规范适用于基于Python语言的产品开发。
除非有特别说明，所有的代码示例都是基于3.6及更高版本。

## 术语定义

**规则：** 编程时必须遵守的约定

**说明：** 某个规则的具体解释

**建议：** 编程时必须加以考虑的约定

**错误示例：** 违背某条规则的例子

**正确示例：** 遵循某条规则的例子

**例外情况：** 相应的规则不适用的场景

**信任边界：** 位于信任边界之内的所有组件都是被系统本身直接控制的。所有来自不受控的外部系统的连接与数据，包括客户端与第三方系统，都应该被认为是不可信的，要先在边界处对其校验，才能允许它们进一步与本系统交互。

**非信任代码：** 非产品包中的代码，如通过网络下载到本地虚拟机中加载并执行的代码。

# 1. 排版

## 缩进

## 规则1.1 程序块采用4个空格缩进风格编写

**说明：**程序块采用缩进风格编写，缩进的空格数为4个，是业界通用的标准。

**错误示例：**空格个数不为4个

```python
def load_data(dirname, one_hot=False):
     X_train = [] # 5个空格
     Y_train = [] # 5个空格
```

**正确示例：**

```python
def load_data(dirname, one_hot=False):
    X_train = []
    Y_train = []
```

## 规则1.2 禁止混合使用空格(space)和跳格(Tab)

**说明：**推荐的缩进方式为仅使用空格(space)。仅使用跳格(Tab)也是允许的。如果已有代码中混合使用了空格及跳格，要全部转换为空格。

**错误示例：**空格和跳格混合使用

```python
def load_data(dirname, one_hot=False):
    X_train = [] # 跳格
    Y_train = []
```

**正确示例：**

```python
def load_data(dirname, one_hot=False):
    X_train = []
    Y_train = []
```

## 规则1.3 新项目必须使用纯空格(spaces)来代替跳格(Tab)

**说明：**对于新项目，必须使用纯空格(spaces)来代替跳格(Tab)。

**错误示例：**新项目使用跳格

```python
def load_data(dirname, one_hot=False):
    X_train = [] # 跳格
    Y_train = [] # 跳格
```

**正确示例：**

```python
def load_data(dirname, one_hot=False):
    X_train = []
    Y_train = []
```

## 语句

## 规则1.4 Python文件必须使用UTF-8编码

**说明：**Python文件必须使用UTF-8编码，文件头可以添加编码声明`# coding: utf-8`

## 规则1.5 一行只写一条语句

**说明：**不允许把多个短语句写在一行中，即一行只写一条语句。多条语句写在一行，这样做一个很明显得缺点就是在调试的时候无法单步执行。

**错误示例：**多条语句在一行，不方便单步调试

```python
rect.length = 0; rect.width = 0;
```

**正确示例：**

```python
rect.length = 0
rect.width = 0
```

## 规则1.6 相对独立的程序块之间、变量说明之后必须加空行

**说明：**相对独立的程序块之间、变量说明之后加上空行，代码可理解性会增强很多。

**错误示例：**程序块之间未加空行

```python
if len(deviceName) < _MAX_NAME_LEN:
...
writer = LogWriter()
```

**正确示例：**

```python
if len(deviceName) < _MAX_NAME_LEN:
...

writer = LogWriter()
```

## 建议1.7 一行长度小于80个字符，与Python标准库看齐

**说明：**建议开发团队用本产品线的门禁工具或者yapf（https://github.com/google/yapf） 自动格式化，或者用IDE自带的格式化功能统一格式化代码后再提交。

较长的语句、表达式或参数（>80字符）要分成多行书写，首选使用括号（包括{},[],()）内的行延续，推荐使用反斜杠（\）进行断行。长表达式要在低优先级操作符处划分新行，操作符统一放在新行行首或原行行尾，划分出的新行要进行适当的缩进，使排版整齐，语句可读。

**错误示例：**一行字符太多，阅读代码不方便

```python
if width == 0 and height == 0 and color == 'red' and emphasis == 'strong' and
highlight > 100:
    x = 1
```

**正确示例：**

```python
if width == 0 \
    and height == 0 \
    and color == 'red' \
    and emphasis == 'strong' \
    and highlight > 100:
    x = 1
```

## 空格

## 规则1.8 在两个以上的关键字、变量、常量进行对等操作时，它们之间的操作符前后要加空格

**说明：**采用这种松散方式编写代码的目的是使代码更加清晰。

在长语句中，如果需要加的空格非常多，那么应该保持整体清晰，而在局部不加空格。给操作符留空格时不要连续留一个以上空格。

1. 逗号、分号（假如用到的话）只在后面加空格。

   错误示例：

   ```python
   print(a,b , c)
   ```

   正确示例：

   ```python
   print(a, b, c)
   ```

2. 比较操作符 ">"、">="、"<"、"<="、"=="，赋值操作符"="、"+="，算术操作符 "+"、"-"、"%"，逻辑操作符 "and"、"or"  等双目操作符的前后加空格。

   错误示例：

   ```python
   a=b+ c
   a+=2
   if current_time>= MAX_TIME_VALUE:
   ```

   正确示例：

   ```python
   a = b + c
   a += 2
   if current_time >= MAX_TIME_VALUE:
   ```

3. "*"、"**" 等作为操作符时，前后可以加空格，但若和更低优先级的操作符同时使用并且不涉及括号，则建议前后不加空格。

   正确示例：

   ```python
   a = b * c
   a = c ** b
   x = x*2 - 1
   ```

## 建议1.9 进行非对等操作时，如果是关系密切的立即操作符，后不应加空格（如 .）

1. 函数定义语句中的参数默认值，调用函数传递参数时使用的等号，建议不加空格

   ```python
   def create(self, name=None)
    self.create(name="mike")
   ```

2. "."前后不加空格。

   错误示例：

   ```python
   result. writeLog()
   ```

   正确示例：

   ```python
   result.writeLog()
   ```

3. 括号内侧，左括号后面和右括号前面，不需要加空格，多重括号间不必加空格。

   错误示例：

   ```python
   a = ( (b + c)*d - 5 )*6
   ```

   正确示例：

   ```python
   a = ((b + c)*d - 5)*6
   ```

4. 紧贴索引切片或被调用函数名，开始的括号前，不需要加空格。

   错误示例：

   ```python
   Dict [key] = list [index]
   conn = Telnet.connect (ipAddress)
   ```

   正确示例：

   ```python
   dict[key] = list[index]
   conn = Telnet.connect(ipAddress)
   ```

## 导入

## 规则1.10 加载模块必须分开每个模块占一行

**说明：**单独使用一行来加载模块，让程序依赖变得更清晰。

**错误示例：**

```python
import sys, os
```

**正确示例：**

```python
import sys
import os
```

注意虽然一行只能加载一个模块，但同一个模块内的多个符号可以在同一行加载。
**正确示例：**

```python
from sys import stdin, stdout
```

## 规则1.11 导入部分(imports)置于模块注释和文档字符串之后，模块全局变量和常量声明之前

**说明：**导入(import)库时，==按照标准库、第三方关联库、本地特定的库/程序顺序导入，并在这几组导入语句之间增加一个空行。==

**正确示例：**

```pyton
import sys
import os

from oslo_config import cfg
from oslo_log import log as logging

from cinder import context
from cinder import db
```

## 建议1.12 避免使用from xxx import *的方式导入某模块的所有成员。

**说明**：from xxx import * 会将其他模块中的所有成员挨个赋值给当前范围的同名变量，如果当前范围已经有同名变量，则会静默将其覆盖。这种方式容易导致名字冲突，且冲突后不容易定位，应当尽量避免使用。

**正确示例**： 如果需要使用yyy，则from xxx import yyy

## 解释器

## 建议1.13 类Unix操作系统上直接执行的Python文件头部建议使用#!/usr/bin/env python指定解释器

**说明：**类Unix操作系统上使用Hashbang “#!/usr/bin/env python”声明的时候，会去取系统的 PATH 变量中指定的第一个 Python 来执行你的脚本，有助于正确指定执行Python文件的解释器。Hashbang的位置需要放在文件编码声明之前。Windows操作系统可忽略此建议。

# 2. 注释

注释和文档字符串的原则是有助于对程序的阅读理解。python没有类型信息，IDE不能帮助提示，如果没有注释，动态语言就很难理解。注释不宜太多也不能太少，==一般建议建议有效注释量（包括文档字符串）应该在20%以上。==
撰写好的注释有以下建议：

- 注释描述必须准确、易懂、简洁，不能有二义性；
- 避免在注释和文档字符串中使用缩写，如果要使用缩写则需要有必要的说明；
- 修改代码时始终优先更新相应的注释/文档字符串，以保证注释/文档字符串与代码的一致性；
- 有含义的变量，如果不能充分自注释，则需要添加必要的注释；
- 全局变量建议添加详细注释，包括对其功能、取值范围、哪些函数或过程修改它以及存取时注意事项等的说明。

## 类、接口和函数

## 规则2.1 类和接口的文档字符串写在类声明(class ClassName:)所在行的下一行，并向后缩进4个空格

**说明：**类和接口的文档字符串的内容可选择包括（但不限于）功能描述，接口清单等。功能描述除了描述类或接口功能外，还要写明与其他类或接口之间的关系；接口清单列出该类或接口的接口方法的描述。

**正确示例：**

```python
class TreeError(libxmlError):
    """
    功能描述：
    接口：
    """
```

## 规则2.2 公共函数的文档字符串写在函数声明(def FunctionName(self):)所在行的下一行，并向后缩进4个空格

**说明：**公共函数文档字符串的内容可选择包括（但不限于）功能描述、输入参数、输出参数、返回值、调用关系（函数、表）、异常描述等。异常描述除描述函数内部抛出的异常外，还必须说明异常的含义及什么条件下抛出该异常。

**正确示例：**

```python
def load_batch(fpath):
    """
    功能描述：
    参数：
    返回值：
    异常描述：
    """
```

## 属性

## 规则2.3 公共属性的注释写在属性声明的上方，与声明保持同样的缩进。行内注释应以#和一个空格作为开始，与后面的文字注释以一个空格隔开

**说明：**行内注释的形式是在语句的上一行中加注释。行内注释要少用。它们==应以#和一个空格==作为开始。

**错误示例：**

```python
#Compensate for border
x = x + 1
```

**正确示例：**

```python
# Compensate for border
x = x + 1
```

## 格式

## 规则2.4 模块文档字符串写在文件的顶部，导入(import)部分之前的位置，不需要缩进

**说明：**==模块文档字符串应当包含功能描述和版权声明==。

**正确示例：**

```python
"""
功 能：XXX类，该类主要涉及XXX功能
版权信息：华为技术有限公司，版本所有(C) 2010-2019
"""
```

## 规则2.5 ==文档字符串多于一行时，末尾的"""要自成一行==

**说明：**对于只有一行的文档字符串，把"""放到同一行也没问题。

**错误示例：**

```python
"""Return a foobang
Optional plotz says to frobnicate the bizbaz first."""
```

**正确示例：**

```python
"""Return a foobang
Optional plotz says to frobnicate the bizbaz first.
"""
```

**正确示例：**

```python
"""API for interacting with the volume manager."""
```

## 规则2.6 注释必须与其描述的代码保持同样的缩进，并放在其上方相邻位置

**说明：**对代码的注释应放在其上方相邻位置，不可放在下面。

**错误示例：**注释与所描述的代码有不同的缩进

```python
    # get replicate sub system index and net indicator
repssn_ind = ssn_data[index].repssn_index
repssn_ni = ssn_data[index].ni
```

**正确示例：**

```python
# get replicate sub system index and net indicator
repssn_ind = ssn_data[index].repssn_index
repssn_ni = ssn_data[index].ni
```

**正确示例：**

```python
if image_service is not None:
    # Deletes the image if it is in queued or saving state
    self._delete_image(context, image_meta['id'], image_service)
```

# 3.命名

## 命名规范推荐表

| **Type**                   | **Public**         | **Internal**                                                 |
| :------------------------- | :----------------- | :----------------------------------------------------------- |
| Modules                    | lower_with_under   | _lower_with_under                                            |
| Packages                   | lower_with_under   |                                                              |
| Classes                    | CapWords           |                                                              |
| Exceptions                 | CapWords           |                                                              |
| Functions                  | lower_with_under() | _lower_with_under()                                          |
| Global/Class Constants     | CAPS_WITH_UNDER    | _CAPS_WITH_UNDER                                             |
| Global/Class Variables     | lower_with_under   | lower_with_under                                             |
| Instance Variables         | lower_with_under   | _lower_with_under or __lower_with_under (当需要名字修饰时)   |
| Method Names               | lower_with_under() | _lower_with_under() or __lower_with_under() (当需要名字修饰时) |
| Function/Method Parameters | lower_with_under   |                                                              |
| Local Variables            | lower_with_under   |                                                              |

## 包和模块

## 规则3.1 包（Package）、模块（Module）名使用意义完整的英文描述，采用小写加下划线（lower_with_under）的风格命名

**说明:**模块应该用小写加下划线的方式（如lower_with_under.py）命名。==尽管已经有很多现存的模块使用类似于CapWords.py这样的命名,但现在已经不鼓励这样做, 因为如果模块名碰巧和类名一致, 这会让人困扰==。

**正确示例：**

```python
from sample_package import sample_module
from sample_module import SampleClass
```

## 类

## 规则3.2 类（Class）名使用意义完整的英文描述，采用大写字母开头的单词（CapWords）风格命名

**说明:**类沿用面向对象语言最常用的CapWords风格命名。

**正确示例：**

```python
class SampleClass(object):
    pass
```

## 函数

## 规则3.3 函数（Function）、方法（Method）、函数参数（Function Parameters）名使用意义完整的英文描述，采用小写加下划线（lower_with_under）的风格命名

**说明:**
函数、方法采用小写加下划线的风格命名，与类名做区分。
函数参数采用小写加下划线的风格命名，与一般变量的命名风格保持一致。
模块内部使用的函数用单下划线(_)开头，表示函数是protected的(使用from module1 import *时不会包含)。

**正确示例：**

```python
def sample_public_function(sample_parameter):
    pass

def sample_internal_function(sample_parameter):
    pass

class SampleClass(object):

    def sample_member_method(self, sample_parameter):
        pass
```

## 变量

## 规则3.4 变量（variable）采用小写加下划线（lower_with_under）的风格命名。常量（constant）采用大写加下划线（CAPS_WITH_UNDER）的风格命名

**说明:**

常量使用大写加下划线的风格命名，与变量做区分。

**正确示例：**

```python
sample_global_variable = 0
M_SAMPLE_GLOBAL_CONSTANT = 0

class SampleClass(object):

    SAMPLE_CLASS_CONSTANT = 0

    def sample_member_method(self, sample_parameter):
        pass

def sample_function():
    sample_function_variable = 0
    sample_instant_variable = SampleClass()
```

## 规则3.5 类或对象的私有成员一般用单下划线_开头；==对于需要被继承的基类成员，如果想要防止与派生类成员重名，可用双下划线__开头==。

**说明**：
Python没有严格的私有权限控制，业界约定俗成的用单下划线“_”开头来暗示此成员仅供内部使用。双下划线“__”开头的成员会被解释器自动改名，加上类名作为前缀，其作用是防止在类继承场景中出现名字冲突，并不具有权限控制的作用，外部仍然可以访问。双下划线开头的成员应当只在需要避免名字冲突的场景中使用（比如设计为被继承的工具基类）。

**正确示例**：

```python
class MyClass:
    def my_func(self):
        self._member = 1    # 单下划线开头，暗示此成员仅供类的内部操作使用，外部不应该访问。

    def _my_private_func(self):   # 单下划线开头，暗示此方法仅供类的内部操作使用，外部不应该访问。
        pass

class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)    # 双下划线开头，会被解释器改名为_Mapping__update。外部如果使用修改后的名字仍可访问

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update   # 作为update方法的私有复制成员，不会跟派生类成员重名

class MappingSubclass(Mapping):
    # 和基类同名方法，修改了参数个数，但是不会影响基类__init__
    def update(self, keys, values):
        for item in zip(keys, values):
            self.items_list.append(item)

    __update = update   # 被解释器改名为_MappingSubclass__update，不会跟基类成员重名
```

参考资料：https://docs.python.org/3/tutorial/classes.html#private-variables

## 建议3.6 变量（variable）命名要有明确含义，使用完整的单词或大家可以理解的缩写，避免使用单个字符

**说明:**

1. 命名中若使用了特殊约定或缩写，建议注释说明。
2. 对于变量命名，除局部循环变量之外，不允许取单个字符（如i、j、k）。
3. 对于允许使用单个字符命名的场景，==不要用字符"l"，"o"来做变量名称==。在有些字体中，这些字符于数字很难1和0很难辨认。若确实需要使用"l"做变量，用"L"来替换。

**错误示例：**

```python
class SampleClass(object):
    pass

def sample_function(sample_parameter):
    i = SampleClass()
    o = [l for l in range(1)]
```

**正确示例：**

```python
class SampleClass(object):
    pass

def sample_function(sample_parameter):
    sample_inst = SampleClass()
    number_list = [i for i in range(10)]
```

# 4. 编码

## 规则4.1 与None作比较要使用“is”或“is not”，不要使用等号

**说明：**

“is”判断是否指向同一个对象（判断两个对象的id是否相等），“==”会调用**eq**方法判断是否等价（判断两个对象的值是否相等）。

**示例：**

同一个实例，使用“is”和“==”的判断结果不同。

```
>>> class Bad(object):
        def __eq__(self, other):
            return True
>>> bad_inst = Bad()
>>> bad_inst == None
True
>>> bad_inst is None
False
```

## 建议4.2 当模块有不对外暴露的成员时，定义__all__列表，将允许外部访问的变量、函数和类的名字放进去

**说明：**

在模块中定义了__all__之后，从外部from module import *只会import __all__中定义的内容。

**示例：**

sample_package.py

```python
__all__ = ["sample_external_function"]

def sample_external_function():
    print("This is an external function..")

def sample_internal_function():
    print("This is an internal function..")
```

main.py

```python
from sample_package import *

if __name__ == "__main__":
    sample_external_function()
    sample_internal_function()

NameError: name 'sample_internal_function' is not defined
```

## 建议4.3 避免直接使用dict[key]的方式从字典中获取value，如果一定要使用，需要注意当key not in dict时的异常捕获和处理

**说明：**

Python的字典dict可以使用key获取其对应的value。但是当key在dict的key值列表中不存在时，直接使用dict[key]获取value会报KeyError，应当使用更为安全的dict.get(key)类型方法获取value。

**错误示例：**

```python
sample_dict = {'default_key': 1}
sample_key = 'sample_key'
sample_value = sample_dict[sample_key]
```

**正确示例：**

```python
sample_dict = {'default_key': 1}
sample_key = 'sample_key'
sample_value = sample_dict.get(sample_key)
```

## 建议4.4 对序列使用切片操作时，不建议使用负步进值进行切片

**说明：**

Python提供了sample_list[start : end : stride]形式的写法，以实现步进切割，也就是从每stride个元素中取一个出来。==但如果stride值为负，则会使代码难以理解，特定使用场景下还会造成错误==。

**错误示例：**

如下写法，在start : end : stride都使用的情况下使用负的stride，会造成阅读困难。此种情况建议将“步进”切割过程和“范围”切割过程分开，使代码更清晰。

```
>>> a = [1,2,3,4,5,6,7,8]
>>> a[2::2]
[3,5,7]
>>> a[-2::-2]
[7,5,3,1]
>>> a[-2:2:-2]
[7,5]
>>> a[2:2:-2]
[]
```

## 建议4.5 传递实例类型参数后，函数内应使用isinstance函数进行参数检查，不要使用type

**说明：**==如果类型有对应的工厂函数，可使用它对类型做相应转换，否则可以使用isinstance函数来检测==。使用函数/方法 参数传递 实例类型 参数后，函数内对此参数进行检查应使用isinstance函数，使用is not None，len(para) != 0等其它逻辑方法都是不安全的。

**错误示例：**

下面的函数保护未能完成应有的检查功能，传入一个tuple就可以轻易绕过保护代码造成执行异常。

```python
>>> def sample_sort_list(sample_inst):
...     if sample_inst is []:
...         return
...     sample_inst.sort()
>>> fake_list = (2,3,1,4)
>>> sample_sort_list(fake_list)
Traceback (most recent call last):
  File "<pyshell#232>", line 1, in <module>
    sample_sort_list(fake_list)
  File "<pyshell#230>", line 4, in sample_sort_list
    sample_inst.sort()
AttributeError: 'tuple' object has no attribute 'sort'
```

**正确示例：**

使用instance函数对入参进行检查，检查后可以按照需求 raise exception或return。

```python
>>> def sample_sort_list(sample_inst):
...     if not isinstance(sample_inst, list):
...         raise TypeError(r"sample_sort_list in para type error %s" % 																			      type(sample_inst))
...     sample_inst.sort()
>>> fake_list = (2,3,1,4)
>>> sample_sort_list(fake_list)
Traceback (most recent call last):
  File "<pyshell#235>", line 1, in <module>
    sample_sort_list(fake_list)
  File "<pyshell#234>", line 3, in sample_sort_list
    raise TypeError(r"sample_sort_list in para type error %s" % type(sample_inst))
TypeError: sample_sort_list in para type error <type 'tuple'>
```

## 建议4.6 使用推导式代替重复的逻辑操作构造序列。但推导式必须考虑可读性，不在一个推导式中使用三个以上的for语句

**说明：**

==推导式（comprehension）是一种精炼的序列生成写法，在可以使用推导式完成简单逻辑，生成序列的场合尽量使用推导式==，但如果逻辑较为复杂（>=3个for语句），则不推荐强行使用推导式，因为这会使推导式代码的可读性变差。

**错误示例1：**
如下逻辑代码，逻辑较为简单，实现此逻辑的代码不但有循环，而且较为复杂，性能不佳。

```python
odd_num_list = []
for i in range(100):
    if i % 2 == 1:
        odd_num_list.append(i)
```

**正确示例1：**
简单逻辑使用列表推导式实现，代码清晰精炼。

```python
odd_num_list = [i for i in range(100) if i % 2 == 1]
```

**错误示例2：**
一个推导式中使用了三个for语句，太复杂。

```python
lengths = [1, 2]
widths = [3, 4]
heights = [5, 6]
cubes = [(x, y, z) for x in lengths for y in widths for z in heights]
```

**正确示例2：**
使用其他手段（比如标准库中的 itertools 工具）来简化代码。

```python
import itertools

lengths = [1, 2]
widths = [3, 4]
heights = [5, 6]
cubes = list(itertools.product(lengths, widths, heights))
```

## 建议4.7 功能代码应该封装在函数或类中

**说明：**在Python中, 所有的顶级代码在模块导入时都会被执行。容易产生调用函数，创建对象等误操作，所以代码应该封装在函数或类中。==即使是脚本类的代码，也建议在执行主程序前总是检查`if __name__ == '__main__'`，这样当模块被导入时主程序就不会被执行==。

**正确示例：**

```python
def main():
    ...

if __name__ == '__main__':
    main()
```

## 建议4.8 需要精确数值计算的场景，应使用decimal模块，且不要用浮点数构造Decimal

**说明：**在Python中，浮点数无法被精确的表示，多次计算以后可能出现尾差。

**示例:**

```python
from decimal import Decimal

print('%.20f' % 3.14) # 输出3.14000000000000012434
print(Decimal('3.14')) # 精确的始终只用两位小数表示
```

==注意不要用浮点数构造Decimal，因为浮点数本身不准确==。

**错误示例:**

```
>>> from decimal import Decimal, getcontext
>>> getcontext().prec = 28
>>> Decimal(3.14)
Decimal('3.140000000000000124344978758017532527446746826171875')
```

**正确示例：**

```
>>> from decimal import Decimal, getcontext
>>> Decimal('3.14')
Decimal('3.14')
>>> getcontext().prec = 6
>>> Decimal(1) / Decimal(7)
Decimal('0.142857')
```

## 规则4.9 避免在无关的变量或无关的概念之间重用名字，避免因重名而导致的意外赋值和错误引用

**说明：**
Python的函数/类定义和C语言不同，函数/类定义语句实际上是给一个名字赋值。因此重复定义一个函数/类的名字不会导致错误，后定义的会覆盖前面的。但是重复定义很容易掩盖编码问题，让同一个名字的函数/类在不同的执行阶段具有不同的含义，不利于可读性，应予以禁止。

==Python在解析一个被引用的名字时遵循LEGB顺序（Local - Enclosed - Global - Builtin）==，从内层一直查找到外层。内层定义的变量会覆盖外层的同名变量。在代码修改时，同名的变量容易导致错误的引用，也不利于代码可读性，应当尽量避免。

**错误示例1**

```python
l = [[1,2,3], [2,3,4], [3,4,5]]

# 对l中每个列表找出最大值
for x in l:
    maxnum = 0
    for x in x:  # Error, 应该是 `for y in x`
        maxnum = max(x, maxnum)  # 应该是 `max(y, maxnum)`
    print("The largest number in the list %s is %s" % (str(x), maxnum)) # x的值已被修改
```

**错误示例2**

```python
def function(para, type):  # 参数名覆盖了builtin函数type
    ...
    para_type = type(para)  # error，type被覆盖为函数参数，不是builtin的实现
```

## 规则4.10 类的方法不需访问实例时，根据具体场景选择使用@staticmethod或者@classmethod进行装饰

**说明**：
一般的类方法要接收一个self参数表示此类的实例，但有些方法不需要访问实例，这时分为两种情况：
1、方法不需要访问任何成员，或者只需要显式访问这个类自己的成员。这样的方法不需要额外参数，应当用@staticmethod装饰。
在Python 3.X版本中，允许直接定义不含self参数的方法，并且允许不通过实例调用。但是一旦通过实例调用这个方法，就会因为参数不匹配而出错。
加上@staticmethod进行修饰，可以让Python解释器明确此方法不需要self参数，提前拦截问题，可读性也更好。

**错误示例**：

```python
class MyClass:
    def my_func():    # 没有用@staticmethod修饰，通过实例调用会出错
        pass

MyClass.my_func()    # Python 3.X中允许，2.X中出错
my_instance = MyClass()
my_instance.my_func()   # Python 3.X和2.X中都会出错
```

**正确示例**：

```python
class MyClass:
    @staticmethod
    def my_func():     # 用@staticmethod修饰后，解释器会将其解析为静态方法
        pass

MyClass.my_func()    # OK
my_instance = MyClass()
my_instance.my_func()   # OK，但是不推荐，容易和普通方法混淆。最好写成MyClass.my_func()
```

2、==方法不需要访问实例的成员，但需要访问基类或派生类的成员。这时应当用@classmethod装饰==。装饰后的方法，其第一个参数不再传入实例，而是传入调用者的最底层类。
比如，下面这个例子，通过基类Spam的count方法，来统计继承树上每个类的实例个数：

```python
class Spam:
    num_instances = 0
    @classmethod
    def count(cls):    # 对每个类做独立计数
        cls.num_instances += 1    # cls是实例所属于的最底层类
    def __init__(self):
        self.count()    # 将self.__class__传给count方法

class Sub(Spam):
    num_instances = 0

class Other(Spam):
    num_instances = 0

x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()
x.num_instances, y1.num_instances, z1.num_instances    # 输出：(1, 2, 3)
Spam.num_instances, Sub.num_instances, Other.num_instances    # 输出：(1, 2, 3)
```

但是使用@classmethod时需要注意，由于在继承场景下传入的第一个参数并不一定是这个类本身，因此并非所有访问类成员的场景都应该用@classmethod。比如下面这个例子中，Base显式的想要修改自己的成员inited（而不是派生类的成员），这时应当用@staticmethod。

**错误示例**：

```python
class Base:
    inited = False
    @classmethod
    def set_inited(cls):   # 实际可能传入Derived类
        cls.inited = True    # 并没有修改Base.inited，而是给Derived添加了成员

class Derived(Base):
    pass

x = Derived()
x.set_inited()
if Base.inited:
    print("Base is inited")   # 不会被执行
```

## 建议4.11 当多个Python源码文件分不同子目录存放时，用包（package）形式管理各个目录下的模块。

**说明**：
通过让子目录包含__init__.py文件，可以让Python代码在import和from语句中，将子目录作为包名，通过分层来管理各个模块，让模块间的关系更清楚。__init__.py文件中可以包含这个包所需要的初始化动作，也可以定义一个__all__列表来指定from *语句会包含哪些模块。对于不需要初始化的包，可以只在目录下放一个名为__init__.py的空文件，标识这个目录是一个包。

**正确示例**：
假设Python源码根目录是dir0，其下有子目录dir1，dir1下面又有个子目录dir2，dir2下面有个mod.py模块。那么，在dir1和dir2下各放置一个__init__.py文件，然后在其他代码中可以这样使用mod.py模块：

```python
import dir1.dir2.mod
dir1.dir2.mod.func()    # 调用mod.py中的func函数

from dir1.dir2.mod import func    # 把func函数添加到当前空间
func()    # 可以省掉包名和模块名直接调用
```

## 建议4.12 避免在代码中修改sys.path列表

**说明**：
sys.path是Python解释器在执行import和from语句时参考的模块搜索路径，==由当前目录、系统环境变量、库目录、.pth文件配置组合拼装而成==。用户通过修改系统配置，可以指定搜索哪个路径下的模块。sys.path只应该根据用户的系统配置来生成，不应该在代码里面直接修改。否则可能出现A模块修改了sys.path，导致B模块搜索出错，且用户难以定位。

**正确示例**：
如果要添加模块搜索路径，应当修改PYTHONPATH环境变量。如果是管理子目录，应当通过包（package）来组织模块。

## 建议4.13 尽量不使用for i in range(x)的方式循环处理集合数据，而应使用for x in iterable的方式

**说明**：
for i in range(x)，然后在循环体内对集合用下标[i]获取元素是C语言的编程习惯，它有很多缺点：容易越界；在循环体内修改i容易出错；可读性差。Python语言建议尽量用for x in iterable的方式直接取集合的每一条数据进行处理。

**错误示例**：

```python
for i in range(len(my_list)):
    print(my_list[i])
```

**正确示例**：

```python
for x in my_list:
    print(x)
```

有些场合下，需要在处理时使用每个元素的序号，这时可以使用enumerate内置函数来给元素加上序号形成元组：

```python
my_list = ['a', 'b', 'c']
for x in enumerate(my_list):
    print(x)
```

运行结果为：
(0, 'a')
(1, 'b')
(2, 'c')

## 建议4.14 避免变量在其生命周期内的对象类型发生变化

Python是动态类型语言，允许变量被赋值为不同类型对象，但这么做可能会导致运行时错误，且因为变量上下文语义变化导致代码复杂度提升，难以调试和维护，也不会有任何性能的提升。

**错误示例**

```python
items = 'a,b,c,d' # 字符串
items = items.split(',') # 变更为列表
```

**正确示例**

```python
items = 'a,b,c,d' # 字符串
item_list = items.split(',') # 变更为列表
```

# 5. 异常处理

## 异常处理

## 规则5.1 使用try…except…结构对代码作保护时，需要在异常后使用finally…结构保证操作对象的释放

**说明：**

使用try…except…结构对代码作保护时，如果代码执行出现了异常，为了能够可靠地关闭操作对象，需要使用finally…结构确保释放操作对象。

**示例：**

```python
handle = open(r"/tmp/sample_data.txt")  # May raise IOError
try:
    data = handle.read()  # May raise UnicodeDecodeError
except UnicodeDecodeError as decode_error:
    print(decode_error)
finally:
    handle.close()  # Always run after try:
```

## 规则5.2 不要使用“except:”语句来捕获所有异常

**说明：**

在异常这方面, Python非常宽容,“except:”语句真的会捕获包括Python语法错误在内的任何错误。使用“except:”很容易隐藏真正的bug，我们在使用try…except…结构对代码作保护时，应该明确期望处理的异常。
Exception类是大多数运行时异常的基类，一般也应当避免在except语句中使用。通常，try只应当包含必须要在当前位置处理异常的语句，except只捕获必须处理的异常。比如对于打开文件的代码，try应当只包含open语句，except只捕获FileNotFoundError异常。对于其他预料外的异常，则让上层函数捕获，或者透传到程序外部来充分暴露问题。

**错误示例：**

如下代码可能抛出两种异常，使用“except:”语句进行统一处理时，如果是open执行异常，将在“except:”语句之后handle无效的情况下调用close，报错handle未定义。

```python
try:
    handle = open(r"/tmp/sample_data.txt")  # May raise IOError
    data = handle.read()  # May raise UnicodeDecodeError
except:
    handle.close()
```

**正确示例：**

```python
try:
    handle = open(r"/tmp/sample_data.txt")  # May raise IOError
    try:
        data = handle.read()  # May raise UnicodeDecodeError
    except UnicodeDecodeError as decode_error:
        print(decode_error)
    finally:
        handle.close()

except(FileNotFoundError, IOError) as file_open_except:
    print(file_open_except)
```

## 规则5.3 不在except分支里面的raise都必须带异常

**说明：**raise关键字单独使用只能出现在try-except语句中，重新抛出except抓住的异常。

**错误示例：**

```
>>> a = 1
>>> if a==1:
...     raise
...
Traceback (most recent call last):
File "<stdin>", line 2, in <module>
TypeError: exceptions must be old-style classes or derived from BaseException, not NoneType
```

**正确示例1：raise一个Exception或自定义的Exception**

```
>>> a = 1
>>> if a==1:
...     raise Exception
...
Traceback (most recent call last):
File "<stdin>", line 2, in <module>
Exception
```

**正确示例2：在try-except语句中使用**

```
>>> import sys
>>>
>>> try:
...     f = open('myfile.txt')
...     s = f.readline()
...     i = int(s.strip())
... except IOError as e:
...     print("I/O error({0}): {1}".format(e.errno, e.strerror))
... except ValueError:
...     print("Could not convert data to an integer.")
... except Exception:
...     print("Unexpected error:", sys.exc_info()[0])
...     raise
```

## 建议5.4 ==尽量用异常来表示特殊情况，而不要返回None==

当我们在一个工具方法时，通常会返回None来表明特殊的意义，比如一个数除以另外一个数，如果被除数为零，那么就返回None来表明是没有结果的

```python
def divide(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return None

result = divide(x, y)
if result is None:
    print('Invalid inputs')
```

当分子为零是会返回什么？应该是零（如果分母不为零的话），上面的代码在if条件检查就会被忽略掉，if条件不仅仅只检查值为None，还要添加所有条件为False的情况了

```python
x, y = 0, 5
result = divide(x, y)
if not result:
    print('Invalid inputs') #This is wrong!
```

上面的情况是python编码过程中很常见，这也为什么方法的返回值为None是一种不可取的方式，这里有两种方法来避免上面的错误。

1.第一种方法是将返回值分割成一个tuple，第一部分表示操作是否成功，第二部分是实际的返回值（有点象go语言里的处理）

```python
def divide(a, b):
    try:
        return True, a / b
    except ZeroDivisionError:
        return False, None
```

调用此方法时获取返回值并解开，检查第一部分来代替之前仅仅检查结果。

```python
success, result = divide(x, y)
if not success:
    print('Invalid inputs')
```

这种方式会带来另外一个问题，方法的调用者很容易忽略掉tuple的第一部分（通过在python里可以使用_来标识不使用的变量），这样的代码乍一看起来不错，但是实际上和直接返回None没什么两样。

```python
_, result = divide(x, y)
if not result:
    print('Invalid inputs')
```

2.接下来，另外一种方式，也是推荐的一种方式，就是触发异常来让调用者来处理，方法将触发ValueError来包装现有的ZeroDivisionError错误用来告诉方法调用者输入的参数是有误的。

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs') from e
```

那么方法调用者必须要处理错误输入值而产生的异常（方法的文档字符串应该注明异常情况）。同时也不用去检查返回值，因为当方法没有异常抛出时，返回值一定是对的，对于异常的处理也是很清晰。

```
x, y = 5, 2
try:
    result = divide(x, y)
except ValueError:
    print('Invalid inputs')
else:
    print('Result is %.1f' % result)
>>>
Result is 2.5
```

需要记住的：
(1)方法使用None作为特殊含义做为返回值是非常糟糕的编码方式，因为None和其它的返回值必须要添加额外的检查代码。
(2)触发异常来标示特殊情况，调用者会在捕获异常来处理。

## 建议5.5 避免finally中可能发生的陷阱，==不要在finally中使用return或者break语句==

通常使用finally语句，表明要释放一些资源，这时候try和except还有else代码块都被执行过了，如果在执行它们的过程中有异常触发，且没有处理这个异常，那么异常会被暂存，当finally代码执行后，异常会重新触发，但是当finally代码块里有**return**或**break**语句时，这个暂存的异常就会丢弃：

```python
def f():
    try:
        1/0
    finally:
        return 42

print(f())
```

上面的代码执行完后1/0产生的异常就会被忽略，最终输出42，因此在finally里出现return是不可取的。

当try块中return，break，continue执行时，finally块依然会被执行。

```python
def foo():
    try:
        return 'try'
    finally:
        return 'finally'


>>> foo()
'finally'
```

最终方法的输出其实不是正确的结果，但出现这个问题的原因是错误使用了return和break语句。

## 规则5.6 ==禁止使用except X, x语法，应当使用except X as x==

**说明**：
except X, x语法只在2.X版本支持，3.X版本不支持，有兼容性问题。而且，except X, x写法容易和多异常捕获的元组（tuple）表达式混淆。因此应该统一用except X as x方式。

## 断言

## 建议5.7 assert语句通常只在测试代码中使用，禁止在生产版本中包含assert功能

**assert**语句用来声明某个条件是真的。例如，如果你非常确信某个列表中至少有一个元素，而你想检验这一点，并且在它非真时触发一个异常，那么**assert**就是这种场景下的不二之选。当**assert**语句失败的时候，会触发**AssertionError**异常

```
>>> mylist = ['item']
>>> assert len(mylist) >= 1
>>> mylist.pop()
'item'
>>> assert len(mylist) >= 1
Traceback (most recent call last): File "<stdin>", line 1, in ? AssertionError
```

**assert**只应在研发过程中内部测试时使用，出现了**AssertionError**异常说明存在软件设计或者编码上的错误，应当修改软件予以解决。在对外发布的生产版本中禁止包含**assert**功能。

# 6. 并发与并行

## 多线程适用于阻塞式IO场景，不适用于并行计算场景

Python的标准实现是CPython。

CPython执行Python代码分为2个步骤：首先，将文本源码解释编译为字节码，然后再用一个解释器去解释运行字节码。字节码解释器是有状态的，需要维护该状态的一致性，因此使用了GIL（Global Interpreter Lock，全局解释器锁）。

==GIL的存在，使得CPython在执行多线程代码的时候，同一时刻只有一个线程在运行，无法利用多CPU提高运算效率==。但是这个特点也带来了一个好处：CPython运行多线程的时候，内部对象缺省就是线程安全的。这个特性，被非常多的Python库开发者所依赖，直到CPython的开发者想要去除GIL的时候，发现已经有大量的代码库重度依赖这个GIL带来的内部对象缺省就是线程安全的特性，变成一个无法解决的问题了。

==虽然多线程在并行计算场景下无法带来好处，但是在阻塞式IO场景下，却仍然可以起到提高效率的作用。==这是因为阻塞式IO场景下，线程在执行IO操作时并不需要占用CPU时间，此时阻塞IO的线程可以被挂起的同时继续执行IO操作，而让出CPU时间给其他线程执行非IO操作。这样一来，多线程并行IO操作就可以起到提高运行效率的作用了。

综上，Python的标准实现CPython，由于GIL的存在，同一个时刻只能运行一个线程，无法充分利用多CPU提升运算效率，因此Python的多线程适用于阻塞式IO的场景，不适用于并行计算的场景。

下面举一个对计算量有要求的求一个数的因数分解的代码实例，来说明Python多线程不适用于并行计算的场景：

```python
# -*- coding:utf-8 -*-
from time import time
from threading import Thread


def factorize(number):
    for i in range(1, number + 1):
        if number % i == 0:
            yield i


class FactorizeThread(Thread):
    def __init__(self, number):
        Thread.__init__(self)
        self.number = number

    def run(self):
        self.factors = list(factorize(self.number))


def test(numbers):
    start = time()
    for number in numbers:
        list(factorize(number))
    end = time()
    print('Took %.3f seconds' % (end - start))


def test_thread(numbers):
    start = time()
    threads = []
    for number in numbers:
        thread = FactorizeThread(number)
        thread.start()
        threads.append(thread)
    for t in threads:
        t.join()
    end = time()
    print('Mutilthread Took %.3f seconds' % (end - start))


if __name__ == "__main__":
    numbers = [2139079, 1214759, 1516637, 1852285]

    test(numbers)
    test_thread(numbers)
```

代码输出：

```
Took 0.319 seconds
Mutilthread Took 0.539 seconds
```

以上代码运行结果只是一个参考值，具体数据跟运行环境相关。但是可以看到单线程方式比多线程方式的计算速度要快。由于CPython运行多线程代码时因为GIL的原因导致每个时刻只有一个线程在运行，因此多线程并行计算并不能带来时间上的收益，反而因为调度线程而导致总时间花费更长。

==对于IO阻塞式场景，多线程的作用在于发生IO阻塞操作时可以调度其他线程执行非IO操作，因此在这个场景下，多线程是可以节省时间的==。可以用以下的代码来验证：

```python
# -*- coding:utf-8 -*-
from time import time
from threading import Thread
import os


def slow_systemcall(n):
    for x in range(100):
        open("test_%s" % n, "a").write(os.urandom(10) * 100000)


def test_io(N):
    start = time()
    for _ in range(N):
        slow_systemcall(_)
    end = time()
    print('Took %.3f seconds' % (end - start))


def test_io_thread(N):
    start = time()
    threads = []
    for _ in range(N):
        thread = Thread(target=slow_systemcall, args=("t_%s"%_,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    end = time()
    print('Multithread Took %.3f seconds' % (end - start))


if __name__ == "__main__":
    N = 5
    test_io(N)
    test_io_thread(N)
```

代码输出：

```
Took 5.179 seconds
Multithread Took 1.451 seconds
```

可以看到单线程花费时间与多线程花费时间之比接近1:4，考虑线程调度的时间，这个跟一般语言的多线程起的作用比较相似。==这是因为当Python执行IO操作时，实际上是执行了系统调用，此时线程会释放GIL，直到系统调用结束时，再申请获取GIL，也就是在IO操作期间，线程确实是并行执行的==。

Python的另外一个实现JPython就没有GIL，但是它并不是最常见的Python实现。

# 7. 性能

## 建议7.1 在list成员个数可以预知的情况下，创建list时需预留空间正好容纳所有成员的空间

说明：与Java、C++等语言的list一样，==Python语言的list在append()成员时，如果没有多余的空间容纳新的成员，就会分配一块更大的内存，并将原来内存里的成员拷贝到新的内存上，并将最新append()的成员也拷贝到此新内存空间中，然后释放老的内存空间。如果append()调用次数很大，则如上过程会频繁发生，因而会造成灾难性性能下降，而不仅仅是一点下降。==

错误示例：

```python
members = []
for i in range(1, 1000000):
     members.append(i)
len(members)
```

正确示例：

```
members = [None] * 1000000
for i in range(1, 1000000):
     members[i] = i
len(members)
```

## 建议7.2 在成员个数及内容皆不变的场景下尽量使用tuple替代list

说明：==list是动态array，而tuple是静态array（其成员个数以及内容皆不可变）。因此，list需要更多的内存来跟踪其成员的状态==。

此外，对于成员个数小于等于20的tuple，Python会对其进行缓存，即当此tuple不再使用时，Python并不会立即将其占用的内存返还给操作系统，而是保留以备后用。

错误示例：

```python
myenum = [1, 2, 3, 4, 5]
```

正确示例：

```python
myenum = (1, 2, 3, 4, 5)  # 如果数据恰好被缓存过，则初始化速度会为错误示例中的5倍以上。
```

## 建议7.3 尽量使用generator comprehension代替list comprehension

说明：list comprehension可以用来代替lambda表达式的map、reduce语法，从已有的list中，生成新的数据。而generator comprehension无需定义一个包含yield语句的函数，就可以生成一个generator。
==二者一个生成list，另外一个生成generator，在内存的占用上，相差悬殊；在生成速度上，相差无几。==

错误示例：

```python
even_cnt = len([x for x in range(10) if x % 2 == 0])
```

正确示例：

```python
even_cnt = sum(1 for x in range(10) if x % 2 == 0)
```

## 建议7.4 在循环中，==使用format方法、"%"操作符 和 join方法代替"+"和"+="操作符来完成字符串格式化==

**说明：**即使参数都是字符串，也可以使用format方法或%运算符来格式化字符串。一般性能要求的场景可以使用+或+=运算符，但需要避免使用+和+=运算符在循环中累积字符串。由于字符串是不可变的，因此会产生不必要的临时对象并导致二次而非线性运行时间。

**正确示例：**

```python
x = '%s, %s!' % (imperative, expletive)
x = '{}, {}!'.format(imperative, expletive)
x = 'name: %s; score: %d' % (name, n)
x = 'name: {}; score: {}'.format(name, n)
name = "Fred"
x = f"He said his name is {name}."
items = ['<table>']
for last_name, first_name in employee_list:
    items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
items.append('</table>')
employee_table = ''.join(items)
```

**错误示例：**

```python
x = imperative + ', ' + expletive + '!'
x = 'name: ' + name + '; score: ' + str(n)
employee_table = '<table>'
for last_name, first_name in employee_list:
    employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
employee_table += '</table>'
```

# 8. 编程实践

## 规则8.1 ==函数参数中的可变参数不要使用默认值，在定义时使用None==

**说明：参数的默认值会在方法定义被执行时就已经设定了**，这就意味着默认值只会被设定一次，当函数定义后，每次被调用时都会有"预计算"的过程。当参数的默认值是一个可变的对象时，就显得尤为重要，例如参数值是一个list或dict，如果方法体修改这个值(例如往list里追加数据)，那么这个修改就会影响到下一次调用这个方法，这显然不是一种好的方式。应对种情况的方式是将参数的默认值设定为None。

**错误示例：**

```
>>> def foo(bar=[]): # bar is optional and defaults to [] if not specified
...    bar.append("baz") # but this line could be problematic, as we'll see...
...    return bar
```

在上面这段代码里，一旦重复调用foo()函数（没有指定一个bar参数），那么将一直返回'bar'。因为没有指定参数，那么foo()每次被调用的时候，都会赋予[]。下面来看看，这样做的结果：

```
>>> foo()
["baz"]
>>> foo()
["baz", "baz"]
>>> foo()
["baz", "baz", "baz"]
```

**正确示例：**

```
>>> def foo(bar=None):
...    if bar is None:      # or if not bar:
...        bar = []
...    bar.append("baz")
...    return bar
...
>>> foo()
["baz"]
>>> foo()
["baz"]
>>> foo()
["baz"]
```

## 规则8.2 严禁使用注释行等形式仅使功能失效

**说明：**python的注释包含：单行注释、多行注释、代码间注释、文档字符串等。除了文档字符串是使用""""""括起来的多行注释，常用来描述类或者函数的用法、功能、参数、返回等信息外，其余形式注释都是使用#符号开头用来注释掉#后面的内容。

基于python语言运行时编译的特殊性，如果在提供代码的时候提供的是py文件，即便是某些函数和方法在代码中进行了注释，==别有用心的人依然可以通过修改注释来使某些功能启用==；尤其是某些接口函数，如果不在代码中进行彻底删除，很可能在不知情的情况下就被启用了某些本应被屏蔽的功能。因此根据红线要求，==在python中不使用的功能、模块、函数、变量等一定要在代码中彻底删除==，不给安全留下隐患。即便是不提供源码py文件，提供编译过的pyc、pyo文件，别有用心的人可以通过反编译来获取源代码，可能会造成不可预测的结果。

**错误示例：**在main.py中有两个接口被注释掉了，但是没有被删除。

```python
if __name__ == "__main__":
    if sys.argv[1].startswith('--'):
        option = sys.argv[1][2:]
        if option == "load":
            #安装应用
            LoadCmd(option, sys.argv[2:3][0])
        elif option == 'unload':
            #卸载应用
            UnloadCmd(sys.argv[2:3][0])
        elif option == 'unloadproc':
            #卸载流程
            UnloadProcessCmd(sys.argv[2:3][0])
#       elif option == 'active':
#           ActiveCmd(sys.argv[2:3][0])
#       elif option == 'inactive':
#           InActiveCmd(sys.argv[2:3][0])
        else:
            Loginfo("Command %s is unknown"%(sys.argv[1]))
```

==在上例中很容易让其他人看到我们程序中的两个屏蔽的接口，容易造成不安全的因素，注释的代码应该删除==。

```python
if __name__ == "__main__":
    if sys.argv[1].startswith('--'):
        option = sys.argv[1][2:]
        if option == "load":
            #安装应用
            LoadCmd(option, sys.argv[2:3][0])
        elif option == 'unload':
            #卸载应用
            UnloadCmd(sys.argv[2:3][0])
        elif option == 'unloadproc':
            #卸载流程
            UnloadProcessCmd(sys.argv[2:3][0])
        else:
            Loginfo("Command %s is unknown"%(sys.argv[1]))
```

## 建议8.3 慎用copy和 deepcopy

**说明：**在python中，对象赋值实际上是对象的引用。当创建一个对象，然后把它赋给另一个变量的时候，python并没有拷贝这个对象，而只是拷贝了这个对象的引用。如果需要拷贝对象，需要使用标准库中的copy模块。copy模块提供copy和deepcopy两个方法：

- copy浅拷贝：拷贝一个对象，但是对象的属性还是引用原来的。对于可变类型，比如列表和字典，只是复制其引用。基于引用所作的改变会影响到被引用对象。
- deepcopy深拷贝：创建一个新的容器对象，包含原有对象元素（引用）全新拷贝的引用。外围和内部元素都拷贝对象本身，而不是引用。

Notes：对于数字，字符串和其他原子类型对象等，没有被拷贝的说法。如果对其重新赋值，也只是新创建一个对象，替换掉旧的而已。==使用copy和deepcopy时，需要了解其使用场景，避免错误使用==。

**示例：**

```
>>> import copy
>>> a = [1, 2, ['x', 'y']]
>>> b = a
>>> c = copy.copy(a)
>>> d = copy.deepcopy(a)
>>> a.append(3)
>>> a[2].append('z')
>>> a.append(['x', 'y'])
>>> print(a)
[1, 2, ['x', 'y', 'z'], 3, ['x', 'y']]
>>> print(b)
[1, 2, ['x', 'y', 'z'], 3, ['x', 'y']]
>>> print(c)
[1, 2, ['x', 'y', 'z']]
>>> print(d)
[1, 2, ['x', 'y']]
```

## 规则8.4 使用os.path库中的方法代替字符串拼接来完成文件系统路径的操作

**说明：**os.path库实现了一系列文件系统路径操作方法，这些方法相比单纯的路径字符串拼接来说更为安全，而且为用户屏蔽了不同操作系统之间的差异。

**错误示例：**如下路径字符串的拼接在Linux操作系统无法使用

```python
path = os.getcwd() + '\\test.txt'
```

**正确示例：**

```python
path = os.path.join(os.getcwd(), 'test.txt')
```

## 建议8.5 使用subprocess模块代替os.system模块来执行shell命令

**说明：**subprocess模块可以生成新进程，连接到它们的input/output/error管道，并获取它们的返回代码。该模块旨在替换os.system等旧模块，相比os.system模块来说更为灵活。

**正确示例：**

```python
>>> subprocess.run(["ls", "-l"])  # doesn't capture output
CompletedProcess(args=['ls', '-l'], returncode=0)

>>> subprocess.run("exit 1", shell=True, check=True)
Traceback (most recent call last):
  ...
subprocess.CalledProcessError: Command 'exit 1' returned non-zero exit status 1

>>> subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
CompletedProcess(args=['ls', '-l', '/dev/null'], returncode=0,
stdout=b'crw-rw-rw- 1 root root 1, 3 Jan 23 16:23 /dev/null\n', stderr=b'')
```

## 建议8.6 建议使用with语句操作文件

**说明**：Python 对一些内建对象进行改进，加入了对上下文管理器的支持，可以用于 with 语句中。使用 with 语句可以自动关闭文件，减少文件读取操作错误的可能性，在代码量和健壮性上更优。==注意 with 语句要求其操作的类型实现"__enter__()"和"__exit__()"方法，需确认实现后再使用==。

**正确示例：**

```python
with open(r'somefileName') as somefile:
    for line in somefile:
        print(line)
        # ...more code
```

此使用with语句的代码等同于以下使用try...finally...结构的代码。

```python
somefile = open(r'somefileName')
try:
    for line in somefile:
        print(line)
        # ...more code
finally:
    somefile.close()
```

在代码量和健壮性上with结构都要优于后者。



# 附录

## 参考资料

1. [https://docs.python.org](https://docs.python.org/)
2. https://www.python.org/dev/peps/pep-0008/
3. https://docs.python.org/3.7/tutorial/controlflow.html#default-argument-values
4. http://google.github.io/styleguide/pyguide.html
5. 《Effective Python：编写高质量Python代码的59个有效方法》
6. 《编写高质量代码：改善Python程序的91个建议》