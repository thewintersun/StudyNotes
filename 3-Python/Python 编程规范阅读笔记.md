# 1. 排版

## 规则1.4 Python文件必须使用UTF-8编码

**说明：**Python文件必须使用UTF-8编码，文件头可以添加编码声明`# coding: utf-8`

## 规则1.6 相对独立的程序块之间、变量说明之后必须加空行

**说明：**相对独立的程序块之间、变量说明之后加上空行，代码可理解性会增强很多。

## 建议1.7 一行长度小于80个字符，与Python标准库看齐

**说明：**建议开发团队用本产品线的门禁工具或者 yapf 自动格式化，或者用IDE自带的格式化功能统一格式化代码后再提交。

较长的语句、表达式或参数（>80字符）要分成多行书写，==首选使用括号（包括{},[],()）内的行延续==，推荐使用反斜杠（\）进行断行。==长表达式要在低优先级操作符处划分新行==，操作符统一放在新行行首或原行行尾，划分出的新行要进行适当的缩进，使排版整齐，语句可读。

## 规则1.8 在两个以上的关键字、变量、常量进行对等操作时，它们之间的操作符前后要加空格

1. 逗号、分号（假如用到的话）只在后面加空格。
2. 比较操作符 ">"、">="、"<"、"<="、"=="，赋值操作符"="、"+="，算术操作符 "+"、"-"、"%"，逻辑操作符 "and"、"or"  等双目操作符的前后加空格。
3. "*"、"**" 等作为操作符时，前后可以加空格，但==若和更低优先级的操作符同时使用并且不涉及括号，则建议前后不加空格==。

1. 正确示例：

   ```python
   a = b * c
   a = c ** b
   x = x*2 - 1
   ```

## 规则1.10 加载模块必须分开每个模块占一行

**说明：**单独使用一行来加载模块，让程序依赖变得更清晰。

注意虽然一行只能加载一个模块，但同一个模块内的多个符号可以在同一行加载。

## 规则1.11 导入部分(imports)置于模块注释和文档字符串之后，模块全局变量和常量声明之前

**说明：**导入(import)库时，==按照标准库、第三方关联库、本地特定的库/程序==顺序导入，并==在这几组导入语句之间增加一个空行==。

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

**说明**：from xxx import *会将其他模块中的所有成员挨个赋值给当前范围的同名变量，如果当前范围已经有同名变量，则会静默将其覆盖。这种方式容易导致名字冲突，且冲突后不容易定位，应当尽量避免使用。

## 建议1.13 类Unix操作系统上直接执行的Python文件头部建议使用#!/usr/bin/env python指定解释器

**说明：**类Unix操作系统上使用Hashbang “#!/usr/bin/env python”声明的时候，会去取系统的 PATH 变量中指定的第一个 Python 来执行你的脚本，有助于正确指定执行Python文件的解释器。Hashbang的位置需要放在文件编码声明之前。Windows操作系统可忽略此建议。



# 2. 注释

注释不宜太多也不能太少，==一般建议建议有效注释量（包括文档字符串）应该在20%以上。==
撰写好的注释有以下建议：

- 注释描述必须准确、易懂、简洁，不能有二义性；
- ==避免在注释和文档字符串中使用缩写==，如果要使用缩写则需要有必要的说明；
- 修改代码时始终==优先更新相应的注释/文档字符串==，以保证注释/文档字符串与代码的一致性；
- 有含义的变量，如果不能充分自注释，则需要添加必要的注释；
- 全局变量建议添加详细注释，包括对其==功能、取值范围、哪些函数或过程修改它以及存取时注意事项==等的说明。

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

##  规则2.4 模块文档字符串写在文件的顶部，导入(import)部分之前的位置，不需要缩进

**说明：**模块文档字符串应当包含功能描述和版权声明。

**正确示例：**

```python
"""
功 能：XXX类，该类主要涉及XXX功能
版权信息：华为技术有限公司，版本所有(C) 2010-2019
"""
```

## 规则2.5 文档字符串多于一行时，末尾的"""要自成一行

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



# 3.命名

## 命名规范推荐表

| **Type**                   | **Public**           | **Internal**                                                 |
| :------------------------- | :------------------- | :----------------------------------------------------------- |
| Modules                    | ==lower_with_under== | _lower_with_under                                            |
| Packages                   | ==lower_with_under== |                                                              |
| Classes                    | ==CapWords==         |                                                              |
| Exceptions                 | ==CapWords==         |                                                              |
| Functions                  | lower_with_under()   | _lower_with_under()                                          |
| Global/Class Constants     | ==CAPS_WITH_UNDER==  | _CAPS_WITH_UNDER                                             |
| Global/Class Variables     | ==lower_with_under== | lower_with_under                                             |
| Instance Variables         | ==lower_with_under== | _lower_with_under or  __lower_with_under ==(当需要名字修饰时)== |
| Method Names               | lower_with_under()   | _lower_with_under() or __lower_with_under() (当需要名字修饰时) |
| Function/Method Parameters | ==lower_with_under== |                                                              |
| Local Variables            | ==lower_with_under== |                                                              |

## 规则3.1 包（Package）、模块（Module）名使用意义完整的英文描述，采用小写加下划线（lower_with_under）的风格命名

**说明:**模块应该用小写加下划线的方式（如lower_with_under.py）命名。==尽管已经有很多现存的模块使用类似于CapWords.py这样的命名,但现在已经不鼓励这样做, 因为如果模块名碰巧和类名一致, 这会让人困扰==。

**正确示例：**

```python
from sample_package import sample_module
from sample_module import SampleClass
```

## 规则3.5 类或对象的私有成员一般用单下划线_开头；对于需要被继承的基类成员，如果想要防止与派生类成员重名，可用双下划线__开头。

**说明**：
Python没有严格的私有权限控制，业界约定俗成的用单下划线“_”开头来暗示此成员仅供内部使用。==双下划线“__”开头的成员会被解释器自动改名，加上类名作为前缀，其作用是防止在类继承场景中出现名字冲突，并不具有权限控制的作用，外部仍然可以访问==。双下划线开头的成员应当只在需要避免名字冲突的场景中使用（比如设计为被继承的工具基类）。

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
        self.__update(iterable)    # 双下划线开头，会被解释器改名为_Mapping__update。
        						   # 外部如果使用修改后的名字仍可访问

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

对于允许使用单个字符命名的场景，==不要用字符"l"，"o"来做变量名称==。在有些字体中，这些字符于数字很难1和0很难辨认。若确实需要使用"l"做变量，用"L"来替换。

# 4. 编码

## 规则4.1 与None作比较要使用“is”或“is not”，不要使用等号

**说明：**

==“is”判断是否指向同一个对象（判断两个对象的id是否相等）==，“==”会调用**eq**方法判断是否等价（判断两个对象的值是否相等）。

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

## 建议4.3 避免直接使用dict[key]的方式从字典中获取value，如果一定要使用，需要注意当key not in dict时的异常捕获和处理

**说明：**

Python的字典dict可以使用key获取其对应的value。但是当key在dict的key值列表中不存在时，直接使用dict[key]获取value会报KeyError，==应当使用更为安全的dict.get(key)类型方法获取value==。

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

如下写法，==在start : end : stride都使用的情况下使用负的stride，会造成阅读困难==。此种情况建议将“步进”切割过程和“范围”切割过程分开，使代码更清晰。

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
==使用其他手段（比如标准库中的 itertools 工具）来简化代码==。

```python
import itertools

lengths = [1, 2]
widths = [3, 4]
heights = [5, 6]
cubes = list(itertools.product(lengths, widths, heights))
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
Python的函数/类定义和C语言不同，函数/类定义语句实际上是给一个名字赋值。==因此重复定义一个函数/类的名字不会导致错误，后定义的会覆盖前面的==。但是重复定义很容易掩盖编码问题，让同一个名字的函数/类在不同的执行阶段具有不同的含义，不利于可读性，应予以禁止。

==Python在解析一个被引用的名字时遵循LEGB顺序（Local - Enclosed - Global - Builtin）==，从内层一直查找到外层。内层定义的变量会覆盖外层的同名变量。在代码修改时，同名的变量容易导致错误的引用，也不利于代码可读性，应当尽量避免。

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
在Python 3.X版本中，允许直接定义不含self参数的方法，并且允许不通过实例调用。但是一旦通过实例调用这个方法，就会因为参数不匹配而出错。加上@staticmethod进行修饰，可以让Python解释器明确此方法不需要self参数，提前拦截问题，可读性也更好。

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

2、==方法不需要访问实例的成员，但需要访问基类或派生类的成员。这时应当用@classmethod装饰。==装饰后的方法，其第一个参数不再传入实例，而是传入调用者的最底层类。
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

==但是使用@classmethod时需要注意，由于在继承场景下传入的第一个参数并不一定是这个类本身，因此并非所有访问类成员的场景都应该用@classmethod。== 比如下面这个例子中，Base显式的想要修改自己的成员inited（而不是派生类的成员），这时应当用@staticmethod。

**错误示例**：

```python
class Base:
    inited = False
    @classmethod
    def set_inited(cls):     # 实际可能传入Derived类
        cls.inited = True    # 并没有修改Base.inited，而是给Derived添加了成员

class Derived(Base):
    pass

x = Derived()
x.set_inited()
if Base.inited:
    print("Base is inited")   # 不会被执行
```

## 建议4.12 避免在代码中修改sys.path列表

**说明**：
sys.path是Python解释器在执行import和from语句时参考的模块搜索路径，==由当前目录、系统环境变量、库目录、.pth文件配置组合拼装而成==。用户通过修改系统配置，可以指定搜索哪个路径下的模块。sys.path只应该根据用户的系统配置来生成，不应该在代码里面直接修改。否则可能出现A模块修改了sys.path，导致B模块搜索出错，且用户难以定位。

## 建议4.13 尽量不使用for i in range(x)的方式循环处理集合数据，而应使用for x in iterable的方式

**说明**：
for i in range(x)，然后在循环体内对集合用下标[i]获取元素是C语言的编程习惯，==它有很多缺点：容易越界；在循环体内修改i容易出错；可读性差。==Python语言建议尽量用for x in iterable的方式直接取集合的每一条数据进行处理。

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



# 5. 异常处理

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

==在异常这方面, Python非常宽容,“except:”语句真的会捕获包括Python语法错误在内的任何错误。== 使用“except:”很容易隐藏真正的bug，我们在使用try…except…结构对代码作保护时，应该明确期望处理的异常。

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

except (FileNotFoundError, IOError) as file_open_except:
    print(file_open_except)
```

## 规则5.3 不在except分支里面的raise都必须带异常

**说明：**==raise关键字单独使用只能出现在try-except语句中，重新抛出except抓住的异常==。

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

## 建议5.4 尽量用异常来表示特殊情况，而不要返回None

当我们在一个工具方法时，通常会返回None来表明特殊的意义，比如一个数除以另外一个数，如果被除数为零，那么就返回None来表明是没有结果的。
(1)==方法使用None作为特殊含义做为返回值是非常糟糕的编码方式==，因为None和其它的返回值必须要添加额外的检查代码。
(2) 触发异常来标示特殊情况，调用者会在捕获异常来处理。

## 建议5.5 避免finally中可能发生的陷阱，不要在finally中使用return或者break语句

通常使用finally语句，表明要释放一些资源，这时候try和except还有else代码块都被执行过了，如果在执行它们的过程中有异常触发，且没有处理这个异常，那么异常会被暂存，==当finally代码执行后，异常会重新触发，但是当finally代码块里有**return**或**break**语句时，这个暂存的异常就会丢弃==：

```python
def f():
    try:
        1/0
    finally:
        return 42

print(f())
```

上面的代码执行完后1/0产生的异常就会被忽略，最终输出42，==因此在finally里出现return是不可取的==。

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

## 规则5.6 禁止使用except X, x语法，应当使用except X as x

**说明**：
except X, x语法只在2.X版本支持，3.X版本不支持，有兼容性问题。而且，except X, x写法容易和多异常捕获的元组（tuple）表达式混淆。==因此应该统一用except X as x方式==。

## 建议5.7 assert语句通常只在测试代码中使用，禁止在生产版本中包含assert功能

**assert**只应在研发过程中内部测试时使用，出现了**AssertionError**异常说明存在软件设计或者编码上的错误，应当修改软件予以解决。在对外发布的生产版本中禁止包含**assert**功能。

## 多线程适用于阻塞式IO场景，不适用于并行计算场景

综上，Python的标准实现CPython，由于GIL的存在，同一个时刻只能运行一个线程，无法充分利用多CPU提升运算效率，因此Python的多线程适用于阻塞式IO的场景，不适用于并行计算的场景。

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

说明：==list是动态array，而tuple是静态array（其成员个数以及内容皆不可变）==。因此，list需要更多的内存来跟踪其成员的状态。

此外，对于成员个数小于等于20的tuple，Python会对其进行缓存，即当此tuple不再使用时，Python并不会立即将其占用的内存返还给操作系统，而是保留以备后用。

错误示例：

```python
myenum = [1, 2, 3, 4, 5]
```

正确示例：

```python
myenum = (1, 2, 3, 4, 5)  # 如果数据恰好被缓存过，则初始化速度会为错误示例中的5倍以上。
```

## 建议7.3 尽量使用generator comprehension 代替 list comprehension

说明：list comprehension 可以用来代替 lambda 表达式的 map、reduce 语法，从已有的list中，生成新的数据。而generator comprehension 无需定义一个包含yield语句的函数，就可以生成一个generator。
==二者一个生成list，另外一个生成generator，在内存的占用上，相差悬殊；在生成速度上，相差无几==。

错误示例：

```python
even_cnt = len([x for x in range(10) if x % 2 == 0])
```

正确示例：

```python
even_cnt = sum(1 for x in range(10) if x % 2 == 0)
```

## 建议7.4 在循环中，使用format方法、"%"操作符 和 join方法代替"+"和"+="操作符来完成字符串格式化

**说明：**即使参数都是字符串，也可以使用format方法或%运算符来格式化字符串。一般性能要求的场景可以使用+或+=运算符，但需要避免使用+和+=运算符在循环中累积字符串。==由于字符串是不可变的，因此会产生不必要的临时对象并导致二次而非线性运行时间==。

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
employee_table = ''.join(items) # join 
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

## 规则8.1 函数参数中的可变参数不要使用默认值，在定义时使用None

**说明：** ==参数的默认值会在方法定义被执行时就已经设定了，这就意味着默认值只会被设定一次，当函数定义后，每次被调用时都会有"预计算"的过程==。

当参数的默认值是一个可变的对象时，就显得尤为重要，例如参数值是一个list或dict，如果方法体修改这个值(例如往list里追加数据)，那么这个修改就会影响到下一次调用这个方法，这显然不是一种好的方式。应对种情况的方式是将参数的默认值设定为None。

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

## 规则8.2 严禁使用注释行等形式仅使功能失效

**说明：**python的注释包含：单行注释、多行注释、代码间注释、文档字符串等。除了文档字符串是使用""""""括起来的多行注释，常用来描述类或者函数的用法、功能、参数、返回等信息外，其余形式注释都是使用#符号开头用来注释掉#后面的内容。

基于python语言运行时编译的特殊性，如果在提供代码的时候提供的是py文件，即便是某些函数和方法在代码中进行了注释，==别有用心的人依然可以通过修改注释来使某些功能启用==；尤其是某些接口函数，如果不在代码中进行彻底删除，很可能在不知情的情况下就被启用了某些本应被屏蔽的功能。

因此根据红线要求，==在python中不使用的功能、模块、函数、变量等一定要在代码中彻底删除==，不给安全留下隐患。即便是不提供源码py文件，提供编译过的pyc、pyo文件，别有用心的人可以通过反编译来获取源代码，可能会造成不可预测的结果。

**总结：不能用#来注释无用的代码，而是一定要删除。**

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

