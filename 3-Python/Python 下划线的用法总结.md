# Python中的下划线的用法总结

平时在看代码的时候，总是看到各种下划线。掌握了下划线的用法无论是对于读代码还是写代码还是有好处的。虽然说网上也有很多的资料讲下划线，但是自己还是要总结一遍加深理解。

### 单划线(_)

#### 在解释器中

在解释器中，`_`代表交互式解释器会话中上一条的执行结果。这种用法有点类似于Linux中的上一条命令的用法。只不过在在Python解释器中表示的上一条执行的结果。

```python
>>> "alright"
'alright'
>>> _
'alright
```



#### 作为一个名称

作为临时性的名称使用，但是在后面不会再次用到该名称。这种用法在循环中会经常用到。

```python
for _ in range(10):
    do_something()
```



#### 作为函数的连接符，仅仅是一种函数名的命名方式，就如同Java的驼峰式的命名法是一样的。

```python
def add_user(user):
	do_something
```



### 名称前的单划线(_get_content)

在Python文档中的解释是：以下划线`_`为前缀的名称(如_get_content)应该被是被API非公开的一部分不管是函数、方法还是属性。此时应该将她们看作一种实现细节，在修改他们的时候无需对外部进行通知。

==当使用了以下划线`_`为前缀的名称，那么在使用`from <模块/包名> import *`时，以`_`开头的名称都不会被导入，除非是模块或包中的`__all__`列表显式地包含了他们==。
由于这种写法仅仅是表示一种细节的实现，在类继承时仍然是可以继承的。

```python
class people(object):
    def _eat(self):
        print('I am eating')

class Student(people):
    @property
    def birth(self):
        return self._brith

    @birth.setter
    def birth(self,value):
        self._birth = value

    @property
    def age(self):
        return self._age
s = Student()
s._eat()    #输出： I am eating
```



### 名称前的双下划线（如:__run）

==名称前带有双下划线的变量，表示的是一个私有函数，无法被继承，也无法在外部访问==。

```python
class People(object):
    def _eat(self):
        print('I am eating')
    def __run(self):
        print('I can run')

class Student(People):
    def torun(self):
        self.__run()		#出错，因为people的方法无法被继承，在Student中不存在__run()方法

s = Student()
s.torun()  
p = People()
p.__run()				#出错，因为私有函数无法在外部访问
```



《Python学习手册》的说明，以双下划线开头的变量名，会自动扩张，从而包含了所在类的名称。例如在Spam类中的**X这样的变量名会自动编程_Spam**X:原始的变量名会在头部加入一个下划线，然后是所在类名称。

```python
dir(People) #输出：_People__run,_eat,...
```



### 名称前后的双下划线(如**init**)

前后有双下划线表示的是特殊函数。通常可以复写这些方法实现自己所需要的功能。最常见的就是复写`__init__`方法。

```python
class People(object):
    def __init__(self, arg):
        super(People, self).__init__()
        self.arg = arg
```