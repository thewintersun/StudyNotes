# 类与继承

[TOC]

## 类的内置方法

```python
	class A:
	    """类A说明
	    """
	    def __new__(self, *agrs, **kwargs):
	        """返回类的实例"""
	        print('run __new__')
	        return super(A, self).__new__(self, *agrs, **kwargs)
	    
	    def __init__(self, start=0, end=5, step=1):
	        """构造方法，创建对象时执行"""
	        print('run __init__')
	        self.start = start
	        self.end = end
	        self.step = step
	    
	    def __del__(self):
	        """析构方法，释放对象的同时执行"""
	        print('run __del__')
	    
	    def __call__(self):
	        """重载括号运算符"""
	        print('run __call__')
	    
	    def __str__(self):
	        """打印对象时的输出该返回值"""
	        return 'run __str__'
	    
	    def __getitem__(self, key):
	        """获取元素时执行"""
	        print('__getitem__',key)
	 
	    def __setitem__(self, key, value):
	        """设置元素时执行"""
	        print('__setitem__',key,value)
	 
	    def __delitem__(self, key):
	        """删除元素时执行"""
	        print('__delitem__',key)
	    
	    def __iter__(self):
	        """
	        iter和for循环均会执行；
	        含有__next__的类已经是一个迭代器，因此可以直接返回self
	        """
	        print("run __iter__")
	        return self
	    
	    def __next__(self):
	        print("run __next__")
	        if self.start >= self.end:
	            raise StopIteration
	        else:            
	            ret = self.start
	            self.start += self.step
	            return ret
```



### `__new__`、 `__init__` 和 `__del__`

new和init这两个方法很容易混淆，平时定义类时，通常使用的都是init方法，很少用到new方法，但他们是有着截然不同的功能的。
new是静态（@staticmethod）方法，用于创建实例对象，方法必须返回一个对象；而init是实例方法，执行实例初始化，在new返回的对象中执行。

1. `__new__`是一个静态方法，而`__init__`是一个实例方法
2. `__new__`方法会返回一个创建的实例，而`__init__`什么都不返回,  如果在init方法中返回值会报错。
3. ==只有在`__new__`返回一个**cls**的实例时，后面的`__init__`才能被调用==
4. 当创建一个新实例时调用`__new__`，初始化一个实例时用`__init__`

**实例1：**

```python
class A(object):
    def __new__(cls):
        print("A 的__new__方法被执行")
        return super().__new__(cls)
 
    def __init__(self):
        print("A 的__init__方法被执行")
 
 
class B(A):
    def __new__(cls):
        print("B 的__new__方法被执行")
        return super().__new__(cls)
 
    def __init__(self):
        print("B 的__init__方法被执行")
 
b = B()

"""
B 的__new__方法被执行
A 的__new__方法被执行
B 的__init__方法被执行
"""
```

如果我们在类的new方法中，返回成其他类型对下，则最终得到的会是新类型。

**实例2：**

```python
class OldObj(object):
    def __new__(cls):
        print("__new__ in <OldObj>")
        return object.__new__(NewObj)
    
    def __init__(self):
        print("__init__ in <OldObj>")
        
class NewObj(object):
    def __init__(self):
        print("__init__ in <NewObj>")
        
obj = OldObj()
print(type(obj))

# 执行输出
>> __new__ in <OldObj>
>> <class '__main__.NewObj'>
```

> 这里有个疑问，为什么new执行之后，==既没执行 OldObj 的init方法，也没执行 NewObj 的init方法==。
> Python Doc里的说明是：If __new__() does not return an instance of cls, then the new instance’s __init__() method will not be invoked.

**实例3：**

```python
# 例1
a = A()  # 先调用__new__返回类的实例，再调用构造方法
a = 1  # a指向其它，使得a的引用计数-1，此时a引用计数为0，被自动释放

"""
输出：
run __new__
run __init__
run __del__
"""

# 例2
a = A()
b = a  # b引用a，使得a的引用计数+1
a = 5  # 这里a指向其他地方，使得a引用计数-1，此时a引用计数为1，未被自动释放

"""
输出：
run __new__
run __init__
"""

b = 5  # 把b也指向其他地方，使得a引用计数-1，此时a引用计数为0，被自动释放

"""
输出：
run __del__
"""

# 例3
a = A()
b = a
a = 5  # 引用计数-1
del b  # 引用计数-1

"""
输出：
run __new__
run __init__
run __del__
"""

# 例4
a = A()
b = a
print(id(a), id(b))
del a
del b  # a和b都是同一个地址，只是b的引用使得a的引用计数+1，因此只会释放一次对象

"""
输出：
run __new__
run __init__
2776419641664 2776419641664
run __del__
"""
```

**实例4：**

```python
class Dog:
    def __del__(self):  # 当内存不需要的时候调用这个删除方法，python解释器自动调用
        print("英雄over")


dog1 = Dog()  # 创建一个对象
dog2 = dog1
del dog1
del dog2
print("==========")

print("----------------------")


class Dog:
    def __del__(self):  # 当内存不需要的时候调用这个删除方法，python解释器自动调用
        print("英雄over")


dog1 = Dog()  # 创建一个对象
dog2 = dog1
del dog1
print("==========")

 
"""
输出：
英雄over
==========
----------------------
==========
英雄over  # 程序运行结束，对象被释放，调用del
"""
```



### `__call__`

```python
a = A()
a()  # 重载括号运算符

"""
输出：
run __new__
run __init__
run __call__
"""
```



### `__str__`

```python
print(a)  # __str__的return是打印对象的输出
```

输出：

```
run __str__
```



### `__getitem__`

```python
1a[1]  # 获取元素1
2a[1:3]  # 切片也是获取元素
```

输出：

```
__getitem__ 1
__getitem__ slice(1, 3, None)
```



### `__setitem__`

```python
a[1] = 2
```

输出：

```
__setitem__ 1 2
```



### `__delitem__`

```python
del a[1]
```

输出：

```
__delitem__ 1
```



### `__iter__`和`__next__`

```python
# for捕捉到StopIteration会自动结束循环
for x in a:
    print(x)
```

输出：

```
run __iter__
run __next__
0
run __next__
1
run __next__
2
run __next__
3
run __next__
4
run __next__
```

### `__enter__`和`__exit__`

enter和exit可以让对象通过with关键字来进行使用，提供进入with块前的初始化工作和退出with块后的清理工作，常用于文件和数据库操作中。

```python
class DbConnect:
    def connect(self):
        print("Init and connect to db.")
    def execute(self):
        print("Execute SQL statement.")
    def disconnect(self):
        print("Disconnect from db.")
    def __enter__(self):
        self.connect()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
with DbConnect() as conn:
    conn.execute()

# 输出
# >> Init and connect to db.
# >> Execute SQL statement.
# >> Disconnect from db.
```



## 类内置属性

系统提供的特殊的属性（内置属性）

| 属性名       | 说明                            |
| :----------- | :------------------------------ |
| `__dict__`   | 查看类或对象成员 , 返回一个字典 |
| `__name__`   | 查看类的名字                    |
| `__doc__`    | 查看类的描述信息 , 即注释部分   |
| `__base__`   | 查看第一个父类                  |
| `__bases__`  | 查看所有父类 , 返回一个元组     |
| `__module__` | 查看类当前所在模块              |
| `__class__`  | 查看对象通过什么类实例化而来    |



```python
a.__doc__  # 类说明文档

"""
输出：
'类A说明\n    '
"""
```

```python
a.__module__  # 当前操作的对象在那个模块

"""
输出：
'__main__'
"""
```

```python
a.__class__  # 当前操作的对象的类是什么

"""
输出：
__main__.A
"""
```

```python
a.__dict__  # 类或对象中所有成员

"""
输出：
{'start': 5, 'end': 5, 'step': 1}
"""
```



### 类变量 + `__new__`实现单例

```python
1class A(object):
2    # 类变量，记录类的第一个实例地址
3    _address = None
4 
5    def __new__(cls):
6        if A._address:
7            print("run __new__, exists")
8            return A._address
9        else:
10            print("run __new__, new")
11            return super(A, cls).__new__(cls)
12 
13    def __init__(self):
14        print("run __init__\n")
15        A._address = self
16 
17a1 = A()
18a2 = A()
19a3 = A()
```

输出：

```
run __new__, new
run __init__

run __new__, exists
run __init__

run __new__, exists
run __init__
```

```python
a1, a2, a3  # 地址一样，验证了单例
```

输出：

```
(<__main__.A at 0x2866f8135f8>,
 <__main__.A at 0x2866f8135f8>,
 <__main__.A at 0x2866f8135f8>)
```



## 类变量

【类名.类变量名】

> 1、类变量在class内，但不在class的任何方法内，存在类的内存里
>
> 2、类变量是该类所有实例共享的变量，但是实例对象只能访问，不可修改，每个实例对象去访问同一个类变量都将得到相同结果【实例名.类变量名】
>
> 3、新增、修改、删除类变量n，不会影响到同名实例变量n
>
> 4、类无权访问实例变量
>
> 5、类变量可修改、新增、删除

Python类变量被赋值

   （1）类的设计里： class里def外，通过变量名能被赋值，2 行；def 里通过类对象即类名字的点运算变量名可被赋值，4行。

   （2）程序里：通过类对象(类名字)的点运算类名字也可被赋值，11行。

```python
class Test:
    w = 10   # 类变量
    def __init__(self):
        Test.h=50

a = Test()
print (a.w)   # 通过实例访问类变量 输出10

a.w = 100     # 类变量并没有修改，而是添加了一个实例变量w，a.w=a.w+100是可以的
print (a.w)  # 通过实例访问实例变量, 已经屏蔽同名类变量 输出100
Test.w = 20  # 通过类名修改类变量
print (Test.w, Test.h)  #输出20 50
Test.t = 15     # 动态添加类变量
print(a.t)      # 通过实例访问类变量t  输出 15
del Test.t      # 删除类变量t
```



```python
class A: 
	x = 0  
 
	def func(self): 
		print(self.x) 
 
a = A() 
a.x = 0 

print('a.func()') 
a.func() # 0 
print('a.func(a)')   
# a.func(a) #TypeError: func() takes 1 positional argument but 2 were given 
print('A.func(a)') 
A.func(a) # 0 
print('A.func(A)') 
A.func(A) # 0
```



### 下划线变量

–     _xxx "单下划线 " 开始的成员变量叫做**保护变量**，意思是==只有类实例和子类实例能访问到这些变量==， 需通过类提供的接口进行访问；不能用 ’from module import *' 导入

–     __xxx 类中的**私有变量**/方法名 （Python的函数也是对象，所以成员方法称为成员变量也行得通。）, " 双下划线 " 开始的是私有成员，意思是==只有类对象自己能访问，连子类对象也不能访问到这个数据==。

–   `  __xxx__ `系统定义名字，前后均有一个“双下划线” 代表python里特殊方法专用的标识，如 `__init__()`代表类的初始化函数。

私有属性：双下划线不能外部使用或直接访问（因为改名，不同解释器可能改名不同），_ 单下划线可以

**实例：**

```python
class MyClass:
    _abc = 1

myobj1 = MyClass()
print(myobj1._abc)
myobj1._abc = 3
print(myobj1._abc)

"""
结果：
1 
3
"""

class Demo:
    def _check(self):
        print('Demo')

    def display(self):
        print(self._check())

class DeriverDemo(Demo):
    def _check(self):
        print('DeriverDemo')

Demo().display()
Demo()._check()
DeriverDemo().display()

"""
结果：
Demo 
None 
DeriverDemo 
None
"""

```

**实例：**

```python
class Student(object):

    def __init__(self, name, score):
        self.__name = name
        self.__score = score
        
    def get_score(self):
        return self.__score
    
    def set_score(self, score):
        self.__score = score
        
bart.__name = 'New Name'  # 改的是其他属性

```



### `__slots__`

**实例：**

```python
class Person: 
	__slots__ = ('name', 'age') 
 
person = Person() 
person.name = 'John' 
person.name = 'Kate' 
person.nationality = 'CN' 
person.nationality = 'UK' 
print(f"{person.name} is from {person.nationality}") 

""" 
Traceback (most recent call last): 
  File "F:\python\科目二\code\0110_test.py", line 116, in <module> 
    person.nationality = 'CN' 
AttributeError: 'Person' object has no attribute 'nationality' 
 
"""
```





### 实例变量

实例变量（实例属性）【实例名.实例变量名】

> 1、实例变量是构造函数下的变量带self.变量
>
> 2、实例变量为每个实例本身独有，不可相互调用、新增、修改、删除，不可被类调用、新增、修改、删除
>
> 3、==可以访问类变量==
>
> 4、如果同时有类变量和实例变量，程序执行时，==先访问实例变量，实例变量存在，会使用实例变量，实例变量不存在，会使用类变量==
>
> 5、实例修改类变量，不可修改，实际是在实例内存里创建了实例变量
>
> 6、新增、修改、删除实例变量n，不会影响到类变量n
>
> 7、a实例不能调用b实例的变量
>
> 8、实例变量可修改、新增、删除

Python实例对象变量被赋值
（1）类的设计里
def里通过self点运算变量名能被赋值，不一定非在`__init__`里，其他已被调用的方法函数里也行
（2）程序里
通过实例对象的点运算变量名可被赋值

```python
>>> class Student:              
		# 类的定义体    
    	classroom = '101'   # 类变量    
        address = 'beijing' # 类变量
        
    	def __init__(self, name, age):        
            self.name = name      #实例变量        
            self.age = age        #实例变量    
        
        def print_age(self):        
            print('%s: %s' % (self.name, self.age))
            
>>> li = Student("李四", 24)        # 创建一个实例
>>> zhang = Student("张三", 23)     # 创建第二个实例
>>> li.classroom # li本身没有classroom实例变量，所以去寻找类变量，它找到了！'101'
>>> zhang.classroom # 与li同理'101'
>>> Student.classroom   # 通过类名访问类变量'101'
>>> li.classroom = '102'    # 关键的一步！实际是为li创建了独有的实例变量，只不过名字和类变量一样，都叫做classroom。
>>> li.classroom    # 再次访问的时候，访问到的是li自己的实例变量classroom'102'
>>> zhang.classroom # zhang没有实例变量classroom，依然访问类变量classroom'101'
>>> Student.classroom   # 保持不变'101'
>>> del li.classroom    # 删除了li的实例变量classroom
>>> li.classroom        # 一切恢复了原样'101'
>>> zhang.classroom  # '101'
```



## 类装饰器

### 属性装饰器@property

@property实例化后使用del删除属性，再赋值，输出的返回结果?报错???

==property 删除某个函数 再次赋值不能成功==

```python
class Student():
   def __init__(self):
     self._score = 10000
 
   @property
   def score(self):
     return self._score
 
   @score.setter
   def score(self, value):
     if value < 0:
       print(11111)
       return
     self._score = value
 
   @score.deleter
   def score(self):
     del self._score
        
 a = Student()
 del a.score
 a.score = -1
 print(a.score)
 """
 11111
 Traceback (most recent call last):
  File "E:\学习\Python\科目二\Python科目二考试\代码\test_run.py", line 25, in <module>
   print(a.score)
  File "E:\学习\Python\科目二\Python科目二考试\代码\test_run.py", line 7, in score
   return self._score
 AttributeError: 'Student' object has no attribute '_score'
 """
```

 

```python
class Person(object): 

   def __init__(self, birth_year): 
    self._birth_year = birth_year 


   @property 
   def birth_year(self): 
    return self._birth_year 
    
   @birth_year.setter 
   def birth_year(self, value): 
    self._birth_year = value 

   @property 
   def age(self): 
    print("age return:",2021 - self._birth_year) 
    return 2021 - self._birth_year 

 person = Person(1990) 
 person.birth_year = 2000 
 A = person.age 

 print("A", A)     
 print(person.__dict__) 
 print(person.birth_year) 
 """ 
 age return: 21 
 A 21 
 {'_birth_year': 2000} 
 2000 
 """ 

print("B")
person.age = 11
print("B:", person.__dict__)
print("B:", person.birth_year)  # 2000
"""
B
Traceback (most recent call last):
  File "D:/Scripts/LeetCode/科目二/test.py", line 219, in <module>
    person.age = 11
AttributeError: can't set attribute
"""
```



### 静态方法@staticmethod

- 采用`@classmethod`定义的方法是类方法，类方法第一个参数为类，可通过实例或者类调用类方法
- 使用`@staticmethod`定义的方法叫做静态方法，==默认无第一个参数==

|               |     实例方法      |      类方法      |     静态方法      |
| :-----------: | :---------------: | :--------------: | :---------------: |
| 实例调用a=A() | a.normalMethod(x) | a.classMethod(x) | a.staticMethod(x) |
|    类调用     |     不能调用      | A.classMethod(x) | A.staticMethod(x) |

- 实例方法：随着实例属性的改变而改变
- 类方法（无论是类调用还是实例调用）：都是类属性的值，不随实例属性的变化而变化
- 静态方法：不可以访问类属性，故直接输出传入方法的值

> 当被 @staticmethod 装饰时，==函数作用域仅在当前类中==，==子类中的类变量不会受影响==
>
> 当被 @classmethod 装饰时，函数相当于被借调到子类中作为一个独立方法，因此影响到类变量的值

### 类方法@classmethod 

静态方法需要访问基类或派生类的成员，其`cls`传入调用者的最底层类。

```python
class Spam:
    num_instances = 0

    @classmethod
    def count(cls):               # 对每个类做独立计数
        cls.num_instances += 1    # cls是实例所属于的最底层类

    def __init__(self):
        self.count()              # 将self.__class__传给count方法
 
class Sub(Spam):
    num_instances = 0
 
class Other(Spam):
    num_instances = 0
 
x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()
x.num_instances, y1.num_instances, z1.num_instances           # (1, 2, 3)
Spam.num_instances, Sub.num_instances, Other.num_instances    # (1, 2, 3)
```

注意和`@staticmethod`区别开

```python
class Spam:
    num_instances = 0
    
    @staticmethod
    def count(cls):               
        cls.num_instances += 1    # 函数作用域仅在当前类中 Spam.__class__
        
    def __init__(self):
        self.count()              
 
class Sub(Spam):
    num_instances = 0
 
class Other(Spam):
    num_instances = 0
 
x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()
x.num_instances, y1.num_instances, z1.num_instances           # (6, 0, 0)
Spam.num_instances, Sub.num_instances, Other.num_instances    # (6, 0, 0)
```

```python
class Spam: 
    num_instances = 0 
    @classmethod 
    def count(cls):  # 对每个类做独立计数 
        cls.num_instances += 1  # cls是实例所属于的最底层类 
 
 
    def __init__(self): 
        self.count()  # 将self.__class__传给count方法 
 
 
class Sub(Spam): 
    pass  # 没有定义 num_instances 
 
class Other(Spam): 
    pass 
 
x = Spam() 
y1, y2 = Sub(), Sub() 
z1, z2, z3 = Other(), Other(), Other() 
 
print(x.num_instances, y1.num_instances, z1.num_instances)  # 输出：(1, 3, 4) 
print(Spam.num_instances, Sub.num_instances, Other.num_instances)  # 输出：(1, 3, 4)

```



## 类继承

Python中，一个类可以通过继承的方式来获得父类中的非私有属性和非私有方法。

继承的语法为在类名后的小括号()中写入要继承的父类名，如果要继承多个类则中间用逗号分隔。

> 1.父类的非私有属性和非私有方法，子类可以直接继承，子类对象可以直接使用。如果子类要调用父类的私有属性和私有方法，只能通过间接的方法来获取。
>
> 2.子类可以实现父类没有的属性和方法，与继承的属性和方法互不干扰。
>
> 3.如果在子类中有跟父类同名的方法，但方法中执行的内容不同，则子类可以重写父类方法。
>
> 当子类实现一个和父类同名的方法时，叫做重写父类方法。直接在子类中定义与父类同名的方法，然后在方法中实现子类的业务逻辑，子类方法就会覆盖父类的同名方法。子类重写了父类方法，子类再调用该方法将不会执行父类的方法。
>
> 4.如果在子类重写父类的方法中，需要使用父类同名方法中的功能，在父类功能基础上做扩展，则子类可以在重写的方法中调用父类被重写的方法，使用super()来调用。



### 类的多层继承

类可以多层继承。

继续上面的类，我们定义的类Mi继承自Phone类，Phone类继承自Electrical类，这样就构成了多层继承。

Mi类对象可以使用Phone中的方法和属性，也可以使用Electrical中的方法和属性，如果Phone重写了Electrical的方法，则继承的是Phone中的方法。

当Mi类对象调用属性和方法时，优先在自己的内部查找是否有该属性和方法，如果没有会到它的父类Phone中查找该属性和方法，如果没有会继续往上在Phone的父类Electrical中查找，一直查找到object类中。到最后也没有找到，就说明对象没有该属性和方法，程序报错，如果找到就会返回找到的属性和方法，不再继续往上查找。

同一个类可以继承多个类

**MRO：** method resolution order，主要用于在多继承时判断 方法、属性的调用路径；

**super(): ** 调用父类，这个好理解；

解决多重继承问题，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用等种种问题。

```python
class GrandParent: 
	def __init__(self): 
		print("Hello, GrandParent") 
 
class Parent(GrandParent): 
	def __init__(self): 
		super().__init__() 
		print("Hello, Parent") 
 
class Child(GrandParent, Parent): 
	def __init__(self): 
		super().__init__() 
		print("Hello, Child") 
 
if __name__ == '__main__': 
	c = Child() 
 
# Traceback (most recent call last): 
#   File "F:\python\科目二\code\0110_test.py", line 62, in <module> 
#     class Child(GrandParent, Parent): 
# TypeError: Cannot create a consistent method resolution 
# order (MRO) for bases GrandParent, Parent

```



```python
class GrandParent:
    def __init__(self):
        print("Hello, GrandParent")


class Parent1(GrandParent):
    def __init__(self):
        super().__init__()
        print("Hello, Parent1")


class Parent2(GrandParent):
    def __init__(self):
        super().__init__()
        print("Hello, Parent2")


class Child(Parent1, Parent2):
    def __init__(self):
        super().__init__()
        print("Hello, Child")


if __name__ == '__main__':
    c = Child()

"""
Hello, GrandParent
Hello, Parent2
Hello, Parent1
Hello, Child
"""
```

super在多继承中按照 MRO 顺序逐个调用，==在多继承中避免 ‘类名调用’ 带来的父类方法重复执行问题==。

**实例: **

1、类名.方法名调用

```python
class Parent(object):
    def __init__(self, name):
        print('parent的init开始被调用')
        self.name = name
        print('parent的init结束被调用')


class Son1(Parent):
    def __init__(self, name, age):
        print('Son1的init开始被调用')
        self.age = age
        Parent.__init__(self, name)
        print('Son1的init结束被调用')


class Son2(Parent):
    def __init__(self, name, gender):
        print('Son2的init开始被调用')
        self.gender = gender
        Parent.__init__(self, name)  #
        print('Son2的init结束被调用')


class Grandson(Son1, Son2):
    def __init__(self, name, age, gender):
        print('Grandson的init开始被调用')
        Son1.__init__(self, name, age)
        Son2.__init__(self, name, gender)
        print('Grandson的init结束被调用')


gs = Grandson('grandson', 12, '男')
print('姓名：', gs.name)
print('年龄：', gs.age)
print('性别：', gs.gender)

# 执行结果
# Grandson的init开始被调用
# Son1的init开始被调用
# parent的init开始被调用
# parent的init结束被调用
# Son1的init结束被调用
# Son2的init开始被调用
# parent的init开始被调用
# parent的init结束被调用
# Son2的init结束被调用
# Grandson的init结束被调用
# 姓名： grandson
# 年龄： 12
# 性别： 男
```

2、super调用

```python
class Parent(object):
    def __init__(self, name, *args, **kwargs):
        print('parent的init开始被调用')
        self.name = name
        print('parent的init结束被调用')


class Son1(Parent):
    def __init__(self, name, age, *args, **kwargs):
        print('Son1的init开始被调用')
        self.age = age
        super().__init__(name, *args, **kwargs)
        print('Son1的init结束被调用')


class Son2(Parent):
    def __init__(self, name, gender, *args, **kwargs):
        print('Son2的init开始被调用')
        self.gender = gender
        super().__init__(name, *args, **kwargs)
        print('Son2的init结束被调用')


class Grandson(Son1, Son2):
    def __init__(self, name, age, gender):
        print('Grandson的init开始被调用')
        super().__init__(name, age, gender)
        print('Grandson的init结束被调用')


print(Grandson.__mro__)

gs = Grandson('grandson', 12, '男')

print('姓名：', gs.name)
print('年龄：', gs.age)
print('性别：', gs.gender)

# 执行结果
# (<class '__main__.Grandson'>, <class '__main__.Son1'>, <class '__main__.Son2'>, <class '__main__.Parent'>, <class 'object'>)

# Grandson的init开始被调用
# Son1的init开始被调用
# Son2的init开始被调用
# parent的init开始被调用
# parent的init结束被调用
# Son2的init结束被调用
# Son1的init结束被调用
# Grandson的init结束被调用
# 姓名： grandson
# 年龄： 12
# 性别： 男
```

对比两者运行结果可以看到==‘类名.方法名’方式在运行过程中对父级方法造成了重复执行==。

### Mixin模式

Mixin模式是一种在python里经常使用的模式，适当合理的应用能够达到复用代码，合理组织代码结构的目的。下面是标准库socketserver中对Mixin的使用：

```python
if hasattr(os, "fork"): 
   class ForkingUDPServer(ForkingMixIn, UDPServer): pass 
   class ForkingTCPServer(ForkingMixIn, TCPServer): pass 
  
 class ThreadingUDPServer(ThreadingMixIn, UDPServer): pass 
 class ThreadingTCPServer(ThreadingMixIn, TCPServer): pass
```

Mixin是一个行为的集合，是受限制的多重继承，有点类似Java中的接口，其特点是:

•      Mixin 类是单一职责的

•      Mixin 类对宿主类一无所知

•      不存在超类方法调用(super)以避免引入 MRO 查找顺序问题

```python
class ListMetaclass(type): 
   def __new__(cls, name, bases, attrs): 
    print(name) 
    print(bases) 
    attrs['add'] = lambda self, value: self.append(value) 
    return type.__new__(cls, name, bases, attrs) 
 
 
 class DefinedList(list, metaclass=ListMetaclass): 
   pass 
 
 definedList = DefinedList() 
 definedList.add(1) 
    
 """ 
 DefinedList 
 (<class 'list'>,) 
 """
```



### 类继承变量



```python
# Python3.X 
class Test0(object): 
    x = “1” 

class Test1(Test0): 
    pass 

class Test2(Test0): 
    pass 

if __name__ == “__main__”: 
    Test1.x = “2” 
    print(“%s, %s, %s”%(Test0.x, Test1.x, Test2.x) 
    Test0.x = “3” 
    print(“%s, %s, %s”%(Test0.x, Test1.x, Test2.x)

"""
1, 2, 1
3, 2, 3
"""
```

**实例：**

```python
class Foo:
    count = 0

    def __init__(self):
        self.count = 0

    def incr_one(self):
        self.count += 1

    @staticmethod
    def incr_two():
        Foo.count += 1

    @classmethod
    def incr_three(cls):
        cls.count += 1


class Bar(Foo):
    pass


foo = Foo()
bar = Bar()
foo.incr_one()  # foo.count = 1
bar.incr_one()  # bar.count = 1
foo.incr_two()  # Foo.count = 1
bar.incr_three()  # Bar.count = 2
Foo.incr_two()  # Foo.count = 2
Bar.incr_three()  # Bar.count = 3
Foo.incr_three()  # Foo.count = 3

print(Foo.count)  # 3
print(Bar.count)  # 3
print(foo.count)  # 1
print(bar.count)  # 1
```



### 元类

关于元类，这列描述，正确的是: （ A\B\C\D ) 
A. 元类是类的类，常可以用在类工厂中
B. Python 中所有的类都是对象，可以通过 type() 来创建元类
C. 在定义类时，可以通过 metaclass 参数来指定此类的元类
D. 从设计的复杂度来讲，尽量少用元类，多用普通类或函数