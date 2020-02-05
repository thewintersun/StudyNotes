### Java编程基础考试考点

### 1.  命题范围

#### 1.1      针对 JDK1.4 及其类库进行命题

虽然是针对 JDK1.4 ，但命题时需要确保题目与 JDK1.5、JDK1.6 等后续版本兼容。

#### 1.2      考点设计原则

##### 1.2.1        确保各部门常用的、常错的和必须了解的内容进行设计。

各部门确保常用、常错的地方被列入考点，同时确保本部门不用的内容不被列入考点。学员掌握了这些考点以后，基本上掌握 Java 语言语法、懂得使用常用的 Java 类库，这是考点设计最重要的原则。由于学员基本上都了解 Java 语法，编译器与IDE在语法方面也可以提供很多帮助，所以重点还在常用类库上面。

##### 1.2.2        确保所有考题是根据本考点文档命题

学习理解了所有考点后，完全有能力考出 100 分。这个文档中不可能将所有题目都摆出来，所以这一点非常重要。

##### 1.2.3        确保文档是容易学习的

这里所说的考点容易学习不是说考点本身没有难度，而是指本文在描述的考点内容的时候，侧重考虑文档的易学性，文档内容应易于理解、便于学习，学员可以较快掌握。本文定位于学员通过自学就可以掌握工作中要求的必备 Java 知识，所以文档的易学性比较重要。



### 2      对象声明与初始化

#### 2.1      静态成员与非静态成员(必须了解的)

##### 2.1.1        静态上下文中只能访问静态成员

**考点说明**

static 函数只能访问 static 变量或 static 函数。

**考题样例**

```java
public class TT
{
    private String str1 = "Hello ";
   
    TT()
    {
        str1 += "china";
    }
  
    static
    {
        str1 += "world";
    }
   
    public static void main(String[] args)
    {
        System.out.println(str1);
    }
}

```

关于以上代码，下面说法正确的是：（）

A.     输出Hello china

B.     输出Hello world

C.     有编译错误

D.     world Hello china

正确答案：C



#### 2.2      数据默认值(必须了解的)

##### 2.2.1        数字类型成员的缺省值是0，布尔型为false，对象引用唯一的缺省值类型是null

**考点说明**

各种类型的缺省值：

boolean: false

byte: 0

char: '\u0000'

short: 0

int: 0

float:0.0

long: 0

double:0.0

Reference: null

**考题样例**

```java
public class TT
{
    private double d;
    private String str;
    private boolean bool;

    public static void main(String[] args)
    {
        TT tt = new TT();
        System.out.println(tt.d + tt.str + tt.bool);
    }

}
```

关于以上代码，说法正确的是：（）

A.  运行时抛异常

B.  输出 "0.0 null false"

C.  输出 "\u00000.0null"

D.  输出 "00.0null"

正确答案：B



#####  2.2.2        局部变量没有默认值

**考点说明**

==局部变量在使用之前必须显式赋值才能使用，否则将会有编译错误。==由于局部变量使用的是栈内存，由于某些原因（主要是效率）虚拟机不能为栈上的变量设置默认值，所以局部变量需要显式赋初始值。

注：==除局部变量以外的所有变量都是使用堆内存，由于在各种现代的机器上都可以很方便、快速地使大块内存初始化为相同的数值，比如可以将它初始化为 0，这个特性使得堆内存都可以由虚拟机赋初始值。==由于在调用函数时，栈需要频繁地做出栈、入栈操作，若虚拟机要为各函数中的局部变量赋初始值的话会极大地影响效率，所以局部变量不可以从虚拟机得到这种好处。

**考题样例**

​    public static void main(String[] args)

​    {

​        String str = "String";

​        int i;

​        System.out.println(str+i);

​    }

   以上程序，下列说法正确的是：（）

A.  编译错误

B.  运行时抛异常

C.  输出”String0”

D.  输出”String”

正确答案：A



#### 2.3      数组(必须了解的)

##### 2.3.1        刚构造的数组中的元素总是设为默认值

**考点说明**

Java 数组总是存在被设定的默认值。各种数据类型的数组的默认值：

boolean: false

byte: 0

char: '\u0000'

short: 0

int: 0

float:0.0

long: 0

double:0.0

Reference: null

**考题样例**

```java
public class Test
{
    public static void main(String[] args)
    {
        final int len = 5;
        int[] array =new int[len];

        for(int i = 0; i < len - 2; i++)
        {
            array[i] = i;
        }
       
        for(int i = 0; i < array.length; i++)
        {
            System.out.print(array[i] + ",");
        }
    }
}
```

以上代码下列说法正确的是：（）

A.    输出 "0,1,2,0,0,"

B.    输出 "0,1,2,"

C.    运行时抛异常

正确答案：A



##### 2.3.2        声明一个数组变量仅产生了一个数组引用（也可叫指针），但未分配数组的存储空间，一个数组的大小将在数组使用new 关键字真正创建时被给定

**考点说明**

数组变量本身仅仅是一个引用，数组的内存要用 new 运算符来分配。定义数组变量的时候不能（也不必）指定数组大小，数组的大小是在分配数组内存时才确定的。以下是定义数组的几种方法：

int[] array1 = new int[3];        //声明一个大小为3的数组

int[] array2 = new int[]{1, 2, 3};  //声明数组同时分配内存

用以下方法定义数组是错误的：

int[3] array3 = new int[3];        //数组变量不必指定大小，编译错误

int[] array4 = new int[3]{1, 2, 3};  //数组大小由初始化列表中元素个数决定，编译错误

**考题样例**

下面几种声明数组的方法，错误的有（）

A.  char[] chr = new char[3]{'A','B','C'};

B.  char[] chr = new char[3];

C.  char[] chr = {'A','B','C'};

正确答案：A



##### 2.3.3        拷贝数组应当使用 System.arrayCopy() 函数以提高效率

**考点说明**

相对于用循环来拷贝数组，System.arrayCopy()执行效率更高，这是因为它是一个原生(native)函数，可以直接高效地将 ”源array” 的数据复制到 ”目标array”。

**考题样例**

下面两种数组的拷贝，哪个更有效率？（）

String[] from = new String[10];

String to = new String[10];

A.  for (int i = 0, n = from.length; i < n; i++)

{

​	to[i] = from[i];

}

B.  System.arrayCopy(from, 0, to, 0, from.length);

正确答案：B



#### 2.4      数据初始化(必须了解的)

##### 2.4.1        一个final 变量的值不能被改变，并且必须在定义的时刻赋值

**考点说明**

声明为 final 的变量被标识为不可改变的，但==该变量引用的对象属性却可以通过访问变量、或调用成员函数改变，当然这要求那些对象提供了“写”数据的接口==（String类就没有提供“写”数据的接口，所以字符串的内容不可变）。声明为 ==final 的成员变量必须在构造对象的过程中完成初始化==：

1）其定义处，也就是说在final变量定义时直接给其赋值；

2）或者是在初始化块中；

3）或者是在构造函数中；

4）这三个地方必须（且只能）选其一

5）在以后的引用中只能读取，不能修改。

声明为 ==final 的静态变量不必在构造函数中初始化，但必须（且只能）在静态上下文中初始化一次==（定义时、静态区段均可）。

声明为==final 的局部变量必须（且只能）在使用前初始化一次，不使用的话可不必初始化==。比如可以定义以下变量(编译通过)：

```java
    public static void main(String[] args)
    {
        final int a = 0;
        final int b;
        b = 0;

        final int c;
    }
```

**考题样例**

下面关于final变量说法正确的是:()

A.  声明为final的数组其元素只能赋值一次；

B.  声明为final的整型变量其内容不能被改变；

C.  声明为final的String对象可以通过replace函数改变其内容；

正确答案：B



##### 2.4.2        final的基本用法及特点（final类、final函数、final参数）

**考点说明**

1、final类：

如果==一个类被声明为final，意味着它不能再派生新的子类==， 因此一个类不能同时既被声明为abstract，又被声明为final。

2、final函数：

如果==一个函数被声明为final，意味着它不能再被子类覆写，但可以在覆写一个父类函数时将这个函数声明为final==。若一个final 函数不是覆写父类的函数，则意味着编译时就可确定被调用的函数是哪一个函数，这样可以提高运行效率。设计final函数时，可考虑两点：第一，子类需要访问这个函数、但不能让子类覆写这个函数；第二，在第一点的基础上，函数的性能非常重要，在多数情况下，这点比较次要一些。

3、final参数：

类似于final变量，只能引用，不能修改。==final 参数仅需声明，不必显式赋值（赋值是通过函数调用传参数完成的）==，这点和final变量不太一样。

**考题样例**

```java
class abstract final A
{
    public abstract void doSomething() ;
}

class B extends A
{
    B()
    {
     System.out.println("init B…");
    }

    public void doSomething()
    {
        System.out.println("doSomething…");
    }

   public static void main(String[] argv)
   {
		B b = new B();
		b.doSomething();
   }

}
```

关于以上代码，下面说明正确的是：（）

A.  编译错误

B.  输出 "doSomething…"

C.  输出 "init B…"

正确答案：A



##### 2.4.3        常量

**考点说明**

尽管各种类型数据都可声明为final，但仅有少数数据在编译期间就可获知确定不变的值，这种数据可称为常量。==在 java 语法中有一个保留字 const，但这个关键字从来就没有被使用过==。除直接用数值表示的常量以外，其它常量都是用 final 来声明的，但不是所有的 final 数据都是常量，只有在编译期间能确定数值的才能算是常量，从以下代码可以识别出一些常量（以及一些定义常量的方法，这段代码可以编译通过）：

```java
public class Test
{
    final static int STATIC_CONST = 1; //静态常量
    final int MEMBER_CONST = 2;    //成员常量

    //下面这个数据编译期不能确定值，它的值需要运行期才确定，不是常量
    final static int staticNoneConst = (int) (Math.random()  20);

    void test(final int noneConst)     //final参数，不是常量
    {
        final int TEMP_CONST = 3;                //局部常量
        final int TEMP_CONST2 = TEMP_CONST + 1;//常量和常量运算结果也是常量     

        final int noneConst2 = noneConst + 1;    //不是常量       

        switch(noneConst)
        {
            case STATIC_CONST:
                System.out.println("STATIC_CONST");
                break;
            case MEMBER_CONST:
                System.out.println("MEMBER_CONST");
                break;
            case TEMP_CONST:
                System.out.println("TEMP_CONST");
                break;
            case TEMP_CONST2:
        }
    }
}
```

需要注意的是，一个数据是不是常量跟这个数据是不是static没有关系，比如上例中，成员数据、局部数据都可能是常量，==关键只在于数据是在编译期间就可以确定它的值==。常量不仅在虚拟机的一次运行期间数值不会变，并且多次重新启动虚拟机反复运行后，它的值还是恒定不变的。从这个意义上说，所有的引用都不是常量，因为引用都是在运行期间才能确定它的值。

**考题样例**

以下说法正确的是：（）

A.  只要是用 final static int 修饰的数据都恒定不变的，多次重新启动虚拟机反复运行后，它的值还是恒定不变的

B.  只要是用 final static int 修饰的数据都可以用在 switch – case 语句的 case 子句中

C.  只能用 final static 定义常量

D.  保留字 const 不能用来定义常量

正确答案：D



#### 2.5      构造函数(必须了解的)

##### 2.5.1        在构造过程中避免调用子类可以覆写的函数

**考点说明**

==有继承关系的类之间的初始化过程，是父类的构造函数先执行，子类的构造函数后执行。==如果在父类构造函数中调用了子类可以覆写的函数，一旦子类覆写了父类的函数，==由于父类构造函数是先于所有子类构造函数执行，在子类覆写的函数中看到的所有子类数据都是未初始化的==，这样做非常危险。

**考题样例**

以下程序输出什么？

```java
class Parent
{
    protected String pStr;
   
    public Parent()
    {
        invokePrint();
        pStr = "parent";
    }
   
    protected void invokePrint()
    {
        System.out.println("Parent print, pStr = " + pStr);
    }

}

public class Child extends Parent
{
    private String cStr;

    public Child()
    {
        super();
        cStr = "child";
    }
   
    protected void invokePrint()
    {
        System.out.println("Child print, pStr = " + pStr + " cStr = " + cStr);
    }

    public static void main(String[] args)
    {
        Child c = new Child();
        c.invokePrint();
        Parent p = new Parent();
        p.invokePrint();
    }
}
```

A.

Child print, pStr = null cStr = null

Child print, pStr = parent cStr = child

Parent print, pStr = null

Parent print, pStr = parent

B．

Parent print, pStr = null

Parent print, pStr = parent

Parent print, pStr = null

Parent print, pStr = parent

C.

Child print, pStr = parent cStr = null

Child print, pStr = parent cStr = child

Parent print, pStr = null

Parent print, pStr = parent

D.

Child print, pStr = null cStr = null

Child print, pStr = null cStr = child

Parent print, pStr = parent

Parent print, pStr = parent

正确答案：A



#### 2.6      有继承关系的类之间的初始化顺序(必须了解的)

##### 2.6.1        有继承关系的类之间的初始化顺序是先静态后非静态，先父后子

**考点说明**

初始化分为类的初始化和对象的初始化。

对于类的初始化，其顺序是：

1、  若父类未加载，则先执行父类的类初始化过程，然后加载本类

2、  给类变量分配存储空间

3、  给类变量显式赋值

4、  执行static块

在虚拟机一次运行的整个生命周期中，类的初始化一般只做一次（==仅在特殊情况下才有可能做多次，比如一个类初始化过程抛出了异常导致类加载失败后，以后再用这个类时，虚拟机会再次尝试加载这个类==）。另外如果在类的初始化过程中需要使用其他未加载的类，则对未加载的类也需执行上述过程。

说明：第 2 步“给类变量分配存储空间”时，由于新分配的存储空间都被虚拟机初始化过了，所以自动取得了默认值。虚拟机会通过某些方法（比如调用memset()函数）将一大块内存全部置为默认值，这在java代码中不可见，编译出来的字节码也没有相关的内容，如果没有显式赋初始值则声明变量的地方不能设置断点就是这个原因。

对象的初始化，其顺序是：

1、  给对象分配存储空间(自动取得了默认值)

2、  若有父类，则执行父类的对象初始化过程

3、  给对象变量显式赋值

4、  执行构造函数。

只有 java.lang.Object 类无需执行步骤 2 ，在执行父类的对象初始化过程中，具体要调用的构造函数由子类构造函数中的 super() 调用确定，若没有明确的 super() 调用，则调用默认构造函数。对象的初始化过程在每次调用 new 运算符时都会被执行，但在对象克隆时不会执行(除非故意在克隆函数中用 new 构造了新对象)。

**考题样例**:

运行下面程序输出的是：

```java
class Parent
{
    protected static String str = getParentMessage();
    static
    {
        str = "static block init";
    }
 
    private static String getParentMessage()
    {
        System.out.println("getParentMessage");
        return "getParentMessage";
    }

    static class InnerClass
    {
        static
        {
            str = "inner class static block init";
        }
    }
}
 

public class Child extends Parent
{
    Child()
    {
        System.out.println("final str=" + str);
    }

    public static void main(String[] args)
    {
        new Child();
    }
}
```

 A.  getParentMessage

final str=static block init

B.  getParentMessage

final str=inner class static block init

C.  抛出运行时异常  

D.  final str=static block init

正确答案：A



### 3      运算

#### 3.1      赋值运算

##### 3.1.1        将一个对象引用赋值给另一个的一般规则是，你可以对继承树向上赋值但不能向下赋值

**考点说明**

将一个对象引用赋值给另一个的一般规则是，可以对继承树向上赋值但不能向下赋值(子类的对象一定是父类的对象，但是父类的对象不一定是子类的对象)。可以这样想，如果将一个子类的实例赋值给基类，Java 知道哪个函数会在子类中。但是一个子类可能有基类没有的额外函数，这时候想调用子类的函数，就必须将该对象强制类型转换成子类的对象。

注：子类对象可以当成父类对象使用，在设计领域就延伸为里氏代换原则。==里氏代换原则：如果对每一个类型为T1的对象o1,都有类型为T2的对象o2，使得以T1定义的所有程序P在所有对象o1都换成o2时，程序P的行为没有变化，那么类型T2是T1的子类型，即如果一个软件实体使用的是基类的话那么也一定适用于子类，但反过来的代换不成立==。

**考题样例**

在下面给定的代码中，哪一个会编译失败：

```java
interface IFace{}

class CFace implements IFace{}

class Base{}

public class ObRef extends Base
{
    public static void main(String[] argv)
    {
        ObRef ob = new ObRef();
        Base b = new Base();
        Object o1 = new Object();
        IFace o2 = new CFace();
    }
}
```

A. o1=o2;   B. b=ob;    C. ob=b;    D. o1=b;

正确答案：C



##### 3.1.2        数值的默认类型

**考点说明**

整数的默认类型是int；==对于长整型(long)数值，需要加上后缀L，避免溢出==。

浮点数的默认类型是double，==将浮点数赋值给值域比它小的变量时，需要进行强制类型转换==；==构造单精度浮点数时，可以用后缀f指定==。

**考题样例**

下面哪些可以编译成功：

1） float f = 10f;

2） char c = 10.0;

3） float f = 10.1f;

4） byte b = 10b;

A. 1)和3)       B. 1)和4）      C. 2)和3）      D. 都可以编译成功

正确答案：A



#### 3.2      数据类型转换(必须了解的)

##### 3.2.1        数据类型转换的默认规则

**考点说明**

==值域小的自动向值域大的方向转换,值域大的向值域小的方向转换需要进行强制类型转换==

**考题样例**

下面哪个会编译失败：

A. double d = 10;

B. long l = 10;

C. float f = 10;

D. float f = 10.0f;

​	int i = f;

正确答案：D



#### 3.3      逻辑运算符的短路效应(必须了解的、常错的)

##### 3.3.1        逻辑运算符短路效应的应用场景

**考点说明**

对于逻辑运算符”&&”来说，如果第一个操作数为假，第二个操作数的值就没有作用了，也不会执行，所有的结果都为假。

对于逻辑运算符”||”来说，如果第一个操作数为真，第二个操作数的值就没有作用了，也不会执行，所有的结果都为真

注意：==”&”和”|”既是位运算符也是逻辑运算符，”&”是按位取与，”|”是按位取或==，当使用这两个运算符进行逻辑运算时，这两个运算符和逻辑运算的”&&”,”||”是不同的，不能产生短路效应。

**考题样例**

尝试编译运行以下代码时会发生什么情况？

```java
int outPut=10;

boolean b1 = false;

if((b1 == true) && ((outPut += 10) == 20))
{
	System.out.println("We are equal "+ outPut);
}
else
{
	System.out.println("Not equal! "+ outPut);
}
```

A.    编译错误

B.    输出 "We are equal 10"

C.    输出"Not equal! 20"

D.    输出"Not equal! 10"

正确答案：D



#### 3.4      instanceof运算符(必须了解的)

##### 3.4.1        getClass() 函数与 instanceof 运算符的差异

**考点说明**

getClass()用于获取对象的真实类型，两个有继承关系的类分别调用getClass()获得的类是不同的。

instanceof 用来判断一个对象是否是某种类型的，即IS-A的关系。子类对象是父类对象的实例。

**考题样例**

以下程序

```java
class Father{}

public class Child extends Father
{
    public static void main(String[] args)
    {
        Child child = new Child();
        Father father = new Father(); 
        System.out.println(child.getClass() == father.getClass());
        System.out.println(child instanceof Father);
    }
}
```

的输出是什么？

A.    输出true,false

B.    输出true,true

C.    输出false,true

D.    输出false,false

正确答案：C



##### 3.4.2        变量为 *null* 时使用 *instanceof* 运算符返回值总是 *false*

**考点说明**

==null 对象不是任何类的实例，所有instanceof运算符返回值总是false==

**考题样例**

以下程序的输出是什么？

System.out.println(null instanceof Object);

A.    输出 Object

B.    输出 true

C.    输出 false

D.    输出 null

正确答案：C



#### 3.5      传值与传引用(必须了解的)

##### 3.5.1        对象是通过引用传递的（引用变量通过值传递）

**考点说明**

所有的变量（基本类型值和对象的引用值）都是值传递。但是这不是全部情况，对象是经常通过Java 的引用变量来操作的。所以也可以说对象是通过引用传递的（引用变量通过值传递）。调用者对基本类型参数（int，char 等）的拷贝在相应参数变化时不会改变。但是，在被调用函数改变相应作为参数传递的对象（引用）字段时，调用者的对象也改变其字段。

**考题样例**

以下程序输出结果是什么？

```
public class Point
{
    private int x;
    private int y;

    public Point(int x, int y)
    {
        this.x = x;
        this.y = y;
    }

    public void setLocation(int x, int y)
    {
        this.x = x;
        this.y = y;
    }

    public static void main(String[] args)
    {
        Point p1 = new Point(0, 0);
        Point p2 = new Point(0, 0);
        modifyPoint(p1, p2);
        System.out.println("[" + p1.x + "," + p1.y + "], [" + p2.x + "," + p2.y + "]");
    }
 
    private static void modifyPoint(Point p1, Point p2)
    {
        Point tmpPoint = p1;
        p1 = p2;
        p2 = tmpPoint;
        p1.setLocation(5, 5);
        p2 = new Point(5, 5);
    }
}
```

A. [0,0], [0,0]

B. [5,5], [0,0]

==C. [0,0], [5,5]==

D. [5,5], [5,5]

正确答案：C



##### 3.5.2        对函数传递基本类型参数，是直接传递值，函数得到它的拷贝，任何修改都不会在外部函数得到反映

**考点说明**

当你对函数传递基本类型参数，是直接传递值。函数得到它的拷贝，任何修改都不会在外部函数得到反映。

**考题样例**

以下程序输出结果是什么？

```java
public static void main(String[] args)
{
    int i1 = 0;
    int i2 = 0;
    modifyInt(i1, i2);
    System.out.println("[" + i1 + "], [" + i2 + "]");
}

private static void modifyInt(int i1, int i2)
{    
    i1 = 10;
    i2 = 20;
}
```

A.    [0], [0]

B.    [0], [10]

C.    [10], [20]

D.    [10], [0]

正确答案：A



#### 3.6      equals() 函数与“==”运算(常错的)

##### 3.6.1      对于对象，“==”运算的语义是用来判断对象是不是同一个对象

**考点说明**

“==”用来判断两个对象是不是相同的，用C语言的说法就是：两个对象的指针是不是指向相同的地址

**考题样例**

以下程序输出结果是什么？

```java
String strTest1 = "abc";
String strTest2 = new String("test");

if (strTest1 == "abc")
{
    System.out.print("true");
}
else
{
    System.out.print("false");
}

if (strTest2 == "test")
{
    System.out.print("true");
}
else
{
    System.out.print("false");
}
```

A. true true    B. true false    C. false true    D. false false

正确答案：B



**3.6.2        equals() 函数用来判断对象的内容(或者是属性)是否相同**

**考点说明**

equals()函数可以认为是对象深层次的比较，比较对象的值是否相同。注意：equals()函数不能用于比较数组是否相同，对两个数组用equals()函数比较等同于”==”比较。

**考题样例**

以下程序输出结果是什么？

```java
String strTest1 = "abc";

String strTest2 = new String("test");

if (strTest1.equals( "abc"))
{
    System.out.print("true");
}
else
{
    System.out.print("false");
}

if (strTest2.equals("test"))
{
    System.out.print("true");
}
else
{
    System.out.print("false");
}
```

A. true true    B. true false    C. false true    D. false false

正确答案：A



##### 3.6.3        覆写 equals() 的注意事项

**考点说明**

覆写equals函数要遵循以下约定：

1.不同类型对象的equals()函数总是返回false

2.对于任意的非空引用值x,x.equals(null)总是返回false

3.自反性原则：对象和它自己比较总是返回true

4.对称性原则：如果x和y比较返回true,那y和x比较也一定返回true

5.传递性：如果x和y比较返回true，y和z比较返回true，那x和z比较也一定返回true

6.一致性原则：对于任意的引用值x和y，如果equals比较的对象信息没有改变的话，那么，多次调用euqals()函数返回的结果总是相同的

**考题样例**

下面哪种说法是正确的?

```java
public class CompObj
{
	public int x;
	public int y;

	public boolean equals(Object o1)
	{
    	CompObj compO1 = (CompObj) o1;
    	return this.x == compO1.x;
	}
}
```

A.    该equals函数实现错误，没有判断是不是相同类型的对象，并且在o1是null的时候会抛出异常

B.    该equals函数实现正确

C.    该equals函数实现错误， 没有比较y的值

正确答案：A



##### 3.6.4        覆写hashCode()函数的注意事项

**考点说明**

==在每个覆写了equals()函数的类中，必须要改写hashCode()函数==

覆写hashCode()函数需要注意以下几点：

1. ==在一个应用程序的执行期间，如果一个对象的equals函数比较所用到的对象的信息没有被修改的话，那么对该对象调用多次hashCode(), 它必须始终如一的返回同一个整数==。

2. 如果两个对象根据equals()函数是相等的，那么调用这两个对象中任意一个对象的hashCode()函数必须产生同样的整数结果。

3. 如果两个对象根据equals()函数是不等的，那么调用这两个对象中任意一个对象的hashCode()函数不要求必须产生相同的结果。

**考题样例**

下面哪种说法是正确的?

int i = 0;

public int hashCode()

{

​    return i++;

}

A.         该hashCode函数实现错误，相同的对象调用多次hashCode()返回不同的结果

B.          该hashCode函数实现正确

正确答案：A



#### 3.7      克隆与复制(常错的)

##### 3.7.1        为防止对象被意外修改，应当在必要时拷贝对象

**考点说明**

Object.clone() 函数 ==仅仅实现浅层拷贝（克隆）==——只把基本数据类型和引用类型的值复制给新对象。这就意味着新旧对象：

1、各自拥有数值相同的ints、floats和booleans等基本数据类型的值。

2、但是==共享数组、HashTable、Vector等此类对象成员==。

如果不想让对象的值发生变更时影响到新对象，就要进行深度拷贝(克隆)。

**考题样例**

以下类哪个是安全拷贝的，可以保证对克隆对象的修改不会影响源对象：（）

```
class A  implements Cloneable
{
    private int a = 1;
    private boolean b = false;
    
    public Object clone()
    {
        try
        {  
            return (super.clone());
        }
        catch(CloneNotSupportedException cnes)
        {
            throw new InternalError();
        }
    }
}

 
class B  implements Cloneable
{
    private int a = 1;
    private boolean b = false;
    private HashTable h = new HashTable();

    public Object clone()
    {
        try
        {  
            return (super.clone());
        }
        catch(CloneNotSupportedException cnes)
        {
            throw new InternalError();
        }
    }
}
```

A.  类A     B. 类B      C. 类A和类B都不是安全拷贝

正确答案：A



##### 3.7.2        如何正确覆写 clone() 函数

**考点说明**

1、==要支持克隆必须实现Cloneable接口==

2、==实现clone()函数时记得调用super.clone() 函数，实现浅层克隆==。

3、如果缺省的浅层克隆不符合要求，再实现深层克隆。

**考题样例**

下面代码段中深度克隆的实现方式是否正确？

```
public class MyCloneableClass
{
    private Map map = new HashMap();

	public Object clone() throws CloneNotSupportedException
	{
		return super.clone();
	}
}
```

A.         不正确，没有对map进行深度克隆

B.          ==不正确，既没有实现Cloneable接口，也没有对map进行深度克隆==

C.          正确，可以完成深度克隆工作

D.         不正确，没有实现Cloneable接口

正确答案：B



#### 3.8      字符串运算(必须了解的、常错的)

##### 3.8.1        *String* 对象一旦被创建，它就决不能被改变

**考点说明**

在 java 类库中，==String 类被设计成不可变类，它仅提供了“读”String数据的接口，没有提供“写”String数据的接口==。

在 String 的 + 运行算中，虚拟机会构造新的 String，原来的 String 保持不变。调用 String 的 replace()、trim() 等函数时，如果结果和原来字符串不同时，也是产生新的String对象，原来的String对象不受影响。

**考题样例**

以下代码的输出结果是：

String abc = "abc";

abc.replace('b', 'd');

System.out.println(abc);

A. abc      B. adc      C. 答案 A 和 B 都不对

正确答案：A



##### 3.8.2        不要使用new String("some string") 方式构造字符串

**考点说明**

==创建字符串时不要使用new String("some string")，在Java中，"some String"默认就已经创建了一个字符串，如果再用new String("some string")，将会创建出另外一个字符串，造成浪费==。

"abc"本身就是一个字符串对象，而在运行时执行String s= new String("abc")时，"abc"这个对象被复制一份到heap中，并且把heap中的对象引用交给s持有。这条语句创建了两个String对象。

**考题样例**

不考虑上下文，以下代码执行过程中创建了几个对象：

String abc = new String("abc");

A. 1        B. 2        C. 答案 A 和 B 都不对

正确答案：B



##### 3.8.3        从 byte[] 数组构造字符串时，应该使用 new String(byte[]) 构造函数

**考点说明**

byte[] 数组的 toString() 函数 实际上是Object的toString函数，而不是数组的内容。==想通过byte[]数组来构造字符串，需要调用new String(byte[])构造函数==。

**考题样例**

​    下面程序输出的结果:（）

​    byte[] a = {49,50,51,52};

​    System.out.println.( a.toString());

A. "1234"    B. 49    C. 52    D. 以上都不是

正确答案：D



##### 3.8.4        避免大量使用字符串的 + 运算，应当使用*StringBuffer* 的 *append() 函数替代

**考点说明**

对于循环体内部的字符串连接，使用StringBuffer和append()函数；否则使用String连接符”+”。

String对象是不可变的，实现”+”的工作原理如下：

> 获取加号两边的操作数；如果右边的操作数不是字符串，那么用toString()函数把它变成字符串，然后创建一个新的字符串，该字符串合并了两个待操作的字符串。创建新的临时字符串的过程是新建一个StringBuffer，然后追加其他字符串，最后调用toString。正是由于每次循环过程中，都会创建一个或多个新的对象，原来的旧对象成为垃圾，这样的话一个循环就会产生n个对象，从而造成内存的浪费。

使用StringBuffer对象可以避免创建多余的字符串对象，StringBuffer用来表示内容可变的字符序列，提供一些函数用来修改底部字符序列的内容。

在循环体外进行字符串连接时，建议采用”+”，虽然也存在上述问题，但是更加易于书写和理解。特别是在仅连接两三个字符串的情况下，性能影响不大。

**考题样例**

下面说法错误的是：

A.     在构造StringBuffer对象时，指定合适的capacity可以减少产生临时对象，从而提高效率

B.      在循环内使用StringBuffer的append() 的效率通常要好于直接对String进行 + 运算

C.      String s = “s1”+”s2”; 和 String s = new StringBuffer().append(“s1”).append(“s2”).toString() 效率一样

D.     由于编译器会将String的 + 运算转换为StringBuffer的 append() 调用，所以二者在效率上根本没有任何差别

正确答案：D



####   4      垃圾收集

#####  4.1.1        当对象不可达的时候，这个对象就成为了可回收的

**考点说明**：

在Java虚拟机中使用有向图的方式来存储对象引用，==当一个对象不可达时，该对象就成为可回收的了==。

==如果从线程的栈或者静态变量可以引用到对象，这个对象就是可达的，如果引用不到，这个对象就是不可达的。==

在程序运行时，HEAP 中从 “Root set of references” 通过直接或间接方式引用到的对象是可达对象(白色)，其它对象是不可达对象(蓝色)，不可达的对象就是垃圾、需要回收，如下图：

![1563248073259](D:\Notes\raw_images\1563248073259.png)

​                                                                      图1 可到达和不可到达的对象

Java 的垃圾收集的实现方法有很多种，但都离不开一个最基本的方法，那就是“跟踪收集”算法，这种算法从根集开始沿着引用跟踪，直到检查了所有可到达的对象。可以在程序注册表中、每一个线程堆栈中的（基于堆栈的）局部变量中以及静态变量中找到根。从这个原理上可以看出，静态变量和线程在垃圾收集的过程中扮演了非常重要的角色。

注意，从上图中可以看出，==对象是否被引用并不是对象是不是垃圾的判断根据==，比如上图中的四个garbage对象中有一个是还被其它garbage对象引用的，但它也是垃圾。

**考题样例**：

下述代码中，执行完语句1后，b是否会被回收？

```JAVA
class Stack
{
    private Object[] elements;
    private int size = 0;
    
    public Stack(int initialCapacity)
    {
        this.elements = new Object[initialCapacity];
    }
    
    public void push(Object e)
    {
        ensureCapacity();
        elements[size++] = e;
    }

    public Object pop()
    {
        if (size == 0)
            throw new EmptyStackException();
        return elements[--size];
    }   

    private void ensureCapacity()
    {
        if (elements.length == size)
        {
            Object[] oldElements = elements;
            elements = new Object[2  elements.length + 1];
            System.arraycopy(oldElements, 0, elements, 0, size);
        }
    }

}

       
public static void main(String[] args)
{
        Stack stack=new Stack(10);
        Integer i=new Integer(10);
        stack.push(i);
        Object b=stack.pop();
        b=null;                          //语句1
        ……；

}
```

A. 可以

B. 不可以

正确答案：B



##### 4.1.2        *Java*代码也会造成内存泄漏

**考点说明**：

==当对象已经没用了，但它还是可达对象时，内存泄漏就出现了。==如果仅仅将对象的引用设置成null，而该对象还是可达的，就不会被回收，会造成内存泄漏。

**考题样例**：

```
class Base
{
    String s;
    Base(String s)
    {
        this.s = s;
    }

    public void setString(String s)
    {
        this.s = s;
    }
}

public class TestPanel
{
    private List list=new ArrayList();
    
    public static void main(String[] argv)
    {
        TestPanel ur = new TestPanel();
        ur.go();
        //其它代码（略）
    }
 
    public void go()
    {
        Base b1 = new Base("One");                 //语句1
        b1.setString("");
        list.add(b1);
        Base b2 = new Base("Two");                 //语句2
        list.add(b2);
        b1=null;
        b2=null;
    }

}
```

上述代码中，执行完方法go后，可能被回收的对象是：

A.  语句1生成的Base实例

B.  语句2生成的Base实例

C.  语句1、2生成的Base实例

D.  都不可能

正确答案：D



#### 4.2      垃圾收集的相关函数(必须了解的)

##### 4.2.1        可以使用System.gc()来建议垃圾回收器收集垃圾，但是这并不能保证执行

**考点说明**：

==System.gc()函数只是建议JVM进行内存回收工作，但是具体的执行时间是不确定的，系统不能保证在调用函数后立即执行回收工作==。

程序中显式的调用gc()函数，有可能无法达到预期的立即执行垃圾回收工作的效果，所以在程序中建议不要依赖于System.gc()函数来释放内存。

此函数同Runtime.getRuntime().gc()。

**考题样例**：

使用System.gc() 函数必然可以有效地改善系统的性能。

A. 正确         B. 错误

正确答案：B



##### 4.2.2        不要依赖finalize()函数来回收资源

**考点说明**：

==finalize ()函数由JVM在进行内存回收的时候调用。同System.gc()，系统并不能保证一定会执行finalize()函数，即使执行，它的时间也是不确定的，因此不要依赖于finalize函数来释放资源==。

此函数声明如下：

protected void finalize() throws Throwable{};

**考题样例**：

下述代码运行结果是

```
public class Chair
{
    Chair()
    {
         System.out.println("Created.");
	}
 
    public static void main(String[] args)
    {
        Chair chair = new Chair();
        ……；（后续处理略）
	}   
 
    public void finalize()
    {
        System.out.println("Finalizing Chair.");
    }
}
```

A.  Created.

B.  Finalizing Chair.

C.  Created.

​	Finalizing Chair.

D.  一定输出Created, 但是不确定是否输出Finalizing Chair

正确答案：D



###         5     流程控制和异常处理

#### 5.1      *switch*语句(必须了解的)

#####  5.1.1        *switch*语句必须的配套的*default*分支

**考点说明**

==从语法上讲switch语句后可以不要default分支，但从编程规范要求必须写default分支==，增加该分支是个良好的编程风格。这样做能保证处理到所有的情况，并做出及时处理避免异常。对错误情况也可提供定位信息。

**考题样例**

下列关于switch语句的default分支说法正确的是（）

A.  switch语句中必须有default分支

B.  switch语句中的default分支不是必须的，所以在编码中不考虑default分支也不会有问题

C.  在语法上,switch语句中的default分支不是必须的，但在编码中不考虑default分支有可能发生问题

正确答案：C



##### 5.1.2       各个分支之间不能遗漏break

**考点说明**

==如果在case子句的末尾忘了加break，那么下一个case子句会被接着执行==。这种情况非常危险，常常引发错误，对于流程要求不加break的特殊情况需要增加注释加以说明。

**考题样例**

下列程序的输出结果是（ ）

```
int a  = 2;
switch(a)
{
  case 1:
     System.out.println("The input is 1");
  case 2:
     System.out.println("The input is 2");
  case 3:
     System.out.println("The input is 3");
  default:
     System.out.println("Bad input");
}
```

A.  The input is 2

​	 Bad input

B.  The input is 2

​	The input is 3

​	Bad input

C.  The input is 2

D.  Bad input

正确答案：B



##### 5.1.3        *switch*语句必须使用和 *int* 类型兼容的数据类型

**考点说明**

switch 语句==只能用byte，char，short 或者 int 类型作参数==

**考题样例**

语句switch(expr){…} ，合法的表达式expr可以具有哪些类型（ ）

A．long

B． string

C． unsigned int

D．byte、char、short、int

正确答案：D



##### 5.1.4        *case*的表达式必须是常量

**考点说明**

==case的表达式必须是常量，不能是变量。==

**考题样例**

以下代码中, 可用在case中的变量有哪些？( )

protected int a1;

protected final int a2 = 1;

public void test(int i1, final int i2)

{

​    final int i3 = 100;

​    //...

}

A. a1, a2       B. a1, i1       C. a2, i3       D. a2

正确答案：C



#### 5.2      异常处理(必须了解的)

##### 5.2.1        *try...catch* 和*finally*块的对应关系

**考点说明**

try、catch 和finally匹配关系。

程序进入try 块，如果没发生错误，程序离开try块后，进入finally程序块，执行该语句块。如果发生错误（在try中检测到一个错误）就会跳转到catch块中，在catch中处理错误，处理完后，进入finally块执行。一个try至少需要一个catch或者一个finally对应，语法上允许三种情况出现：

try ... catch 配对出现

try ... finally 配对出现

try ... catch ... finally 同时出现

**考题样例**

关于try block, 说法正确的是（ ）

A.  每个try block必须有一个以上的catch block相对应

B.  每个try block必须有一个finally block相对应

C.  每个try block至少需要有一个finally block或者一个catch block相对应

D.  每个try block至少需要有一个finally block和一个catch block相对应

正确答案：C

 

#####  5.2.2        对于资源的释放建议放在*finally*中进行

**考点说明**

为保证资源能够在任何情况下被释放，建议放在finally中进行处理。

**考题样例**

下面的代码中，输入流reader的关闭放在什么位置是最合适的（）

```java
public static void readFileByLines(String fileName) throws IOException
{
    File file = new File(fileName);
    BufferedReader reader = null;
    try
    {
        reader = new BufferedReader(new FileReader(file));
        String tempString = null;
        int line = 1;
        while((tempString = reader.readLine()) != null)
        {
            line++;
        }
        reader.close(); ////////////////////// 答案 A
    }
    catch(Exception e)
    {
        reader.close(); ////////////////////// 答案 B
    }
    finally
    {
        if(reader != null)
        {
            try
            {
                reader.close(); ///////////// 答案 C
            }
            catch(IOException e1)
            {
                e1.printStackTrace();
            }
        }
    }
}
```

正确答案：C



##### 5.2.3        通常*finally*块始终被执行

**考点说明**

不论在try/catch 部分是不是有返回，try/catch 块的finally 子句总会执行。只有一种情况下，finally 子句不会被执行 那就是在try 或 catch 块中执行了 System.exit() 函数。

**考题样例**

如下代码：

```
public void test()
{
    try
    {
        oneMethod();
        System.out.println("one");
    }
    catch(Exception x2)
    {
        System.out.println("two");
    }
    finally
    {
        System.out.println("finally");
    }
}
```

如果oneMethod()运行过程中未抛出任何异常，以上代码正确的输出是什么（ ）

A.  one

B.  two

C.  three

D.  one

​	finally

正确答案：D



##### 5.2.4        在子类中覆写函数时只能抛出父类中声明过的异常或者异常的子类。

**考点说明**

也可理解为==子类中只能抛出比父类更少的异常==。

**考题样例**

以下说法正确的是：（ ）

A.  覆写函数的参数列表可以不同于被覆写函数

B.  覆写函数的可见性可以低于被覆写函数，以屏蔽父类的某些函数

C.  覆写函数可以抛出更少的异常，或者抛出对应异常的子类

D.  覆写函数可以抛出不一样的异常以补充父类为考虑到的异常

正确答案：C

 

#### 5.3      循环

#####　5.3.1        避免在循环中删除元素时循环次数错误

**考点说明**

在删除集合类中元素时，使用循环删除，没有控制循环次数，造成错误。

掌握集合类的remove函数和迭代器Iterator的remove函数用法与限制：如果使用集合类的remove函数，要注意调整循环次数，并仔细检查代码逻辑。==很多容器是fail-fast的, 在Iterator创建后, 如果使用非Iterator的方法修改容器数据，再通过Iterator遍历时, 就会抛出ConcurrentModificationException==，在使用时多加注意。

**考题样例**

以下程序，如果要删除容器中所有"two",哪一段代码填入deleteElement()方法中是正确的（）

```java
public static void main(String[] args)
{
    List list = new ArrayList();
    list.add("one");
    list.add("two");
    list.add("two");
    list.add("two");
    deleteElement(list);
}

private static void deleteElement(List list)
{
    //在这里填入一段代码
}
```

 代码一：

```java
 for(int i = 0; i < list.size(); i++)
{
        String item = (String)list.get(i);
        if(item.equals("two"))
        {
            list.remove(i);
        }
}
```

代码二：

```java
 for(Iterator iter = list.iterator(); iter.hasNext(); )
 {
        String item = (String)iter.next();
        if(item.equals("two"))
        {
            iter.remove();
        }
}
```

代码三：

```java
  int i = 0;
  for(Iterator iter = list.iterator(); iter.hasNext(); )
  {
        String item = (String)iter.next();
        if(item.equals("two"))
        {
            list.remove(i++);
        }
   ｝
```

代码四：

```java
    for(int i = 0, len = list.size(); i < len; i++)
    {
        String item = (String)list.get(i);
        if(item.equals("two"))
        {
            list.remove(i);
        }
    }
```

A. 代码一    B. 代码二    C. 代码三    D. 代码四

正确答案：B



#####  5.3.2        循环的性能

**考点说明**

1、避免在循环体中创建对象。

2、==循环的嵌套层次——小的写在外面。==

3、==尽量避免在循环体中使用try-catch块，最好在循环体外使用try-catch块以提高系统性能。==

4、==建议使用局部变量保存循环的次数。==

**考题样例**

下列哪种说法正确：（ ）

```java
private static void testA()
{
    String[] strArr = {"33", "44", "55", "66"};
    try
    {
        for(int i = 0; i < strArr.length; i++)
        {
            int val = Integer.parseInt(strArr[i]);
        }
    }
    catch(Exception ex)
    {
        ex.printStackTrace();
    }
}

private static void testB()
{
    String[] strArr = {"33", "44", "55", "66"};
    for(int i = 0; i < strArr.length; i++)
    {
        try
        {
            int val = Integer.parseInt(strArr[i]);
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
```

A.  方法testA()效率比testB()高

B.  方法testA()效率比testB()低

C.  方法testA()效率与estB()一样

D.  无法确定

正确答案：A



#### 6      访问控制、继承和多态

##### 6.1.1        实例变量的声明和获取

**考点说明**

尽量==避免将一个类中的各变量都声明为public，建议通过getter/setter访问==。

在类的设计中，变量的直接访问总会导致紧耦合，==因为若一个变量被外界直接访问，那么它的变量名称、类型一有变动将对其客户造成直接影响==，而且public的变量可被多个客户访问，若它不是final的，那么每个客户都可对其不断修改，造成安全隐患。而通过getter/setter方式通过增加一层中间层，提供了更多的灵活性：变量名称的更改不会影响客户；其类型可以是getter返回类型的任一子类，==因此通过getter/setter方式设计类也更好的遵从了面向接口编程的思想==；当外界调用getter函数取值时可以添加一定适配策略：如憜性初始化避免无用对象的生成，必要时返回对象的复制品将增加更高的安全性，对象为空时返回一个null对象等；当外界调用setter函数设值时同样可添加适配策略：如限制设值次数；当传入对象为空时可将其转为null对象等。

**考题样例**

对于类B的说法不正确的是（    ）

```java
public class B
{
    private A aa = null;
    
    public A getA()
    {
        return aa;
    }

    public void setA(A aa)
    {
        aa = aa;
    }

    public void doTask()
    {
        aa.method();
    }
}
```

A.     类B的设计具有较好的封装性

B.      getA()方法能返回任意A的子类型

C.      setA()方法能传入任意A的子类型

D.     变量名aa的更改会影响到客户

正确答案：D



##### 6.1.2        类、变量、函数的访问权限

**考点说明**

各访问修饰符的访问权限如下（ OK 表示可以访问，空白表示不能访问）：

|           | 同一个类 | 同一个包 | 不同包的子类 | 不同包非子类 |
| --------- | -------- | -------- | ------------ | ------------ |
| private   | OK       |          |              |              |
| 默认      | OK       | OK       |              |              |
| protected | OK       | OK       | ==OK==       |              |
| public    | OK       | OK       | OK           | OK           |

说明：==上表中的“默认”不是指Java中的default关键字，当没有用修饰符时我们称其为“默认”权限==。

**考题样例**

关于以下的代码片断, 说法错误的是？()

```java
public class A
{
    int a;
    protected final int b = 1;
    //...
}
```

A.     a对于与A在同一个包的类是可见的

B.      ==a对于A的子类是可见的==   # 需要同一个包

C.      b对于与A在同一个包的类是可见的

D.     b对于A的子类是可见的

正确答案：B



##### 6.1.3        接口变量和函数的默认访问权限

**考点说明**

==接口中声明的属性默认为public、static、final的==；

==接口的函数定义默认为public abstract的==。

**考题样例**

对于下面代码片断说法正确的是（   ）  

public interface S

{

​    int a = 0；

​    void method();

}

A.     此代码编译不能通过，因为method（）没有abstract修饰

B.      此代码编译不能通过，因为interface中不能给变量赋值

C.      变量a的访问权限是包访问权限

D.     变量a的访问权限是public的

正确答案:D



##### 6.1.4        内部类对包含它的外部类中的字段，函数的访问权限

**考点说明**

内部类是在一个类的内部嵌套定义的类，它==可以是其它类的成员==，也==可以在一个语句块的内部定义==，==还可以在表达式内部匿名定义==。内部类有如下特性：

- ==可以使用包含它的类的静态和实例成员变量，即使它们在外围类中是private 的。==

- ==若被声明为static,就不能再访问其外部类的非静态成员。==

- ==若想在Inner Class中声明任何static成员,则该Inner Class必须声明为static。==

**考题样例**

对于下面代码片断说法正确的是（    ）

```java
public class OuterClass
{
    int a;
    static int b;
    
    private void method()
    {
        System.out.println("method()");
    }

    class Inner
    {
        static int c;//A

        void innerMethod()
        {
            method();//B
        }
    }

    public static void main(String[] s)
    {
        Inner inner = new Inner();//C
        inner.innerMethod();//D
    }
}
```

A.   A处编译错误，因为==只有静态内部类才能声明静态成员==

B.   B处编译错误，因为method() 是private的

C.  此类编译正确

正确答案：A



#####  6.1.5        内部类可以访问声明为 *final* 的局部变量(常用的)

**考点说明**

内部类可以使用外围类的静态和实例成员变量，即使它们在外围类中是private 的。

但是==对于局部变量，只有final的才能被内部类访问==。

**考题样例**

对下面代码片断说法正确的是（  ）

```java
public class OuterClass
{
    private int a;
    private void method()
    {
        int methodVal = 3;
        class TempInner
        {
            int cccc = methodVal;//A
            Inner inner = new Inner();//B
            void doTest()
            {
                inner.c = 5;//C
            }
        }
    }

    class Inner
    {
        int c = a;
        void innerMethod()
        {
            method();
        }
    }
}
```

A.     A处编译错误，因为==内部类只能访问final的局部变量==

B.      C处编译错误

C.      此类编译正确

D.     B处编译错误，因为非静态内部类实例化时需要从属于一个外部类对象。

正确答案：A



#### 6.2      继承

#####  6.2.1        类的单继承性

**考点说明**

java.lang.Object类没有父类，但它是其它所有类的共同父类，Java中所有的类都是通过直接或间接地继承java.lang.Object类得到的。继承而得到的类称为子类，被继承的类称为父类。==Java不支持类的多重继承，即一个类从多个父类派生的能力==。

**考题样例**

下列描述不正确的是（    ）

A.     ==一个Java类总是且只能有一个直接父类（java.lang.Object除外）==

B.      一个Java类可以同时继承一个父类和实现一个接口

C.      一个Java类要么继承父类，要么实现一个接口，两者不能同时存在

正确答案：C



##### 6.2.2        接口的多继承性

**考点说明**

==Java只允许类的单继承，但可以通过接口实现多继承==：

1. 一个类只可以继承自另一个类，但可以实现多个接口

2. ==一个接口可以继承多个其它接口==

**考题样例**

下列描述正确的是（    ）

A.     一个接口可以继承多个接口

B.      一个Java类可以实现多个接口或者多个类

C.      一个接口可以继承一个类，但是可以继承多个接口

D.     上面说法都正确

正确答案：A

 

##### 6.2.3        构造函数

**考点说明**

1、==构造函数与类名必须相同，且不能有返回值。==

2、默认情况下，==编译器会为每个类默认生成一个无参构造函数。==

3、但是若类中定义了构造函数，==那么编译器就不会自动生成默认构造函数。==

4、==子类的构造函数中若没显式调用父类构造函数，那么会编译器会自动调用父类的默认构造函数，若父类没有默认构造函数，那么将会编译错误==。

**考题样例**

对于下列代码片断，描述正确的是（    ）

```java
class Base
{
    public Base()
    {
        System.out.println("Base()");
    }

    public Base(int i)
    {
        System.out.println("Base(int i)");
    }

}

public class MyOver extends Base
{
    public void MyOver(int i) 
    {

    }

    public static void main(String[] argv)
    {
        MyOver m = new MyOver(10);
    }
}
```

A.     运行此程序输出 "Base(int i)"

B.      运行此程序输出 "Base()"

C.      此程序运行后什么也不输出，因为MyOver的构造函数中没有显示调用父类构造函数

D.     此程序编译错误   # 不能有返回值

正确答案：D



#### 6.3      多态

##### 6.3.1        函数的重载和覆写以及两个的区别

**考点说明**

==成员函数重载，即几个名字相同、参数不同的成员函数==。

==成员函数覆写，即子类的成员函数与父类的成员函数名字，参数，返回值都相同时，子类成员函数将覆写父类成员函数的行为。==

重载与覆写在编译期有明显的区别：

1. 对于重载，每一个重载函数都有各自的参数列表对应。

2. 对于覆写，该函数只有唯一的参数列表。

方法覆盖的原则：

- ==覆盖方法的返回类型、方法名称、参数列表必须与原方法的相同。==

- ==覆盖方法不能比原方法访问性差（即访问权限不允许缩小）。==

- ==覆盖方法不能比原方法抛出更多的异常。==

- ==被覆盖的方法不能是final类型，因为final修饰的方法是无法覆盖的。==

- ==被覆盖的方法不能为private==，否则在其子类中只是新定义了一个方法，并没有对其进行覆盖。

- ==被覆盖的方法不能为static==。如果父类中的方法为静态的，而子类中的方法不是静态的，但是两个方法除了这一点外其他都满足覆盖条件，那么会发生编译错误；反之亦然。即使父类和子类中的方法都是静态的，并且满足覆盖条件，但是仍然不会发生覆盖，因为静态方法是在编译的时候把静态方法和类的引用类型进行匹配。

方法的重载：

​	Java父类和子类中的方法都会参与重载，例如，父类中有一个方法是 func(){ ... }，子类中有一个方法是 func(int i){ ... }，就构成了方法的重载。

覆盖和重载的不同：

- 方法覆盖要求参数列表必须一致，而方法重载要求参数列表必须不一致。
- 方法覆盖要求返回类型必须一致，方法重载对此没有要求。
- 方法覆盖只能用于子类覆盖父类的方法，方法重载用于同一个类中的所有方法（包括从父类中继承而来的方法）。
- 方法覆盖对方法的访问权限和抛出的异常有特殊的要求，而方法重载在这方面没有任何限制。
- ==父类的一个方法只能被子类覆盖一次==，而一个方法可以在所有的类中可以被重载多次。

**考题样例**

对于下面的Upton类

```java
public class Upton
{
    public static void main(String[] argv)
    {

    }

    public void amethod(int i)
    {

    }
    //Here
}
```

下面哪一个在替换//Here 后是不合法的？

A.     public int amethod(int z){}

B.      public int amethod(int i,int j){return 99;}

C.      protected void amethod(long l){ }

D.     private void anothermethod(){}

正确答案：A



##### 6.3.2        *static* 函数不能被覆写，所以*static*函数是没有多态性的

**考点说明**

==static函数是属于类的，它不能被覆写，不具多态性。==

**考题样例**

考虑下面的两个类ClassA、ClassB

```java
public class ClassA
{
    public void methodOne(int i)
    {

    }

    public void methodTwo(int i)
    {

    }

    public static void methodThree(int i)
    {

    }

    public static void methodFour(int i)
    {

    }
}

public class ClassB extends ClassA
{

    public static void methodOne(int i)
    {

    }

    public void methodTwo(int i)
    {

    }

    public void methodThree(int i)
    {

    }

    public static void methodFour(int i)
    {

    }
}
```

以下说法正确的是？( )

A. 子类ClassB中methodOne方法overrides父类ClassA的方法

B. 子类ClassB中methodTwo方法overrides父类ClassA的方法

C. 子类ClassB中methodThree方法overrides父类ClassA的方法

D. 子类ClassB中methodFour方法overrides父类ClassA的方法     

正确答案：B



##### 6.3.3        抽象类与接口

**考点说明**

1、抽象类不能被直接实例化；

2、抽象类可以没有抽象函数；

3、==有抽象函数的类必须是抽象类；==

4、==抽象函数只需声明，无需实现；==

5、==接口中不能有函数实现。==

**考题样例**

关于抽象类, 说法正确的是？()

A.     抽象类必须有一个以上的抽象函数

B.      包含抽象函数的类必须声明为抽象类

C.      抽象类可以实例化

D.     抽象类可以继承多个抽象类

正确答案：B



##### 6.3.4        面向接口和抽象编程(常用的)

**考点说明**

面向接口编程与面向实现编程是相对的，这里的接口并非就是Java中的interface，在这里，接口可以理解为Java中的interface或者抽象类，即“依赖倒置原则”中的依赖于抽象而非依赖于具体。

面向接口编程是设计可扩展性框架的必要条件，它要求我们：

- ==函数的参数和返回值尽量使用接口==。
- ==定义变量时尽量使用接口==。

**考题样例**

下列说法不正确的是？()

A.     面向接口编程即要求我们在设计时尽量用interface而不是class

B.     面向接口编程与面向实现编程是相对的，即在设计时应当尽量依赖于抽象（包括JAVA接口和抽象类），而不是具体。

C.      当一函数申明的返回值为类型A时，那么事实上它可以返回任意类型A的子类型对象

D.     当一变量申明为类型B时，那么事实上它可以被赋值为任意类型B的子类型对象

正确答案：A



#### 6.4      内部类

##### 6.4.1        静态内部类和非静态内部类的区别(静态内部类不持有外部类的引用)

**考点说明**

非静态内部类都持有其外部类对象的引用，可以访问其外部类的所有成员，==而静态内部类则不持有其外部类对象的引用，只能访问其外部类的静态成员==。

**考题样例**

关于静态内部类, 说法正确的是？()

A.     静态内部类不持有其外部类对象的引用，因此它不能访问其外部类的所有成员

B.      静态内部类持有其外部类对象的引用，因此它能访问其外部类的所有成员

C.      静态内部类在实例化时需要有其外围类的对象。

D.     ==当内部类无需访问其外部类成员时，应当尽量使用静态内部类==

正确答案：D



### 7      并发与多线程

#### 7.1      线程

##### 7.1.1        线程的基础知识

**考点说明**

现代操作系统要执一个程序时，就会创建一个新的进程（process）。从java虚拟机看，==进程是资源分配的单位，而线程是调度的单位，每个进程至少有一个运行的线程（thread）==。线程是程序执行代码一条路径，每个线程均有自己的局部变量，程序计数器（当前执行指令的指针）以及生存期。

大部分现代操作系统允许在一个进程内并发运行多个线程。当操作系统启动Java虚拟机时，就创建了一个新进程，在该进程内可以创建很多线程。

生成线程对象的方法：

- 从 java.lang.Thread 类继承一个类出来，并覆写 run() 函数，然后构造这个类的实例即可。

- 用一个类实现 java.lang.Runnable 接口，构造这个类的实例，然后通过这个实例构造一个 java.lang.Thread 类的实例。


**考题样例**

在一个Java进程中，==每个线程都拥有自己的栈(Stack)和局部变量，但所有的线程共享同一个堆(Heap)内存==。( )

A．正确    B．错误

正确答案：A



##### 7.1.2        尽管*run()函数是线程代码执行的起点但要启动一个线程只能调用*start()函数(必须了解的)

**考点说明**

==只能通过调用start()函数启动一个新线程==，对start()函数的调用会立即返回，不会等待新线程开始执行，原线程不受干预地继续执行start()后的语句。在start()中，JVM得到通知并生成新线程，即产生了一个新的调度单位，在不久将来的某一不可预测的时间内(一般是非常短的时间，毫秒级)新线程会被调度。新线程被调度时，由JVM调用新线程对象的run()函数。

在==代码直接调用 run() 函数并不会新生新线程，只是在原来的线程中执行 run() 函数中的代码而已==。

**考题样例**

运行以下程序，结果是：

```java
public class TwoThread extends Thread
{
    public void run()
    {
        for(int i = 0; i < 10; i++)
        {
            System.out.print(i);
        }
    }

    public static void main(String[] args)
    {
        TwoThread tt = new TwoThread();
        tt.run();
        for(int i = 0; i < 10; i++)
        {
            System.out.print(i);
        }
    }

}
```

A.     01234567890123456789

B.      00112233445566778899

C.      两个0~9的序列间插,运行结果可能不同

正确答案：A



##### 7.1.3        实现 *Runnable* 接口不等于生成了线程(常错的)

**考点说明**

在 Java 语言中，==产生线程对象的唯一方法是构造 Thread 类或其子类的实例==。虽然实现了 Runnable 接口的对象可以做为参数用于构造 Thread 对象，==但实现了 Runnable 接口的类不等同于它就是 Thread 的子类==。如果不跟 Thread 类扯上关系，那Runnable 接口是一个普通的接口，跟其它的接口没什么两样。

**考题样例**

实现了 Runnable 接口的对象，就是一个线程。()

A．正确    B．错误

正确答案：B



##### 7.1.4        只有线程运行结束了，线程对象引用的资源才有可能被回收(常错的)

**考点说明**

线程的栈是垃圾回收的其中一个起点（官方说法是“root”），==所以只要线程运行没有结束，线程引用的资源（包括线程对象本身、线程对象的所有成员变量、栈中的所有临时变量）无法回收==。

**考题样例**

有以下代码，说法错误的是：(  )

```
Thread t = new Thread()
{
    public void run()
    {
        while(true)
        {
        }
    }
};
t.start();
t = null;
```

A.  这段代码产生了一个线程。

B.  当变量t被赋值为null后，上述代码产生的线程对象符合垃圾回收的条件。

C.  这段代码生成的线程将消耗极多的资源。

D.  当这段代码执行后，即使main()函数返回进程也不会终止、除非在别的线程调用System.exit()函数。

正确答案：B



#### 7.2      同步(必须了解的)

##### 7.2.1        *synchronized* 关键字、类锁与对象锁

**考点说明**

在 java 语言中，==synchronized 关键字用来声明针对某个对象的互斥锁，所有的互斥锁都是针对某个具体对象而产生的，用来保证某一段时间内最多仅有一个线程可以访问通过同一个对象互斥锁定的代码==。在这里需要注意，这些代码未必是同一段代码。

在 java 语法中，有三种声明互斥代码的方法：

1)        同步块（对象锁），用语法 “synchronized(对象){ 需要互斥的代码 }” ==在一个对象上声明互斥锁，在运行期，互斥锁通过这个对象进行互斥操作，可以确保该对象的所有同步块是多线程互斥的==。

2)        成员函数锁（实例锁），在成员函数上增加 synchronized 修饰。==在运行期，互斥锁通过当前的 this 对象进行互斥操作，可以确保该对象的所有同步函数以及该对象的所有同步块是多线程互斥的。== 相当于为整个函数的代码声明了一个synchronized(this) 同步块。

3)        静态函数锁（类锁），在静态函数上增加 synchronized 修饰。在运行期，==互斥锁通过类的 class 对象进行互斥操作，可以确保该类的所有静态同步函数以及该类的 class 对象同步块是多线程互斥的==。相当于为整个函数的代码声明了一个 synchronized(类名.class) 同步块。注意这里的“类名.class”和通过对象实例调用 getClass() 函数取得的对象是一样的。

**考题样例**

有以下代码，标识为 A/B/C/D 的四行中哪一行获得锁的对象与其他的是不同的？( )

```java
public class Foo
{
    private static synchronized void synchronizedStatic(){}
    private synchronized void synchronizedInstance(){}
    public static void main(String[] args)
    {
        Foo foo = new Foo();
        synchronized(foo.getClass()){}  // (A)
        foo.synchronizedStatic();       // (B)
        foo.synchronizedInstance();     // (C)
        synchronized(Foo.class){}       // (D)
    }
}
```

正确答案：C



##### 7.2.2        *synchronized* 块应当最小化

**考点说明**

使用synchronized进行并发控制对于多线程环境下共享数据的安全致关重要，==但当线程需要同时持有多个锁时可能产生死锁问题，同时获取和释放锁的任务给处理器添加了更多的工作量，因而减慢了执行速度==。

**考题样例**

线程同步是为了保证在多线程环境下共享数据的安全，要尽可能使用同步机制()

A．正确     B．错误

正确答案：B



###  8      *util*库

####  8.1      一些常用类之间的区别

##### 8.1.1        *Collection*和*Collections*的区别

**考点说明**

==Collections是个java.util下的类，它包含有各种有关集合操作的静态函数实现对各种集合的搜索、排序、线程安全化等操作，但是Collections不允许被继承==。Collection是个java.util下的接口，==它是各种集合结构的父接口，继承于它的接口主要有Set和List==。

需要熟悉两者之间的区别和基本用法。

![1563265054521](D:\Notes\raw_images\1563265054521.png)

![1563265187984](D:\Notes\raw_images\1563265187984.png)

**考题样例**

下面说法正确的是：

A. Collection是个java.util下的接口

B. Collections是个java.util下的接口

C.Set、List和Map继承于Collection

D. Collections继承于Collection

正确答案：A



##### 8.1.2        *Collections*和*Arrays*的区别

**考点说明**

数组是一种简单的复合数据类型，它的特点是用来存储和访问一连串对象效率是最高的，但其容量固定且无法改变。==Arrays是个java.util下的类，它拥有五个static函数：equals()、fill()、sort()、binarySearch()和asList()，这几个函数中可以接受的参数是基本类型和对象（以及他们的数组）==，这与Collections在参数上面存在最大的不同。如果说Arrays是对数组的一个扩展类，那么Collections就是对应于Collection的一个扩展类。Collections类中的函数也是static，主要提供对Collection的一些基本操作（查找、排序、取值等等）。==用这两个类的时候，通常只能用它们的静态函数，不必（也不能）构造它的对象==。

需要熟悉这两个类的区别和基本用法。

**考题样例**

下面程序运行结果是：

```
import java.util.Arrays;

public class test
{
  public static void main(String[] args)
  {
    String[] t1 = {"F", "E", "D", "C", "B", "A"};
    Arrays.sort(t1);
    System.out.println(Arrays.asList(t1));
  }
}
```

A.     [A, B, C, D, E, F]

B.      [F, A, D, B, C, E]

C.      [F, E, D, C, B, A]

D.     产生数组越界异常

正确答案：A



##### 8.1.3        *HashMap*和*Hashtable*的区别

**考点说明**

HashMap类与Hashtable类对外的接口比较类似，==只不过Hashtable中的大部分的函数都是同步的，所以Hashtable是多线程安全的==，而HashMap则不是。==由于HashMap非线程安全，效率上可能高于Hashtable==。==另外HashMap允许将null作为一个entry的key或者value，而Hashtable不允许==。

他们都实现了Map接口，Hashtable和HashMap采用的hash/rehash算法都很相似。

需要了解两者之间的区别和基本用法。

**考题样例**

下面描述错误的是：

A.     Hashtable不允许null值(key和value都不可以)，而HashMap允许将null作为一个entry的key或者value

B.     Hashtable是线程安全的，也就是说是同步的，HashMap线程不安全的，不是同步的

C.     Hashtable和HashMap都实现了Table接口

D.    Hashtable是继承于一个陈旧的类，多线程时可以对HashMap进行同步，因此在很多场合可以用HashMap代替Hashtable。 

正确答案 C



#### 8.1.4        *Vector*和*ArrayList*的区别

**考点说明**

==Vector是多线程安全的，也就是说是同步的==，==而ArrayList是多线程不安全的，不是同步的==。==当需要增长时,Vector默认增加一倍的容量，而ArrayList却是增加 0.5 倍的容量==。

需要了解两者之间的区别和基本用法。

**考题样例**

下面说法错误的是：

A.     ==不管是ArrayList还是Vector，在它们内部都是使用一个Object[]来保存数据的==

B.      ==随着数据的插入，当内部数组空间不够了ArrayList还是Vector就要扩展它的大小，Vector默认增加一倍的容量，而ArrayList却是增加 0.5 倍的容量==

C.      ==特别是对效率优先的代码来说使用一个普通的原始的数组来代替Vector和ArrayList==

D.     Vector中不允许有重复的元素，而ArrayList却允许

正确答案 D



#### 8.2      *Map*接口(常用的)

##### 8.2.1        *key-value*结构

**考点说明**

Map提供key到value的映射。==一个Map中不能包含相同的key，每个key只能映射一个value==。Map接口通过put()函数写入数据，而通过get()函数取出数据。另外Map接口的keySet()、entrySet()函数返回一个Set、通过values()函数返回所有value的列表，通过这些函数可以实现Map数据向其它容器接口的转换。

需要熟悉Map的基本原则，会将Map数据（key/value）向其它容器接口转换。

**考题样例**

下面程序运行结果描述正确的是：

```java
import java.util.ArrayList;
import java.util. TreeMap;
import java.util.List;

public class test
{

  public static void main(String[] args)
  {
    TreeMap tm = new TreeMap();
    tm.put("a","A");   
    tm.put("b","B");   
    tm.put("c","C");

    List result=new ArrayList(tm.entrySet());  
    System.out.println(result);
  }
}
```

A.    ==[a=A, b=B, c=C]==

B.      [a, b, c]

C.      [A, B, C]

D.     运行出现异常

正确答案：A



##### 8.2.2        *Map*中不允许重复*key*

**考点说明**

Map中的key是不可重复的，用相同的key存入不同对象的时候，仅最后一次保存的对象有效。必要时，可以调用containsKey(Object key)来判断判断key是否已经存在。

注：==不要通过get()函数返回值是否是null判断key是否存在，当返回null时，有两种可能性：一是Map中确实没有这个key；二是Map中有这个key，但人家保存的value就是null==。

熟悉Map的基本原则和常规操作。

**考题样例**

下面描述错误的是：

A.     Map中的key不可重复，但是value可以重复

B.      当通过一个key从Map中取值返回为null时说明Map中不存在该key

C.      查看Map是否存在指定的“keyX”时可以用函数containsKey(keyX)来判断

D.     可以通过迭代器遍历Map中的key和value

正确答案：B



#### 8.3      *List*接口(常用的)

##### 8.3.1        几个常用的*List*

ArrayList、Vector、LinkedList

**考点说明**

==ArrayList和Vector都是使用数组方式存储数据==，此数组元素数大于实际存储的数据以便增加和插入元素，它们都允许直接按序号索引元素，但是插入元素要涉及数组元素移动等内存操作，所以索引数据快而插入数据慢，==Vector由于使用了synchronized函数（线程安全），通常性能上较ArrayList差==，而LinkedList使用双向链表实现存储，按序号索引数据需要进行前向或后向遍历，但是插入数据时只需要记录本项的前后项即可，所以插入速度较快。==另外LinkedList是非线程安全的==。

需要了解三者的不同和基本使用。

**考题样例**

下面说法不正确的是：

A.     查找一个指定位置的数据，vector和arraylist 使用的时间都是O(1)，而移动一个指定位置的数据花费的时间为O(n-i)，n为总长度

B.      LinkedList移动一个指定位置的数据所花费的时间为 O(1), 而查询一个指定位置的数据时花费的时间为O(i)

C.      ArrayList、Vector和LinkedList都实现了List接口

D.     LinkedList类不允许继承，该类定义的函数也不允许重载或覆写

正确答案：D



####  8.4      *Set*接口

##### 8.4.1        *Set*的特点

**考点说明**

Set拒绝持有重复的元素，另外不能通过任何索引的函数来操作Set对象。

了解Set基本原则，需要熟悉HashSet、TreeSet和LinkedHashSet的基本用法。

**考题样例**

下面说法错误的是：

A.     Set拒绝持有重复的元素

B.     可以向Set插入多个null元素，它也能保存多个重复的null元素

C.     不能通过任何索引的函数来操作Set元素

D.    ==HashSet插入和查找性能优于TreeSet，但TreeSet能够维护其内元素的排序状态==

正确答案：B



###  9      *Java*开发环境

####    9.1      *classpath*指定

#####  9.1.1        类的加载顺序

**考点说明**

==JVM在加载类的时候，如果没有给出classpath则使用当前路径；如果给出了classpath则使用classpath==。特别注意此时==如果classpath中没有说明当前路径，即没有包含(.)，则JVM不会在当前路径搜索==。

==在classpath中如果有重名类（包名、类名全相同），JVM只加载第一个找到的类，其它被忽略==。

**考题样例**

样题一：

文件夹temp的目录结构如下

│  myPackage.jar

└─hello

​        Hello.class

​        Hello.java

Hello.java引用了myPackage.jar中内容。在windows操作系统中执行以下可以运行Hello程序的命令是( )

A.     java hello/Hello.class

B.      java hello.Hello

C.      java -cp myPackage.jar hello.Hello

D.     ==java -cp myPackage.jar; . hello.Hello==

正确答案：D

 

样题二：

如果两个Jar包p1.jar和p2.jar都有类com.Tool，但是实现不同。

p1.jar中com.Tool有方法

public static void out(String msg)

{

​    System.out.println("p1:" + msg);

}

p2.jar中com.Tool有方法

public static void out(String msg)

{

​    System.out.println("p2:" + msg);

}

类Main使用jar包p1.jar编译通过，其代码如下

public class Main

{

​    public static void main(String[] args)

​    {

​        com.Tool.out("Hello");

​    }

}

在windows操作系统中执行命令行 java -cp p2.jar;p1.jar;. Main会输出什么?

A.     因为Main是通过p1.jar编译的，所以输出为p1:Hello

B.      ==因为-cp 参数种p2.jar放在了前面，所以先找到p2中内容，因此输出p2:Hello==

C.      -cp参数中有重复的内容，系统不能确定使用哪一个，报错

D.     不能确定输出内容，系统随机选择

正确答案：B



#### 9.2      跨平台

##### 9.2.1        换行

**考点说明**

各平台在处理换行时，所使用的字符串有所不同。==Windows默认使用"\r\n"，而Linux则是用"\n"==。在作与换行符相关操作时==建议使用System.getProperty("line.separator")进行区别==。System.getProperty("line.separator")会返回当前系统使用的换行字符串。

**考题样例**

样题一：

以下说法正确的是( )

A.     在输出换行动作时，建议使用System.getProperty("line.separator")换行，这可以做到平台无关。

B.      在输出换行时最好直接使用"\n"，调用System.getProperty("line.separator")会影响效率。

C.      输出换行时应该使用"\r\n","\n"只适用于Linux系统。

D.     全不对

正确答案：A

 

样题二：

下面关于文件分割符的说法错误的是

A.     各种系统不区分\和/

B.      ==Windows默认使用\做为文件分割符==

C.      ==unix只能使用/做为文件分割符==

D.     ==使用在需要使用文件分割符的地方使用System.getProperty("file.separator") 可以保证应用程序的跨平台特性==

正确答案：A



##### 9.2.2        文件路径

**考点说明**

各平台所使用的路径分隔符不一样，有时模糊使用"/"或"\"没有影响，有的系统对此是有严格要求的。System.getProperty("file.separator")返回系统默认使用的文件分隔符。建议使用System.getProperty("file.separator")进行拼接或拆分操作。

除使用System.getProperty("file.separator")以外，==还可以使用 File.separator 或者File.sepatatorchar 取得系统默认使用的文件分隔符，后者比前者使用更方便，并且不容易出错==。

**考题样例**

下面关于文件分割符的说法错误的是

A.     各种系统不区分\和/

B.      Windows默认使用\做为文件分割符

C.      unix只能使用/做为文件分割符

D.     使用在需要使用文件分割符的地方使用System.getProperty("file.separator") 可以保证应用程序的跨平台特性

正确答案：A



#### 9.3      本地化和国际化

##### 9.3.1        *ResourceBundle*

**考点说明**

资源包包含特定于语言环境的对象。当程序需要一个特定于语言环境的资源时（如 字符串资源），==程序可以从适合当前用户语言环境的资源包中装入它==。以这种方式可以编写独立于用户语言环境的程序代码，它将资源包中大部分（如果不是全部）特定于语言环境的信息隔离开来。

需要了解Java在本地化时的机制和资源文件的格式。

**考题样例**

某资源文件中有如下内容：

\#Key1=Value1

Key2 = Value2

Key3 =             Value3

Key4 = Super Value4

如下说法正确的是：

A.     可以正常读出Key1为"Value1"

B.      可以正常读出Key2为"Value2"

C.      可以正常读出Key3为"    Value3"

D.     可以正常读出Key4为"SuperValue4"

正确答案：B



### 10  *Java*的*GUI*库

#### 10.1   事件（**）

**考点说明**

Java的各种动作事件，是通过的方式处理的。被对象可能是控件，也可能是模型，这取决于需要的内容。如果需要的内容和显示有关，则被对象应该是控件；如果需要的内容是控件表示的数值，则被对象是模型。

在各事件中javax.swing.event.ListSelectionListener相对比较复杂。比如在表格中使用javax.swing.event.ListSelectionListener捕获表格的行选择改变事件，此时valueChanged方法被执行的次数可能不是1。在用鼠标选择时，顺序如下：

(1)、按下鼠标左键，执行一次valueChanged，e.getValueIsAdjusting()==true

(2)、拖动鼠标，改变选择的行数，执行一次valueChanged，e.getValueIsAdjusting()==true

(3)、放开鼠标左键，执行一次valueChanged，e.getValueIsAdjusting()==false

多数情况下我们需要关注的是用户选择后，放开鼠标左键的结果，只有(3)这种情况才是我们真正想要执行的情况。因此，在使用选择事件时要注意是否需要检查e.getValueIsAdjusting()的状态。

**考题样例**

样题一：

不可以将下面哪些器接口添加到==TextArea对象==中？()

A.     TextListener

B.      ==ActionListener==

C.      MouseMotionListener

D.     MouseListener

E.      ComponentListener

正确答案：B

 

样题二：

下面哪些器接口不可以添加到TextField对象中？()

A.     ActionListener

B.      FocusListener

C.      MouseMotionListener

D.     WindowListener

正确答案：D

 

样题三：

下面可以监视JTextField内容改变的器接口是(  )

A.     ActionListener

B.      AdjustmentListener

C.      ChangeListener

D.    ==DocumentListener==

正确答案：D

 

样题四：

关于javax.swing.event.ListSelectionListener事件参数javax.swing.event.ListSelectionEvent的getValueIsAdjusting()方法中错误说法是( )

A.     使用鼠标选择内容时，如果只选择一条记录，则选择改变事件只被触发一次，且getValueIsAdjusting()等于false

B.      鼠标在按下之后触发的任何与选择有关的动作，都会触发改变事件，且e.getValueIsAdjusting()都等于true

C.      鼠标在选择多条记录后，松开左键会触发一次改变事件，且e.getValueIsAdjusting()等于false

D.     使用键盘改变选择内容时，选择改变事件只被触发一次，且e.getValueIsAdjusting()等于false

正确答案：A

 

#### 10.2   数据（模型）

**考点说明**

==Java控件采用MVC模型，有效的将界面显示和其表示的数据分开。==不同的控件有不同的模型与其对应，常用控件及其模型对应关系如下：

| 控件                   | 模型                         | 获取模型方法  |
| ---------------------- | ---------------------------- | ------------- |
| javax.swing.JTextField | javax.swing.text.Document    | getDocument() |
| javax.swing.JTextArea  | javax.swing.text.Document    | getDocument() |
| javax.swing.JTable     | javax.swing.table.TableModel | getModel()    |
| javax.swing.JList      | javax.swing.ListModel        | getModel()    |
| javax.swing.JComboBox  | javax.swing.ComboBoxModel    | getModel()    |
| javax.swing.JTree      | javax.swing.tree.TreeModel   | getModel()    |

可选择的控件，被选择内容也有自己的选择模型，常用控件及其选择内容模型对应关系如下：

| 控件               | 模型                                | 获取模型方法        |
| ------------------ | ----------------------------------- | ------------------- |
| javax.swing.JTable | javax.swing.ListSelectionModel      | getSelectionModel() |
| javax.swing.JList  | javax.swing.ListSelectionModel      | getSelectionModel() |
| javax.swing.JTree  | javax.swing.tree.TreeSelectionModel | getSelectionModel() |

**考题样例**

样题一：

下面哪种类型的model承载了文本控件的实际内容？()

A.     TableModel

B.      ListModel

C.      ==Document==

D.     TreeModel

正确答案：C

 

样题二：

哪些swing组件使用ListSelectionModel？()

A.      JList and JCombox

B.      JPopupMenu and JTable

C.      JTable and JCombox

D.     ==JList and JTable==

正确答案：D

 

#### 10.3   常用的布局管理器

​       Java 用一个丰富出色的概念来实现动态布局：容器内所有组件都由一个布局管理器进行定位。在容器中，可以调用setLayout()函数设置各种的布局管理器。

##### 10.3.1    *BorderLayout*

**考点说明**

BorderLayout允许我们选择每个组件的放置位置，可以选择把组件放置在容器的中部、北部、南部、东部及西部。

**考题样例**

在下列布局管理器中，哪个布局管理器可以实现东西南北中来放置控件？ ()

A.  BorderLayout

B.  CardLayout

C.  FlowLayout

D.  GridBagLayout

正确答案：A



#####  10.3.2    *FlowLayout*

**考点说明**

FlowLayout的特点是在一行上水平排列组件，直到没有空间为止，然后开始新的一行。当用户缩放窗口时，布局管理器自动地调整组件的位置使其填充可用的空间。

**考题样例**

在下列布局管理器中，哪种布局管理器从左到右、从上到下排列组件？ ()

A.  BorderLayout

B.  CardLayout

C.  FlowLayout

D.  GridBagLayout

正确答案：C



##### 10.3.3    *GridLayout*

**考点说明**

GridLayout的特点是组件排列方式是在一行中并且大小相同而且充满整个容器空间。

**考题样例**

容器中要放置一行组件，这些组件排列方式是在一行中并且大小相同而且充满整个容器空间（中间没有间隔），如下图所示，请问使用哪个布局管理器比较合适？ ()

![1563265383438](D:\Notes\raw_images\1563265383438.png)

A.  BorderLayout

B.  CardLayout

C.  FlowLayout

D.  GridLayout

正确答案：D



##### 10.3.4    *GridBagLayout*

**考点说明**

GirdBagLayout按行列排列所有的组件，是最复杂的布局管理器。它的各种设置都是通过GridBagConstraints来实现的。GridBagConstraints的参数包括gridx,gridy,gridwidth,gridheight,weightx,weighty,anchor,fill,insets,ipadx,ipady。可以通过这些参数的不同组合，绘制出满意的界面布局。

**考题样例**

在下列Java布局管理器中，哪种布局管理器像电子表格一样，可以实现用行列来排列组件？ ()

A.  BorderLayout

B.  CardLayout

C.  FlowLayout

D.  GridBagLayout

正确答案：D

 

#### 10.4   事件派发线程

#####  10.4.1    灰屏现象

**考点说明**

==在 Java AWT 的图形界面中，有一个线程专门用于派发、处理界面的事件，这个线程就是事件派发线程。==当图形界面程序启动时，Java的VM自动启动事件派发线程。所有的事件通知，例如调用actionPerformed()或paintComponent，都在事件派发线程中运行。当事件派发线程被阻塞时，可能会导致用户界面不响应，出现灰屏现象。

在一个Swing应用中，本质上来说所有的代码都应该包含在事件处理器中，以便响应用户界面和重绘的请求。以下是Core Java 2对Swing操作用户界面的几条建议：

1．  ==如果一个动作占用的时间太长，就启动一个新的线程来执行它。==

2． ==如果一个动作在输入或输出上阻塞了，就启动一个新的线程来处理它。如处理网络连接。==

3． ==如果需要等待指定时间，不要让事件派发线程睡眠，而应该使用定时器事件。==

4．  ==在线程中做的事情，不要直接操作界面。==

**考题样例**

造成Swing应用程序灰屏的根本原因是()

A.  在非事件派发线程中没有使用invokeLater()和invokeAndWait()方法

B.  在事件派发线程中执行了大量的IO操作

C.  主线程被阻塞

D.  ==事件派发线程被阻塞==

正确答案：D

 

#### 10.5   *SwingUtilities*类(常用的)

#####  10.5.1    *invokeLater() 和*invokeAndWait()函数的作用

**考点说明**

由于AWT/Swing的众多组件没有多线程保护机制，所以，从事件派发线程之外的线程访问Swing组件是不安全的。为了解决这个问题，==SwingUtilities类提供了invokeLater()和invokeAndWait()两个方法，用于将调用的方法放入到事件派发线程中执行==。

注意：==可以从事件派发线程调用invokeLater()，却不能从事件派发线程调用invokeAndWait()==。如果从事件派发线程调用invokeAndWait()，则将抛出一个 Error。

**考题样例**

采用以下措施,可以很好的抑制Swing应用程序灰屏的是 ()

A.  ==不要在事件派发线程中执行耗时操作==

B.  在事件派发线程中执行耗时操作

C.  将耗时操作放到invokeLater()方法中

正确答案：A



### 11  *Web*

##### 11.1.1    *http*是无状态的协议，如何跟踪客户的状态

**考点说明**

Web服务器跟踪客户的状态通常有4种：

- 建立含有跟踪数据的隐藏表单字段

- 重写包含额外参数的URL

- 使用持续的Cookie

- 使用Session(会话)机制


**考题样例**

下面哪种方法不可以用来跟踪客户状态（）

A.  Session

B.  Cookie

C.  URL rewriting

D.  Hidden Field

E.  Request

正确答案：E



##### 11.1.2    *Get* 和*Post* 的区别

**考点说明**

==Get 请求的参数是添加在 URL后面,会有安全问题，对于提交的数据大小有限制==

==Post 请求的参数封装在请求体中，比较安全，可以支持大数据量的提交==

**考题样例**

关于Servlet中 doGet() 与 doPost() 函数的区别描述错误的是（）

A.     调用Servlet的doGet()和doPost()函数与提交的表单有关

B.      如果函数类型为GET,就调用doGet()函数，如果为POST就调用doPost()函数

C.      doPost()函数常用于处理客户端的请求，doGet()用于服务器端向客户端发送响应

D.     doGet()和doPost()都可以处理业务逻辑

正确答案：C



##### 11.1.3    *Servlet*生命周期的3个函数，及其用途考点

**考点说明**

Servlet的生命周期可以分为3个阶段：==初始化阶段、响应客户请求阶段和终止阶段==。Javax.servlet.Servlet接口中定义了3个函数init(),service和destory()，它们分别在不同的阶段调用。

![1563265420702](D:\Notes\raw_images\1563265420702.png)

**考题样例**

关于Servlet生命周期的init(), service(), destroy()方法，那些说法是错误的。

A.     init() 和destory()方法只会被调用一次

B.      service() 方法会多次调用

C.      以上说法都不对

正确答案：C



##### 11.1.4    在*JSP*页面中可以包含的元素

**考点说明**

JSP文件（扩展名为.jsp）可以包含如下内容：

- l  JSP指令（或称为指示语句）

- l  JSP声明

- l  Java程序片断（Scriptlet）

- l  变量数据的Java表达式

- l  隐含对象


**考题样例**

以下选项不是JSP组成部分的是（）

A.     标签

B.      脚本

C.      隐含对象

D.     插件

正确答案：D



##### 11.1.5   *JSP*页面如何引用*JavaBean,设置*bean的范围

**考点说明**

JSP最强有力的一个方面是能够使用JavaBean组件体系。JavaBean往往封装了程序的页面逻辑，它是可重用的组件。<jsp:us erBean>标签用来在JSP页面中创建一个Bean实例，并指定它的名字以及作用范围。它保证对象在标签指定的范围内可以使用。定义的语法如下：

==<jsp:useBean  id="id"  scope="page|request|session|application"  typeSpec/>==

例如：

<jsp:useBean id="shopcart" scope="session" class="demo.Carts" />

**考题样例**

在JSP页面中引用javabean的正确形式是（）

A.     <jsp:useBean id="name" class="package.class" scope="request">

B.      <file:useBean id="name" class="package.class" scope="page" >

C.      <page:useBean id="name" class="package.class" scope="session" >

D.     <jsp:useBean id="name" class="package.class" scope="response" >

正确答案：A



##### 11.1.6    转发和重定向：*Redirect*与*Forward*的区别

**考点说明**

==Redirect 浏览器会重新发送请求到服务器，增加了网络来回的负担。==

==Forward 在服务端转发，跟浏览器没有关系，不会产生新的请求。==

**考题样例**

从Servlet转发到jsp页面使用下面哪个函数是正确的（）

A.     redirect

B.     ==forward==

C.      redirected

D.     mapping

正确答案：B



##### 11.1.7    常见的*request*对象的函数

**考点说明**

了解常用的request对象的函数，例如：==获取页面传递参数可以使用getParameter()==，==获取请求的消息头信息可以使用getHeader()==,==获取cookie的信息可以使用getCookies()==。

**考题样例**

Servlet查找从客户端传送过来的信息，以下函数错误的是（）

A.     getParameter()

B.      getAttribute()

C.      getParameterValues()

D.     getParameterNames()

正确答案：B



##### 11.1.8    静态包含和动态包含的区别

**考点说明**

==静态包含<%@include %> 在编译期完成，编译后内容不可变，执行效率高==。如果要修改，需要重新编译。

==动态包含<jsp:include …>的内容是动态改变的，在执行期才确定，增加了运行的开销。==

**考题样例**

以下哪一种描述表示JSP页面中的动态INCLUDE（）

A.     <jsp:include page="included.jsp" flush="true">

​			<jsp:param  name="a1"  value="<%=request.getParameter("name")%> "/>

​		</jsp:incl ude>

B.      <%@ include file="included.jsp" %>

C.      <%page include file="included.jsp"%>

D.     <%jsp include file="included.jsp"  flush="true"%>

正确答案：A



##### 11.1.9    *Servlet*过滤器及其作用

**考点说明**

Servelt过滤器能够对Servlet容器的请求和响应对象进行检查和修改。多个过滤器可以串联使用，形成管道效应，协同修改请求和响应对象。

**考题样例**

下面关于过滤器的说法那个是不正确的（）

A.     过滤器可以串联使用

B.      过滤器可以拦截的Web组件包括Servlet,Jsp或者html文件

C.      在过滤器中可以修改request和response对象的内容

D.     javax.servlet.Filter是一个抽象类

正确答案：D
