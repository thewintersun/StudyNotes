## 24式加速你的Python

作者 | 梁云1991

来源 Python与算法之美

https://www.huaweicloud.com/articles/c56fe4d4cea0ec80e344fd31ab257797.html

### **一、分析代码运行时间**

**第1式，测算代码运行时间**

平凡方法

![24式加速你的Python2](https://res-static.hc-cdn.cn/fms/img/5a87080c1e0903446cb13b0d84ebba881603440421205)

快捷方法（jupyter环境）

![24式加速你的Python3](https://res-static.hc-cdn.cn/fms/img/c65e26c21f47a9936c21e516eaff591e1603440421205)

**第2式，测算代码多次运行平均时间**

平凡方法

![24式加速你的Python4](https://res-static.hc-cdn.cn/fms/img/e09df4031003a05dc2da3d919c14c3cd1603440421206)

快捷方法（jupyter环境）

![24式加速你的Python5](https://res-static.hc-cdn.cn/fms/img/7a8a5c67276ebdac08e5fe1b84074f101603440421206)

**第3式，按调用函数分析代码运行时间**

平凡方法

![24式加速你的Python6](https://res-static.hc-cdn.cn/fms/img/cadd21e72515b0b1a8a675e129e276891603440421206)

![24式加速你的Python7](https://res-static.hc-cdn.cn/fms/img/0967886d193d3b7f1a41a6633e1c24951603440421206)

快捷方法（jupyter环境）

![24式加速你的Python8](https://res-static.hc-cdn.cn/fms/img/f06703c17bf6c5a6e806561ae7b18b921603440421206)

**第4式，按行分析代码运行时间**

平凡方法

![24式加速你的Python9](https://res-static.hc-cdn.cn/fms/img/96fbd4386c6307acee2f395b106c43811603440421207)

![24式加速你的Python10](https://res-static.hc-cdn.cn/fms/img/b02973977ab9a726e561095f4ace8a671603440421207)

![24式加速你的Python11](https://res-static.hc-cdn.cn/fms/img/0172d4be757068f109153cadba86227c1603440421207)

![24式加速你的Python12](https://res-static.hc-cdn.cn/fms/img/4d1d181b8dcad5214fc19ba8363e72161603440421207)

快捷方法（jupyter环境）

![24式加速你的Python13](https://res-static.hc-cdn.cn/fms/img/15dd43b779f791d406b90c8e967cf5411603440421208)

###  

### **二、加速你的查找**

**第5式，用set而非list进行查找**

低速方法

![24式加速你的Python14](https://res-static.hc-cdn.cn/fms/img/c577491f7bce42ae85b2b737395a2fb01603440421208)

![24式加速你的Python15](https://res-static.hc-cdn.cn/fms/img/bec0cbdc1b893a50ee15ed5f133a2d6a1603440421208)

高速方法

![24式加速你的Python16](https://res-static.hc-cdn.cn/fms/img/59ece5ee8f774ed96a1923a3f56e9b401603440421208)

**第6式，用dict而非两个list进行匹配查找**

低速方法

![24式加速你的Python17](https://res-static.hc-cdn.cn/fms/img/e6bc57d4e861b082a92bdf4348be1e8f1603440421209)

![24式加速你的Python18](https://res-static.hc-cdn.cn/fms/img/861b9090281d4cddbb28442ba359edd51603440421209)

高速方法

![24式加速你的Python19](https://res-static.hc-cdn.cn/fms/img/887c79d97ee253c5e2989b76159fa9b41603440421209)

###  

### **三、加速你的循环**

**第7式，优先使用for循环而不是while循环**

低速方法

![24式加速你的Python20](https://res-static.hc-cdn.cn/fms/img/73209b808fee6fc261a8c53f694ee9af1603440421209)

高速方法

![24式加速你的Python21](https://res-static.hc-cdn.cn/fms/img/de46dc8ebc7d72314376423d8c0073041603440421212)

**第8式，在循环体中避免重复计算**

低速方法

![24式加速你的Python22](https://res-static.hc-cdn.cn/fms/img/fd68c5519ee7fb749d8c81d6e2b0591d1603440421212)

高速方法

![24式加速你的Python23](https://res-static.hc-cdn.cn/fms/img/e657eff0d91df20087c14ef320d912c21603440421213)

### **四、加速你的函数**

**第9式，用循环机制代替递归函数**

低速方法

![24式加速你的Python24](https://res-static.hc-cdn.cn/fms/img/df2e8b40c56d36b41f8b99bca8491adc1603440421213)

高速方法

![24式加速你的Python25](https://res-static.hc-cdn.cn/fms/img/5380e77d70fd97fac876c17629b30fbc1603440421213)

**第10式，用缓存机制加速递归函数**

低速方法

![24式加速你的Python26](https://res-static.hc-cdn.cn/fms/img/877c58e6d1a49001d1123cb3108059821603440421214)

高速方法

![24式加速你的Python27](https://res-static.hc-cdn.cn/fms/img/03696ac92c5f8832756a507efca67b9f1603440421214)

**第11式，用numba加速Python函数**

低速方法

![24式加速你的Python28](https://res-static.hc-cdn.cn/fms/img/052f5061a724fe8d075cf05f281aa4021603440421214)

高速方法

![24式加速你的Python29](https://res-static.hc-cdn.cn/fms/img/f82d291c75623b14b85e4497a1e9c6061603440421215)

###  

### **五、使用标准库函数进行加速**

**第12式，使用collections.Counter加速计数**

低速方法

![24式加速你的Python30](https://res-static.hc-cdn.cn/fms/img/780368497403898052cf073e682888121603440421215)

高速方法

![24式加速你的Python31](https://res-static.hc-cdn.cn/fms/img/366b418511ba5dda60756bd7bedd84771603440421215)

**第13式，使用collections.ChainMap加速字典合并**

低速方法

![24式加速你的Python32](https://res-static.hc-cdn.cn/fms/img/d898811d554dfa310cdda3ac28d99fd41603440421216)

![24式加速你的Python33](https://res-static.hc-cdn.cn/fms/img/3f5fa8280fcb19330ecd8f99e8d9cd5a1603440421216)

高速方法

![24式加速你的Python34](https://res-static.hc-cdn.cn/fms/img/c34353abeae46ce0aa50187390b605621603440421216)

###  

### **六、使用高阶函数进行加速**

**第14式，使用map代替推导式进行加速**

低速方法

![24式加速你的Python35](https://res-static.hc-cdn.cn/fms/img/3198ee203643ded6226daa81d42510f81603440421217)

高速方法

![24式加速你的Python36](https://res-static.hc-cdn.cn/fms/img/7b9fa048e7bbd7dac2729b5f908a78b61603440421217)

**第15式，使用filter代替推导式进行加速**

低速方法

![24式加速你的Python37](https://res-static.hc-cdn.cn/fms/img/45c394d911dce0ca0c6ec78a47b6393c1603440421217)

高速方法

![24式加速你的Python38](https://res-static.hc-cdn.cn/fms/img/c72a79b3c5acfacab46e21e9589d5de01603440421218)

### **七、使用NumPy向量化进行加速**

**第16式，使用np.array代替list**

低速方法

![24式加速你的Python39](https://res-static.hc-cdn.cn/fms/img/f59ce3c1bed03abd7345ec13aa5906ca1603440421222)

高速方法

![24式加速你的Python40](https://res-static.hc-cdn.cn/fms/img/686be017824ef6aef71e70f7af5825491603440421222)

**第17式，使用np.ufunc代替math.func**

低速方法

![24式加速你的Python41](https://res-static.hc-cdn.cn/fms/img/8fb1ff9ccad16fba5dfcb17a52e9355e1603440421222)

高速方法

![24式加速你的Python42](https://res-static.hc-cdn.cn/fms/img/eb33d86abffd1f00b0dd28a49da2cfe11603440421223)

**第18式，使用np.where代替if**

低速方法

![24式加速你的Python43](https://res-static.hc-cdn.cn/fms/img/4c51be510be317c359c78251d723e1f31603440421223)

![24式加速你的Python44](https://res-static.hc-cdn.cn/fms/img/0f10fa5e2aa310710c8a1205967771451603440421223)

高速方法

![24式加速你的Python45](https://res-static.hc-cdn.cn/fms/img/ac7e805514f2f256eecdbb925229be0e1603440421224)

###  

### **八、加速你的Pandas**

**第19式，使用csv文件读写代替excel文件读写**

低速方法

![24式加速你的Python46](https://res-static.hc-cdn.cn/fms/img/ae9dc2bda98db5aa83b1c0079e8ec5e11603440421224)

高速方法

![24式加速你的Python47](https://res-static.hc-cdn.cn/fms/img/a2378fcccd28f008d016f25f92ef7b421603440421224)

**第20式，使用pandas多进程工具pandarallel**

低速方法

![24式加速你的Python48](https://res-static.hc-cdn.cn/fms/img/4dabf919a2657422b06698defeac71e11603440421225)

![24式加速你的Python49](https://res-static.hc-cdn.cn/fms/img/c115119170e4b8a4faf4e9abf67635b51603440421225)

高速方法

![24式加速你的Python50](https://res-static.hc-cdn.cn/fms/img/94b846c630e3651a72aeb84133d87bd01603440421225)

###  

### **九、使用Dask进行加速**

**第21式，使用dask加速dataframe**

低速方法

![24式加速你的Python51](https://res-static.hc-cdn.cn/fms/img/d5d8fc2469c8e720e55346e57b4f5c971603440421226)

高速方法

![24式加速你的Python52](https://res-static.hc-cdn.cn/fms/img/e7124513e52b63ce29c33acb299d28821603440421226)

**第22式，使用dask.delayed进行加速**

低速方法

![24式加速你的Python53](https://res-static.hc-cdn.cn/fms/img/a9c4e34ca03242800efd7b2fba0343af1603440421226)

![24式加速你的Python54](https://res-static.hc-cdn.cn/fms/img/35d1b0e5c4de3882dce17a7285fbda351603440421227)

高速方法

![24式加速你的Python55](https://res-static.hc-cdn.cn/fms/img/ca0217cba11d0a9c2e90f04c8403c6a21603440421227)

### **十、应用多线程多进程加速**

**第23式，应用多线程加速IO密集型任务**

低速方法

![24式加速你的Python56](https://res-static.hc-cdn.cn/fms/img/9009c06f0e01a424f734163815f32f621603440421227)

高速方法

![24式加速你的Python57](https://res-static.hc-cdn.cn/fms/img/1b4b705637954cbb17bc6e5501f336981603440421228)

**第24式，应用多进程加速CPU密集型任务**

低速方法

![24式加速你的Python58](https://res-static.hc-cdn.cn/fms/img/2ddd4852475e2f6d53cd17c1bccb5ea91603440421228)

高速方法

![24式加速你的Python59](https://res-static.hc-cdn.cn/fms/img/182a1661eb5222f224228d2b72e984b81603440421228)