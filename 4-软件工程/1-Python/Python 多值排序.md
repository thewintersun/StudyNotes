## Python多值排序

按照多个值排序（统一升降序）：

```python
    countries = []
    for counter in range(country_num):
        line = input().strip()
        contents = line.split(' ')
        if len(contents)!=4:
            print('Error input skip:%s'%line)
            continue
        country_info = (contents[0],int(contents[1]),int(contents[2]), int(contents[3]))
        countries.append(country_info)
        
    out = sorted(countries, key=lambda x: (x[1],x[2],x[3],x[0]), reverse=True)

```

按照多个值排序（升降序不统一）：

```python
import functools

def mycmp(a,b):
    if a[1]>b[1]: return 1 
    if a[1]<b[1]: return -1 
    if a[2]>b[2]: return 1 
    if a[2]<b[2]: return -1 
    if a[3]>b[3]: return 1 
    if a[3]<b[3]: return -1 
    if a[0]>b[0]: return -1 
    if a[0]<b[0]: return 1 
    return 0

def func():
    country_num = int(input().strip())
    # please finish the function body here.
    countries = []
    for counter in range(country_num):
        line = input().strip()
        contents = line.split(' ')
        if len(contents)!=4:
            print('Error input skip:%s'%line)
            continue
        country_info = (contents[0],int(contents[1]),int(contents[2]), int(contents[3]))
        countries.append(country_info)
        
    #采用自己实现的cmp_to_key的比较方法
    countries.sort(key=functools.cmp_to_key(mycmp), reverse=True)
    
    # please define the python3 output here. For example: print().
    for item in countries:
        print(item[0])
```

