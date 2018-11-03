# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:30:42 2018

@author: USER
"""

"""
scikit-learn/sklearn/linear_model/base.py
Utilities for linear regression
"""
# base 最基础的线性回归模型
# 函数 _pprint
# 自定义的可以比较漂亮的打印 python dictionary

import numpy as np
from .externals import six

def _pprint(params, offset=0, printer=repr):
    
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    
    # Do a multi-line justified repr:
    
    #-----------------------------------------------------------------------
    # set_printoptions, get_printoptions
    # numpy array 打印定制
    #
    # 参数：
    # 1. precision 浮点数打印精度。默认精度为 8，对应的默认值是 None
    
    # 2. threshold 长度阈值，超过这个阈值就缩写。注意这个长度是缩写前的 elements
    #              个数和缩写后的 elements 长度。举例，如果 threshold = 2，那
    #              么 np.arange(10) 的 print 结果是 0 1 ... 8 9
    
    # 3. edgeitems 作用跟 threshold 类似。只是是作用在多维数组的，针对每一个维度，
    #              打印多少个elements，剩下的省略
    
    # 4. linewidth 每行能够打印的字符个数，包括 [ . 空格等
    
    # 5. suppres   选项有 Ture False
    #              min 1e-4 max 1e3 如果 True  如实打印所有的浮点数
    #                               如果 False 超过这个范围就使用科学计数法
    
    # 6. nanstr infstr 非有限性数字，改怎么打印呢？默认 nanstr = nan infstr = inf
    #                  即对于 array 中的 np.nan np.inf 就打印出 nan 和 inf
    #                  当然也可以自己设置别的打印字符串。比如 nanstr = 'test'
    #                  nanstr 的含义更广，非数字的项，比如 None 等，都可以用 nanstr
    #                  设置的字符串来进行打印。但是 infstr 只针对 inf
    
    # 7. sign      数字的符号，'+' 正数也要打印出 +
    #                          ' ' 正数不打 + 但是要加个空格（注意不是'',而是' '）
    #                          '-' 忽略正数的符号-default
    
    # 8. formatter 为每个类型制定一个格式化函数。注意是制定一个函数，一般 lambda 函数
    #              比较合适
    #              类型包括：
#                            ‘bool’
#                            ‘int’
#                            ‘timedelta’        : a numpy.timedelta64
#                            ‘datetime’         : a numpy.datetime64
#                            ‘float’
#                            ‘longfloat’        : 128-bit floats
#                            ‘complexfloat’
#                            ‘longcomplexfloat’ : composed of two 128-bit floats
#                            ‘numpystr’         : types numpy.string_ and numpy.unicode_
#                            ‘object’           : np.object_ arrays
#                            ‘str’              : all other strings
#                            
#                              Other keys that can be used to set a group of types at once are:
#                            
#                            ‘all’              : sets all types
#                            ‘int_kind’         : sets ‘int’
#                            ‘float_kind’       : sets ‘float’ and ‘longfloat’
#                            ‘complex_kind’     : sets ‘complexfloat’ and ‘longcomplexfloat’
#                            ‘str_kind’         : sets ‘str’ and ‘numpystr’
#
#                  example： np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#                            np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})
#                            np.set_printoptions(formatter={'all':lambda x: 'test: '+str(-x)})
# 剩下的不常用就不注释了
    
# 另外，如果想返回默认值呢？ undo set_printoptions
# 目前没有好的方法，只能重新使用 set_printoptions 设置成默认值
# np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8,
#                     suppress=False, threshold=1000, sign = ' ', formatter=None)
    
    # options, 得到所有的 set_printoptions 设置项
    options = np.get_printoptions()
    
    # 设置 浮点数精度 5， 打印阈值为 64， 行数打印阈值为 2
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list() # list() == [] 初始化一个空 list
    this_line_length = offset # 字符偏移量
    
    # 构建一个换行符
    # 字符串接受整数 int 的乘法  1//2 = 0 截断除法，line_sep 定制了一个换行符
    # 下一行要空 (1 + offset // 2) * ' ' 大小的空格
    line_sep = ',\n' + (1 + offset // 2) * ' '
    
    #-------------------------------------------------------
    # python 中 dict 迭代器的构造
    
    # enumerate 是 python 中构造迭代器的一个函数
    # 方法，对于可迭代对象，对元素按照顺序以 (0,element0),(1,element1) 元组的形式
    # 构建迭代元素，start = 1 可以以 1 为起始，当然也可以是别的 int。
    
    # dict {'name':value ....}有自己的迭代器构造函数 items 和 iteritems
    # items 跟 enmuerate 一样，构造一个 list，里面是一系列的 tuple ('name': value)
    # iteritems 则是构造了一个迭代器 dict_itemiterator 而不是list，
    # 当然打印出来是一样的
    
    # 注意 dict 本身是无序的。所以使用 sorted 本身排序我没发现有什么意义
    
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        
        if type(v) is float:
            
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
            
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v)) # repr 返回一个对象的字符串形式
        
        # 自定义了 print threshold 前后阈值分别为 300 和 100
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
            
        if i > 0:
            
            # 如果长度过长（长度包括前缀+字符长度），或者有换行符，添加一个
            # 换行符 line_sep
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
                
            else:
                # 如果不够长，添加一个，和空格，并将长度计入 this_line_length
                params_list.append(', ')
                this_line_length += 2
                
        # 添加要打印的字符串，并且计算长度        
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list) # 讲所有要打印的字符串都链接成一个字符串，无空格链接
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines