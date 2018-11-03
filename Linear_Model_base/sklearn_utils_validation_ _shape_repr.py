# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:12:03 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 _shape_repr
# 用于 report data shape 时，打印信息时不会因为平台和版本的问题打印
# 出错误的信息，‘long’type 会有后缀 L，自己定义打印信息
# 依赖的函数和库：

def _shape_repr(shape):
    
    """
    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    
    Return a platform independent representation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.
    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.
    """

    
    if len(shape) == 0:
        return "()"
    
    joined = ", ".join("%d" % e for e in shape)
    
    #------------------------------------------------------
    # x for x in shape 语法:
    # for x in shape:
    #     return x
    # example [x for x in [1,2,3]] = [1,2,3]
    # 因此 '%d' % e for e in shape 等于把 shape 中的数字以 str 形式
    # 打印出来
    # example ['%d' %x for x in [1,2]] = ['1','2']
    
    #------------------------------------------------------
    # join 语法
    # str.join(sequence)
    # sequence:要连接的元素序列
    # str 连接字符
    # ','.join('%d' % e for e in shape) 即是将 shape 中的元素用‘，’连接起来
    # if shape  = [1,2] -> '(1,2)'
    
    if len(shape) == 1:
        
        # special notation for singleton tuples
        joined += ','
        
        # 如果 len(shape) = 1, 比如 shape = (1,)
        # joined = ", ".join("%d" % e for e in shape)
        # 的结果是 '1'
        # '(%s)' % joined = '(%s)' % '1,' = '(1,)'
        
    return "(%s)" % joined
