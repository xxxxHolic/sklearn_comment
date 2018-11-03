# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:33:10 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""

import numpy as np

# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 _ensure_no_complex_data: 不接受 array 中有 complex data
# 函数 _assert_all_finite: 
# 依赖的函数库：numpy， get_config

import os

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024))
}

#----------------------------------------------------
# sklearn.set_config 用于控制如下两个行为
# assume_finite: 跳过/接受 检查 validation。这样可以有更快的运算速度，但是如果数据中含有 nan，
# 那么跳过检查可能会造成一些数据分割时的错误
# working_memory: 优化某些算法调用的array大小

#----------------------------------------------------
# sklearn 中的环境变量
# SKLEARN_SITE_JOBLIB:   设置 joblib 的使用版本
# SKLEARN_ASSUME_FINITE: 设置 sklearn.set_config 中 assume_finite 的默认值，
# SKLEARN_WORKING_MEMORY:设置 numpy 向量化操作时会耗费大量的内存，设置默认的 limiting working memory
#                        如 _global_config 设置是 1024M = 1G

#---------------------------------------------------
# os.environ.get(enviroment_variables, value)
# 获取系统变量的返回值，如果为空 (print 结果为 None)，返回 value

def _get_config():
    """Retrieve current values for configuration set by :func:`set_config`
    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    """
    return _global_config.copy()

#---------------------------------------------------
# python 的数据赋值方式

# python object（对象）有三个基本要素：id（身份标志），python type() （数据类型）， value （值）

#---------------------------------------------------
#  = 赋值：
# 新建一个 python 对象 list-[1,2,3]，并赋值给 test0 = [1,2,3] id = 1747592899080
# 将 test0 赋值给 test1，test1 = test0， id(test1) = 1747592899080
# ! 可以发现，通过 = 赋值，是将 id，dtype，value 一同进行赋值，只是变量名不同
# 因此，可以发现，如果改变 test0 (test1) 的value，就会同时改变 test1 (test0) 的 value
# example：test0[0] = 3 -> test1 = [3,2,3]
# 但是，同时可以发现，如果对 test0 重新赋值，test0 = [0,0,0] -> test1 = [3,2,3], 并没有改变
# 原因：看 test0 的id，id(test0) = 1747593088456, id 变了？
# 可见， test0 = [0,0,0], 实际上是使得 test0 成为新的对象的命名，id发生了改变 （[0,0,0]实际上是新的对象，可以 id([0,0,0]试试看)）
# 而 test0[0] = 3 只是改变了对象的 value (或者 type， 虽然这里并没有)，因此，并不是新的对象
# 而两个不同的对象，test0，test1 自然也就不会有什么关联
    

#--------------------------------------------------
# copy 赋值:
# copy 赋值的方式有：test.copy() 以及在 import copy 之后， copy.copy(test)
# test1 = test0.copy()
# id(test0) = 1747593610120 id(test1) = 1747593642824
# 可见，copy赋值后，test1 和 test0 是两个对象

# 但是并不仅仅是如此
# 对于copy 对象的子对象，并没有 copy 出一个新的对象，而是引用
# test0 = [0,1,[2,3]], test0 是一个 list， 它里面还嵌套了一个子对象 [2,3]
# test1 = test0.copy()
# 现在改变 test0 的 value， test0[0] = 'test'
# test0 = ['test',1,[2,3]] test1 = [0,1,[2,3]]
# 现在改变 test0 嵌套的子对象的value test0[2][0] = 'test'
# test0 = ['test',1,['test',3]] test1 = [0,1,['test',3]]
    
#--------------------------------------------------

# deepcopy 赋值：
# deepcopy 赋值方式：import copy, copy.deepcopy(test)
# 相比于 copy 赋值， deepcopy 的意义在于，将深层嵌套的子对象也一起copy一个新的对象
# example：
# test0 = [0,1, [2,3]]
# test1 = copy.deepcopy(test0)
# test0[2][0] = 'test'
# test0 = [0, 1, ['test', 3]]
# test1 = [0,1, [2,3]]
    
#--------------------------------------------------

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))

#-------------------------------------------------
# dtype.kind numpy 中用于识别数据类型的字符

#b	boolean
#i	signed integer
#u	unsigned integer
#f	floating-point
#c	complex floating-point
#m	timedelta
#M	datetime
#O	object
#S	(byte-)string
#U	Unicode
#V	void

# _ensure_no_complex_data 拒接接受 complex float-point 类型的数据
        
def _assert_all_finite(X, allow_nan=False):
    # 检查输入的有限性
    
    """Like assert_all_finite, but only for ndarray."""
    
    # assume_finite 对应的环境变量是 SKLEARN_ASSUME_FINITE, 默认返回值是 False
    # 如果修改了返回值是 True，那么就不需要在进行输入检查了，直接 return empty
    
    if _get_config()['assume_finite']:
        return
    
    X = np.asanyarray(X)
    
#-----------------------------------
# numpy 中的 array 赋值
# np.array(test)
# 分为两种情况，1> test 是 ndarry matrix, np.array 会返回一个 test 的 copy
#                  这个 copy 可以类比为 python 的 copy，即value 和
#                  dtype 不变，但是id 会改变
#               2> test 不是 ndarry,包括 list ，tuple 等，生成
#                  一个新的 ndarry object
# np.asarray(test)
# 分为两种情况，1> test 是 ndarry matrix，这时，np.asarray 会对test进行重新
#                  命名。如果是 ndarray id，dtype 和 value 不变
#                  如果是 matrix，则返回的是ndarray，生成一个新的对象
#               2> test 不是 ndarray matrix,生成一个新的 ndarry object
# np.asanyarray(test)
# 返回的是 ndarry 的子类
# 以 matrix 为例。matrix 是 ndarray 的一个子类。使用 asarray 方法 copy 返回的是
# ndarray 而不是 matrix，是一个新的对象。而使用 asanyarray 方法，则会返回 ndaray
# 的子类 matrix
    
    
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    
    is_float = X.dtype.kind in 'fc' # 'f' floating-point # 'c' complex floating-point
    if is_float and np.isfinite(X.sum()):
        
        # np.isfinite 检查数字的有限性。如果不是数字(array,list,tuple 等可迭代的变量，元素不是数字)
        # 返回 Value error
        # 返回 True：数字是有限的
        # 返回 False：np.nan, np.inf, np.NINF, 以及数学运算得到的无限，如：log(-1), tan90, log(0), 1/0 (这个python的基本运算就会报错)
        # np.isfinite(x,y)
        # isfinite 允许输入两个 size 一致的 array， 这样，第一个数组的有限性判断结果会以第二个数组元素的dtype 输出
        # 如果没有第二个数组，默认输出是bool，布尔值
        
        # 即如果数据类型不接受 （floating-point and complex floating-point）并且含有极大或者极小值，则直接报错
        
        pass
    elif is_float:
        
        # allow_nan 默认值是 False，即不接受非有限性数字，但是如果输入允许 True，则接受
        # 并且只对 finite number 报错
        
        # np.isinf 的用法与 isfinite 一致，只是返回 True，False 的逻辑顺序是颠倒的
        
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(msg_err.format(type_err, X.dtype))