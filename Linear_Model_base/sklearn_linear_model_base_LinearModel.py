# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:56:20 2018

@author: USER
"""

import six
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy import sparse

from sklearn_linear_model_base__preprocess_data import _preprocess_data
from sklearn_utils_validation_check_array import check_array
from sklearn_utils_validation_check_is_fitted import check_is_fitted
from sklearn_linear_model_base_BaseEstimator import BaseEstimator


def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b  # 如果是 sparse matrix 做矩阵乘法
        if dense_output and hasattr(ret, "toarray"):  # 如果需要 dense_output 顾名思义就是压缩输出
                                                      # 并且 ret 对象可以转换成 array， 那就换成 array 吧
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b) # 如果不是 sparse matrix 直接做 np.dot

#------------------------------------------------------------------------------------
# metaclass 元类  参考 https://www.cnblogs.com/ajianbeyourself/p/4052084.html
        
# 为了理解这个概念，我花了整整两天的时间，这里给出我的理解
        
# 1. type 类
        
# python 的一切都是对象。而一切对象又都是类实例化的结果。那么问题来了，类又是什么的实例呢？
# 实际上，类也是某种类实例化的结果！这种又是实例化产生类的类，被成为元类 metaclass
# 但是问题又来了，metaclass 作为一种类，肯定又是另外一种元类的实例化类，这样一来该怎么办呢？
# 实际上，Python 的元类追溯到 type 类为止。type 类是一切类最终的元类

# example: class test(object):
#              pass
#                      随便构造的一个类。检测它的元类 type(test) -> type 或者 isinstance(test, type) -> True

# 如何理解追溯的意思呢？
# example: x = int(1) y = 'test' z = dict()
#          这样分别实例化了 int，test，和 dict 对象
#          x.__class__ -> int
#          y.__class__ -> str
#          z.__class__ -> dict  与预想中的一致
        
# 但是如果继续看 int str dict 作为一个实例的元类呢?
#          x.__class__.__class__
#          y.__class__.__class__
#          z.__class__.__class__
# 其结果都是 type
# 说明这些类的元类都是 type

# 既然元类是实例类的类，那么怎么使用这种方法类来实例化一个类呢？
# 标准方法: type('class_name', parent_class, attr_dict)
# 'class_name' 顾名思义，就是要被实例化的类的名称,比如 (object,)
# parent_class 被实例化的类的 parent_class,必须是 tuple
# attr_dict 被实例化的类的属性和它的值。这是一个 dict {'attr_name': method}
#           它的结构：{属性名：属性（这可以是各种对象，int str dict function class...都行哦）}

# example: test = type('test', (object,), {'test_attr':1})
#          test.test_attr -> 1
        
# 2. 通过函数来拓展 type 的功能

#    使用函数来进一步拓展 type 实例化类的方法
# example:
#           def test(class_name, parent_class, attr_dict):
#               attr_dict['added_attr'] = 'test'
#               return type(class_name)    
# x = test('x', object, {})
# x.added_attr -> 'test'     这样以来，就构建一个可以用来实例化元类，并为类添加特定属性的函数
        
# 3. 通过类来拓展 type 的功能
        
# 既然可以用函数，那必然也可以用 class 来实现这项功能
# 但是，必须要使用 __new__ magic method!
# 为什么呢？
# __new__ 和 __init__ 都可以创建并初始化对象 object，但是 init 只会将 object 作为参数进行
# 传递。但是 new 却是直接返回对象本身
#
# __new__ is the method called before __init__
# it's the method that creates the object and returns it
# while __init__ just initializes the object passed as parameter
# you rarely use __new__, except when you want to control how the object
# is created.
# here the created object is the class, and we want to customize it
# so we override __new__
# you can do some stuff in __init__ too if you wish
# some advanced use involves overriding __call__ as well, but we won't
# see this

#class test1(type):
#    
#    def __new__(cls, class_name, parent_class, class_attr): # 注意第一个参数是类本身
                                                             # 由于总会传递这个参数，(就跟 self 至于 init method 一样)
                                                             # 所以调用的时候不要传递这个参数
#        class_attr['add_attr0'] = ‘test’
#        return type(class_name, (parent_class,), class_attr)

# 当然，可以进一步使用 super 方法来调用 type

#class test2(type):
#    
#    def __new__(cls, class_name, parent_class, class_attr):
#        class_attr['add_attr0'] = add_attr0
#        return super(test2, cls).__new__(cls, class_name, parent_class, class_attr)

# 发现，这里调用的而是 type 的__new__ 方法
# 没错，type 创建类也是用的它自己的 new 方法

# 使用方法 x = test2('test', (object,), {})
# 记得，得到的 x 是一个类

# 4. 为什么要使用元类？
#
#        拦截类的生成
#        修改类
#        返回修改后的类

# 太绕了，别轻易用 元类
# 另外，并不是只有 type 才可以作为元类，也可以是 type 的子类。当然所有的元类追溯到最后都是
# 使用 type 来创建类
                                                             
class LinearModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    
    #---------------------------------------------------------------------------------
    # 什么是 with_metaclass？
    # 先看看它的源码
    
#    def with_metaclass(meta, *bases):
#    class metaclass(type):
#        def __new__(cls, name, this_bases, d):
#            return meta(name, bases, d)
#    return type.__new__(metaclass, 'temp_class', (), {})
    
    # 结合上面对元类的分析，可以看出，metaclass 传递的两个参数。第一个是
    # 一个元类，用来生成一个类。第二个是 parent_class 
    # 可见如果 *bases 为空，那么相当与是 object
    # 总的来说
    # 就是: 以 *bases 为 parent_class， 以 meta 为元类，生成一个为类的实例
    
    # 但是得到的。。似乎只能在类的继承中作为 parent_class 使用
    # 比如上面提到的 test2
    
    # x = six.with_metaclass(test2, object)
    # isinstance(x, test2) -> False
    # hasattr(x, 'add_attr0') -> False
    
    # class test_meta(six.with_metaclass(test2)):
    #     pass
    # isinstance(test_meta, test2) -> True
    # test_meta.add_attr0 -> 'test'
    
    # 这里，使用 ABCMeta 的元类，说明 LinearModel 继承了 BaseEstimator 的 method 但是
    # 需要有 抽象方法
    # 显然，这个 class 也是要作为 parent_class 被继承的
    # 代码中没有发现 super 方法，说明这里不需要使用 BaseEstimator 中的方法
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y): # 抽象方法，说明所有的 LinearModel 必须要有一个 fit 方法
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self, "coef_") # 这个函数用来检查 self 中是否有 fit 属性
                                       # 或者指定的 attribute 这里指定的是 'coef_'

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo']) # 检查 array 的数据格式和 sparse matrix 格式
        return safe_sparse_dot(X, self.coef_.T,   # 
                               dense_output=True) + self.intercept_ # 返回和 X 和 coef_ 的 dot

    def predict(self, X):
        
        """Predict using the linear model

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        
        return self._decision_function(X)

    _preprocess_data = staticmethod(_preprocess_data) # 将 _preprocess_data 作为 LinearModel 类的一个静态方法，可以作为函数使用

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.