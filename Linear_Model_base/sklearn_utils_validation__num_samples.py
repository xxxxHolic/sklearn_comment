# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:34:00 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 _num_samples
# 用于 array-like x 中样品个数的统计
# 依赖的函数和库：

import numpy as np

def _num_samples(x):
    
    """Return number of samples in array-like x."""
    
    #--------------------------------------------------------------------------------
    # hasattr(object, name):如果 object 有名为 name 的属性，返回True，否则返回False
    # 注意！在py2中，hasattr 会隐藏掉 property：
    #  class test(object):
    #      @property
    #      def y(self):
    #     .....
    # hasattr(test(), 'y')  -> False
    # 因此， try: print(test.y) except: print('no y!') 可以代替
    # 在 py3 中没有这个问题   
    # 参考：http://codingpy.com/article/hasattr-a-dangerous-misnomer/
    
    #-------------------------------------------------------------------------------
    # callable(object) 检查一个对象是否可以调用
    # 返回 True：函数，方法，lambda函数式，类，以及实现了__call__方法的类实例
    
    if hasattr(x, 'fit') and callable(x.fit):
        
        # input x 存在 fit 这个属性而且可以调用
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
        
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        
        #-----------------------------------------------------------------
        # __len__ 方法
        # [1,2,3].__len__() = len([1,2,3]) -> 3
        # class test:
        #     def __len__():
        #         return 'test!'
        # test.__len__() -> test!
        # len(test) -> TypeError:object of type 'type has no len()'
        
        if hasattr(x, '__array__'):
            
            #------------------------------
            # __array__ 方法，对应 numpy object
            
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)
    