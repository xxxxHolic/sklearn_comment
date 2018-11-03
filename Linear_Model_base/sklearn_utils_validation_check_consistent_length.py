# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:46:43 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for linear regression
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 check_consistent_length
# 检查所有输入的 array 是否剧中相同的行-即相同的 sample 个数


import numpy as np
from sklearn_utils_validation__num_samples import _num_samples

def check_consistent_length(*arrays):
    
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None] # 统计所有非 None array 的样品个数(array 的行数)
    uniques = np.unique(lengths)
    if len(uniques) > 1: # 如果有不同长度的 array，报错
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])