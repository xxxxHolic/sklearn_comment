# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:23:17 2018

@author: USER
"""

"""
scikit-learn/sklearn/linear_model/base.py
Utilities for linear regression
"""
# base 最基础的线性回归模型
# 函数 _rescale_data
# Rescale data so as to support sample_weight

import numpy as np
from scipy import sparse
from sklearn_utils_extmath_safe_sparse_dot import safe_sparse_dot

def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = np.full(n_samples, sample_weight,
                            dtype=np.array(sample_weight).dtype)
    
    # np. full Return a new array of given shape and type, filled with fill_value.
    # very usefull for weight
    # for example: np.full(3, [1,2,3], dtype = np.int8) 得到 array([1,2,3], dtype = int8) 
    
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                  shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y) # Dot product that handle the sparse matrix case correctly
    return X, y