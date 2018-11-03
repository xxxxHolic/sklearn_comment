# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:37:05 2018

@author: USER
"""

"""
scikit-learn/sklearn/linear_model/base.py
Utilities for linear regression
"""
# base 最基础的线性回归模型
# 函数 _preprocess_data
# 数据的预处理


# numbers:定义python中数据阶梯规则的抽象类
import numbers

import numpy as np
import scipy.sparse as sp # scipy 中用于处理稀疏矩阵的库

from ..utils.validation import FLOAT_DTYPES
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..preprocessing.data import normalize as f_normalize

from sklearn_utils_validation_check_array import check_array

def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True,
                     sample_weight=None, return_mean=False, check_input=True):
    
    """
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """

    if isinstance(sample_weight, numbers.Number):
        sample_weight = None # 如果输入的权重只是一个数字，那么相当于无效权重 - None
   
    # 对输入进行检查
    # 如果 check_input, 使用 check_array 进行格式，稀疏矩阵的检查以及格式转换
    if check_input:
        X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'], dtype=FLOAT_DTYPES)
    
    # 是否对数据进行一份拷贝，这样不改变原始输入数据
    # 是否为稀疏矩阵的拷贝数据格式是不一样的
    
    elif copy:
        if sp.issparse(X):
            X = X.copy() 
        else:
            X = X.copy(order='K') 
            
# numpy order
#‘K’	unchanged	F & C order preserved, otherwise most similar order
#‘A’	unchanged	F order if input is F and not C, otherwise C order
#‘C’	C order	C order
#‘F’	F order	F order

    # 将输入的 y 转换成 numpy array
    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            
            # mean_variance_axis 用于求稀疏矩阵的平均值，方差等参数
            # 我看了函数代码，稀疏矩阵类型包括 csr csc ，别的类型会
            # 报错。原理很简单，就是求和然后除以个数，没有别的算法
            # 返回平均值以及方差
            X_offset, X_var = mean_variance_axis(X, axis=0) # mean_variance_axis return means and variances
            
            if not return_mean:
                
                # X.dtype.type(test) 使用 X 的格式来转换数据 test
                # 这里 not return_mean ，所以直接返回 0 只是用的 X 的 data type
                X_offset[:] = X.dtype.type(0)

            if normalize:

                # TODO: f_normalize could be used here as well but the function
                # inplace_csr_row_normalize_l2 must be changed such that it
                # can return also the norms computed internally
                # transform variance to norm in-place
                
                X_var *= X.shape[0]
                
                # numpy sqrt 有一个用法 numpy.sqrt(x, out = test)
                # test 是长度与x一致的 ndarray，用于存储 sqrt(x)
                # 的结果。也可以直接使用 x 本身，即直接修改 x 本身。
                
                X_scale = np.sqrt(X_var, X_var)
                
                del X_var  # X_var 已经被修改了，直接删除防止歧义使用
                
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False, return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale