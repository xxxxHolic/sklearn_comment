# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:26:15 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/sparsefuncs.py
Utilities for sparse matrix analysis
"""
# sparsefuncs 用于稀疏矩阵的预处理
# 函数 inplace_column_scale
# 用于 dataset 列方向的数据归一化

import numpy as np
import scipy.sparse as sp

#-----------------------------------------------------------
# 这里，处理 CSR 和 CSC 格式的稀疏矩阵，不处理别的矩阵形式
# scipy sparse 中的 csr 和 csc 矩阵性质

def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    input_type = X.format if sp.issparse(X) else type(X)
    err = "Expected a CSR or CSC sparse matrix, got %s." % input_type
    raise TypeError(err)

#----------------------------------------------------------
# 
def inplace_csr_row_scale(X, scale):
    
    """ Inplace row scaling of a CSR matrix.

    Scale each sample of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Matrix to be scaled.

    scale : float array with shape (n_samples,)
        Array of precomputed sample-wise values to use for scaling.
    """
    
    assert scale.shape[0] == X.shape[0]
    X.data *= np.repeat(scale, np.diff(X.indptr))
    

def inplace_csr_column_scale(X, scale):
    """Inplace column scaling of a CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    """
    assert scale.shape[0] == X.shape[1]
    X.data *= scale.take(X.indices, mode='clip')


def inplace_column_scale(X, scale):

    """Inplace column scaling of a CSC/CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSC or CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    """
	
    if isinstance(X, sp.csc_matrix):
        inplace_csr_row_scale(X.T, scale)
    elif isinstance(X, sp.csr_matrix):
        inplace_csr_column_scale(X, scale)
    else:
        _raise_typeerror(X)