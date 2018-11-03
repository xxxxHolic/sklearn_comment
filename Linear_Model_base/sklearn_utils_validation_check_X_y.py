# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:52:48 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""

# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 check_X_y: 
# 保证 1.X 和 y 具有相同的长度 2.X 和 y 的维度 X-2d y-1d

import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning

from sklearn_utils_validation_check_array import check_array
from sklearn_utils_validation__assert_all_finite import _assert_all_finite
from sklearn_utils_validation__num_samples import _num_samples

# 将 y 降维至 1d
def column_or_1d(y, warn=False):
    
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    
    #--------------------------------------------------------------
    # np.ravel 与 np.flatten
    # 功能，将多维 array 以 row0，row1，row2.... 的顺序降维到 1d
    # 区别：与 = 和 copy 以及 array asarray 的区别类似
    # ravel 是指向原来 array 的一个 reference
    # flatten 则返回一个 copy
    
    #-------------------------------------------------------------
    # 但是这个函数只接受维度最大为 2d 的 array
    # 否则报错
    # 即使是 2d 的 array 也会报 warning
    
    shape = np.shape(y)
    
    if len(shape) == 1:
        return np.ravel(y)
    
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def check_consistent_length(*arrays):
    
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    # 这一段等价于
    # lengths = []
    # for X in arrays:
    #     if X is not None:
    #          lengths.append(_num_samples(X))
    
    lengths = [_num_samples(X) for X in arrays if X is not None]
    
    #-----------------------------------------
    # np.unique(test, return_index = True, return_inverse = True)
    # 一维数组或者列表
    # 去除重复元素，按照大小返回一个新的五元素重复元组或者列表
    # return_index = True    返回新列表元素在旧列表中的位置
    # return_inverse = True  返回旧列表元素在新列表中的位置
    
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])
        

def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2d and sparse y.  If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    # 必须要有 y
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
    
    #------------------------------------------------
    # 是否接受输入 2d 的 y 输入
    # check_array 条件
    # sparse_matrix 压缩方法是 ‘csr’
    #               强制转换所有极限数据为有限数据
    #               对非二维结构报错
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
        
    else:
        # 如果不接受，对 y 进行降维
        y = column_or_1d(y, warn=True)
        # 进行数据有限性检查
        _assert_all_finite(y)
    
    # 如果 y 的数据类型是 object，转换为 np.float64
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    # 检查X 和 y 的数据长度是否一致
    check_consistent_length(X, y)

    return X, y