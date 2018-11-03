# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:44:33 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 _ensure_sparse_format
# 将输入的 sparse matrix 按照指定格式进行格式转换。
# 依赖的函数：_check_large_sparse，_assert_all_finite

from externals import six
import warnings
#-------------------------------------------------------
# warings.warn(str,warning_type)

# str: warning message 警告信息
# warning_type: 警告类
# 警告类包括：
# DEprecationWarning 使用已经弃用的功能
# SyntaxWarning 可能出问题的语法
# RuntimeWarning 运行时间可能出问题
# ResourceWarning 资源使用的警告
# FutureWarning 对未来可能修改的功能进行警告

from sklearn_utils_validation__check_large_sparse import _check_large_sparse
from sklearn_utils_validation__assert_all_finite import _assert_all_finite

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite, accept_large_sparse):
    
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.
    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.
    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.
    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:
        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    
    # 获得输入的 sparse matrix 的 data type
    # 如何控制输入的默认值：var = False/True or others
    #                       var，在具体语段中用 None 来判断输入与否，然后设置默认值
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    # 对不同格式的 accept_sparse 进行处理
    # 如果是 str （six 库解决版本兼容问题，sklearn 自己copy了一个six库使用）
    # 就转换成 list [str]

    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    # 判断输入 sparse matrix 的格式和大小是否满足判据
    _check_large_sparse(spmatrix, accept_large_sparse)

    # 如果 accept_sparse 的格式是 Boolean
    # 如果 bool 是 False
    # 报错不接受 sparse matrix，提示转换成 dense data
    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    
    # 如果是 list, tuple
    # 如果是 list tuple 是空，那么报错
    # 如果 list tuple 不是空 (注意 list 不是空的情况包括 如果 accept_sparse 是 str
    # 那么 上一个 if 判断已经将str 转换成了 list [str])
    # 那么判断输入的 sparse_matrix 的格式是不是属于可接受的 sparse matrix 格式，如果
    # 不属于，那么转换格式,取 list 或者 tuple 中接受的第一个格式
    
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
           
            # create new with correct sparse
            # sparse_matrix.asformat(str) str = 'csr','coo','bsr'等
            # 返回给定格式的稀疏矩阵
            
            spmatrix = spmatrix.asformat(accept_sparse[0])
            
            # 格式转换状态改为 True
            changed_format = True
    
    # accept_sparse 只能是list，tuple，str，boolen. 由于 str 已经被上一个条件判断
    # 转换为了 list，list tupe 和 False 也已经被判断。因此如果再不是为 True 的Boolen
    # 则报错没有输入正确的格式
    
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))
    
    # 如果制定了sparse matrix 的格式，那么就按照这个格式转换
    # 注意这个格式是数据格式 int32 float32等，不是稀疏矩阵的压缩格式
    # 使用方法 astype
    
    if dtype != spmatrix.dtype:
        
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    
    # changed_format = True 表明 sparse matrix 已经经过了格式转换，生成了一份 copy
    # 如果要求 copy sparse matrix copy = True，那么生成一份 copy
    
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    # 关于sparse matrix 中数据有限性的判断
    # 变量 force_all_finite
    # False:接受所有的极限数据，不需要操作
    # True：只接受有限性数据，需要将极限数据转换为有限数据
    # allow_nan 只接受 np.nan 这一个极限数据
    # 这些通过 _assert_all_finite 函数完成
    
    if force_all_finite:
        
        # 如果无法解析 sparse matrix 的数据则报 warning
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data,
                               allow_nan=force_all_finite == 'allow-nan')

    return spmatrix