# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:59:14 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""

# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 check_array: 
# 检查 array 
# 1. sparse matrix 是否接受
# 2. array 格式检查，是否统一转换成 numpy array
# 3. 数据有限性检查及格式转换
# 4. array 维度检查，是否对非二维数据进行报错或转化
# 5. 对样品个数进行检查

# 总之，检查输入数据进行检查，按照参数要求，能转化的转化，不能转化的报错
# 最终得到满足统一标准的 array 格式 

from externals import six
import warnings

import numpy as np
from numpy.core.numeric import ComplexWarning
import scipy.sparse as sp
from sklearn.exceptions import DataConversionWarning

from sklearn_utils_validation__assert_all_finite import _ensure_no_complex_data,_assert_all_finite
from sklearn_utils_validation__ensure_sparse_format import _ensure_sparse_format
from sklearn_utils_validation__shape_repr import _shape_repr
from sklearn_utils_validation__num_samples import _num_samples

def check_array(array, 
                accept_sparse=False, 
                accept_large_sparse=True,
                dtype="numeric", 
                order=None, 
                copy=False, 
                force_all_finite=True,
                ensure_2d=True, 
                allow_nd=False, 
                ensure_min_samples=1,
                ensure_min_features=1, 
                warn_on_dtype=False, 
                estimator=None):

    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, boolean or list/tuple of strings (default=False)
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
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.
        .. versionadded:: 0.20
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : boolean (default=False)
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
    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.
    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    
    #----------------------------------------------------------------
    # 对应函数 _ensure_sparse_format
    # accept_sparse 'None' deprecation check
    # 如果 accept sparse 没有输入参数，那么warning 然后设置默认值
    # 为 False，表明不接受 sparse matrix 而只接受 dense data type
    
    if accept_sparse is None:
        
        # DeprecationWarning 警报类型，功能弃用
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        # 设置 accept_sparse 的默认值为 False
        accept_sparse = False

    #---------------------------------------------------------------
    # 存储一个到 array 的 reference
    # 所以使用的是 = 赋值方式，参见 _assert_all_finite 关于赋值的注释
    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array


    #--------------------------------------------------------------------------------

    # 这一段进行输入的格式判断
    # 如果判断输入数据的格式满足需求，那么 dtype = None， 后续不进行数据格式转换
    # 如果不满足需求 dtype = np.float64, list[0], tuple[0] 等进行格式转换


    # dtype 这个变量对应的函数是 _ensure_sparse_format
    # dtype 这个参数是用于控制输入的 array 的格式
    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"
    
    # dtype_orig 先获得输入的 array 的原始数据格式，如果输入的数据没有对应的 dtype 属性
    # 那么返回 None
    
    dtype_orig = getattr(array, "dtype", None)
    
    # 一般 numpy 的 array 都会有个 dtype.kind
    # 例外是 pandas 的 DataFrame 数据结构，有 dtype，但不会得到 numpy 中 dtype
    # 对应的 kind，此时仍然返回 None
    # 因为后续的处理仍然以 numpy array 为基础，要进行格式转换
    # dtype_orig  = None
    
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    
    # 对于 numpy array 返回其 data type 数据格式
    if hasattr(array, "dtypes") and hasattr(array, "__array__"):
        dtypes_orig = np.array(array.dtypes)
    
    # 如果输入的是一个对象，这里 dtype_orig.kind = 'O' O 表示 object 类型
    # dtype = np.float64, 将格式转换为 float64
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None
     
    # 如果 input array 的格式是属于需求格式的， 那么后面就不需要再
    # 对数据进行格式转换
    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
            
        # 如果不是，那么就按照要求的数据格式 list tuple 的第一个格式进行转换
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    # ----------------------------------------------------------------------------------
    
    # 判断进行数据有限性检查的参数是否有满足标准
    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    # 得到 estimator_name
    # 如果 estimator 不是空
    # 如果 estimator 是字符串 estimator_name 就是 estimator
    # 如果 estimator 是 object，estimator_name = estimator.classname
    # 如果 estimator 是空
    # estimator_name = 'Estimator'
    
    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    
    #----------------------------------------------------
    # 语法结构 test0 if 条件1 else test1
    #          = if 条件1:
    #                test0
    #            else：
    #                test1
    # 即 if estimator is not None 返回 estimator_name，如果是空的，返回空字符串
    
    #----------------------------------------------------
    # is not None 语法结构解析
    # python 中的空结构 - None False 空字符串'' [] {} ()
    # python 中判断相等 is 和 ==
    
    # is 和 ==
    # == 是比较操作符，判断两个对象是否相等，判据是对象的 value
    # is 是同一性算符，判断两个对象是否相同，判据包括 id, dtype value
    # example： test0 = [0,1,2] test1 = test0
    #           test0 == test1 -> True
    #           test0 is test1 -> True
    #           test0 = [0,1,2] test1 = test0.copy()
    #           test0 == test1 -> True
    #           test0 is test1 -> False
    
    # 辨析 python 中的空结构
    # 首先，上面列举的这些空结构都是不同的，无论是作为对象或者是value
    # 但是在 python 中 not 是一个逻辑算符
    # 这个结果是，not 条件 = Boolen，结果都是布尔值
    # 而对于空结构 not 的结果都是 False
    # 即 not False == not None == not [] ...... 等，换成 is 也成立，因为他们的结果都是 True
    # 因此，如果是想严格的判断 test 是 None
    # 逻辑顺序是 test is None ;而想判断不是就应该是 not(test is None)
    # 可惜这样子不符合语法规范，在 python 中，表达这个逻辑的正确语法
    # 是 test is not None 
    
    context = " by %s" % estimator_name if estimator is not None else ""
    
    
    # 检查输入的 array 是否稀疏矩阵 sparse matrix
    # 如果是，进行复杂数据筛查，数据格式转化，稀疏矩阵压缩方法转化
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)
        
        # ensure_2d 对于非二维结构的 X 是否报错
        if ensure_2d:
            
            #--------------------------------------------------------
            # 非二维的 X ndim == 0/1
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
        
        #---------------------------------------------------------------
        # np.issubdtype(arg1, arg2)
        # return True 如果数据结构等级上 arg1 <= arg2
        # np.flexible 是numpy 的一个数据结构，我还没有看明白
        # 参考 https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
        #      https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
        
        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
            
        # allow_nd 变量：是否接受ndim > 2 的 input
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        # 检查数据的有限性，是否有 np.nan np.inf 等
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')
    # 正确打印 array 的 shape
    shape_repr = _shape_repr(array.shape)
    
    # ensur_min_sample 变量：所能接受的 array 中包含的样品个数，即 array row 方向的大小
    # _num_samples 先统计 array 中的样品个数
    # 如果小于要求则报错
    # 可以看出，如果 ensure_min_sample = 0 则跳过检查
    
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))
    
    # array.shape = ['col','row']
    # 因此 array.shape[1] 对应的是 features
    # ensure_min_features 对应于 ensure_min_samples features 的检查 
    
    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))
            
    # if warn_on_dtype  = True 如果对于输入的格式在数据检查要是发生改变，那么给出警告
    # dtype_orig 判断输入格式是否真的发生了改变
    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    
    #--------------------------------------------------
    # copy 判断是否对 array 进行一份 copy
    # 这里 if 有两个判别条件
    # 一个是输入的 copy 是否为 True
    # 另外一个是之前的 array 检查过程中，array 是否有生成新的 copy
    # 一般如果 array 的 dtype 等发生了改变，如 asformmat 等方法会生成
    # 新的 copy。而如果没有发生改变，array 和 array_orig 共享内存
    # 因为 = 赋值的方法只是生成一个新的 reference，而不是用了新的内存
    
    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)
    
    #--------------------------------------------------
    # set(iteriable) 返回一个集合对象，满足集合的定义，可进行集合的操作
    # example：x = set('test1') -> {'1','e','s','t'}
    #          y = set('test2') -> {'2','e','s','t'}
    #          x & y = {'e','s','t'}
    #          x | y = {'1','2','e','s','t'}
    #          x - y = {'1'}
    #          y - x = {'2'}
    # 其他操作，x.add x.update x.remove len(x)
    
    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array