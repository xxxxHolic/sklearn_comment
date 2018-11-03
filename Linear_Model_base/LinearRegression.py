# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:16:38 2018

@author: USER
"""

"""
scikit-learn/sklearn/linear_model/base.py
Utilities for linear regression
"""

# base 最基础的线性回归模型
# 函数 LinearRegression
# Ordinary least squares Linear Regression

import warnings
import six
import numpy as np
import scipy.sparse as sp
import functools

from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import lsqr as sparse_lsqr # Find the least-squares solution to a 
                                                    # large, sparse, linear system of equations
from parallel import Parallel
from cloudpickle import dumps

from sklearn_linear_model_base_LinearModel import LinearModel
from sklearn_base_RegressorMixin import RegressorMixin
from sklearn_utils_validation_check_X_y import check_X_y
from sklearn_linear_model_base_preprocess_data import _preprocess_data
from sklearn_linear_model_base__rescale_data import _rescale_data


#--------------------------------------------------------------------------------

def delayed(function, check_pickle=None):
    
    """Decorator used to capture the arguments of a function."""
    
    if check_pickle is not None:
        warnings.warn('check_pickle is deprecated in joblib 0.12 and will be'
                      ' removed in 0.13', DeprecationWarning)
    
    # Try to pickle the input function, to catch the problems early when
    # using with multiprocessing:
    if check_pickle:
        dumps(function)

    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        " functools.wraps fails on some callable objects "
    return delayed_function


#--------------------------------------------------------------------------------
# 这里在调用 mixin 类的编程规范上似乎是有些不太合适的

# 首先，python 在类的继承上，顺序是从右到左
# class class_name(mixin2, mixin1, parent_class)
# 首先继承 parent_class
# 然后添加 mixin1 和 mixin2 的功能
# 但是如果有属性上的冲突，那么会左侧后继承的类会对右侧先继承的类进行
# 覆盖。一般来说，mixin 是添加 class 必须的属性方法。因此，尽量将mixin
# 放置在左侧，按照继承顺序安排代码中的属性
        
# example: class mixin0(object):
#              def test(self):
#                  print('mixin0')
        
#          class mixin1(object):
#              def test(self):
#                  print('mixin1')
        
#          class test0(mixin1, mixin0, object):
#              pass

# t = test0() - t.test() -> mixin1
        
class LinearRegression(LinearModel, RegressorMixin):
    """
    Ordinary least squares Linear Regression.

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
        an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_ # doctest: +ELLIPSIS
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.

    """
    
    # 并没有在初始化的时候传输数据，而只是提供参数
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None): # 线性回归方法
        
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : returns an instance of self.
        """

        n_jobs_ = self.n_jobs
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True) # 检查数据并进行必要的格式转换

        #------------------------------
        # atleast_xd 支持将输入的数据
        # 直接视为 x 维
        # atleast_1d
        # atleast_2d
        # atleast_3d
        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar") # sampple_weight 必须为一维 array

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight) # 对 X 数据进行中心以及归一化

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight) # 给数据添加 sample_weight

        if sp.issparse(X):
            
            if y.ndim < 2:
                
                out = sparse_lsqr(X, y) # 哈哈哈！直接调用 scipy sparse linalg 的 
                                        # least square solution
                self.coef_ = out[0]
                self._residues = out[3]
                
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack(out[0] for out in outs)
                self._residues = np.vstack(out[3] for out in outs)
                
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y) # Compute least-squares solution to equation Ax = b 哈哈哈！！
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            
            self.coef_ = np.ravel(self.coef_)
            self._set_intercept(X_offset, y_offset, X_scale)
        
        return self


def _pre_fit(X, y, Xy, precompute, normalize, fit_intercept, copy,
             check_input=True):
    
    """Aux function used at beginning of fit in linear models"""
    
    n_samples, n_features = X.shape

    if sparse.isspmatrix(X):
        # copy is not needed here as X is not modified inplace when X is sparse
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=fit_intercept, normalize=normalize,
            copy=False, return_mean=True, check_input=check_input)
    else:
        # copy was done in fit if necessary
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=fit_intercept, normalize=normalize, copy=copy,
            check_input=check_input)
    if hasattr(precompute, '__array__') and (
            fit_intercept and not np.allclose(X_offset, np.zeros(n_features)) or
            normalize and not np.allclose(X_scale, np.ones(n_features))):
        warnings.warn("Gram matrix was provided but X was centered"
                      " to fit intercept, "
                      "or X was normalized : recomputing Gram matrix.",
                      UserWarning)
        # recompute Gram
        precompute = 'auto'
        Xy = None

    # precompute if n_samples > n_features
    if isinstance(precompute, six.string_types) and precompute == 'auto':
        precompute = (n_samples > n_features)

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype,
                              order='C')
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, '__array__'):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, '__array__') and Xy is None:
        common_dtype = np.find_common_type([X.dtype, y.dtype], [])
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order='C')
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype,
                          order='F')
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy
