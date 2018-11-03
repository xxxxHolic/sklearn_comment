# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:43:28 2018

@author: USER

"""

"""
scikit-learn/sklearn/metrics/regression.py
Utilities for linear regression
"""
# regression.py 一系列用于回归任务的功能
# 函数 r2_score
#Functions named as ``*_score`` return a scalar value to maximize: 
#the higher the better
#Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
#the lower the better


import numpy as np
import warnings
from six import string_types
from ..exceptions import DataConversionWarning
from sklearn_utils_validation_check_consistent_length import check_consistent_length
from sklearn_utils_validation__check_reg_targets import _check_reg_targets


def column_or_1d(y, warn=False): # 降维到一维
    
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


def r2_score(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average"):
    
    """R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.
    Read more in the :ref:`User Guide <r2_score>`.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average', \
'variance_weighted'] or None or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.
    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred, multioutput='variance_weighted')
    ... # doctest: +ELLIPSIS
    0.938...
    >>> y_true = [1,2,3]
    >>> y_pred = [1,2,3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1,2,3]
    >>> y_pred = [2,2,2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1,2,3]
    >>> y_pred = [3,2,1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)  #检测长度和格式等

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    # R2 的分子
    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    # R2 的分母
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    # 排除 R2 中的 0 项
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
                
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)