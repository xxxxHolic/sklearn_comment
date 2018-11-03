# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:50:24 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""

# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 check_is_fitted: 
# 对估算器 estimator 进行 is_fitted 检查

from ..exceptions import NotFittedError

#-----------------------------------------------------------
# 今天还是头一次发现，函数作为对象也可以对变量进行赋值传递
# all_or_any = all 将 all 函数赋值给 all_or_any
# 经验证，函数赋值可以用 = copy.copy copy.deepcopy 
# 三种方法进行赋值。与其他 python 变量赋值的区别在于
# 这几种方法赋值得到的id是一致的
# example: def test():
#              print('test')
# x = test y = copy.copy(test) z = copy.deepcopy(test)
# id 均为 1747600948352
# 但是这些变量的函数名都是 test print(x.__name__) = test

#-------------------------------------------------------------
# any() 和 all()
# 用于检查一个循环结构体(list or tuple)中是否有 False,None等元素
# 我的理解是：
# 如果元素 test 是循环结构体的一个元素，any 和 all 的判别标准是
# not (not test)
# 因为 None != False != [] != () 等等
# 但是 not None == not False == not [] == not () == True
# 因此如果有这些元素则返回 not True == False

# 注意两个的区别：
# all 对于空的循环结构体报 True
# any 对于空的循环结构体报 False

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):

    # 通过 attributes 中的选项来检查 estimator 是否 fitted
    # 判别条件 1.estimator 没有 fit 属性。根本就不是 estimator
    #          2.estimator 是否有 attribulte 中的属性
    
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))
    
    # 如果 attributes 不是一个循环结构体
    # [attributes] 先生成一个循环结构体
    
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
   
    # estimator 没有 attribute 中的任何一个属性都报错， not fitted
    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})