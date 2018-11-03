# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:57:00 2018

@author: USER
"""

"""
scikit-learn/sklearn/base.py
base functions and classes
"""
# base.py 各种基础的函数和类
# 函数 RegressorMixin
# Mixin class 用于所有的回归评估器

#-------------------------------------------------------------------------------
# 什么是 mixin？
# 看了一些资料，这似乎是一种约定的 python 编程规范，而不是一种语法要求

# 首先，python 是可以进行多重继承的。但是从编程规范上，多重继承
# 似乎破坏了继承 is-a 原则，一个类不应该属于不同的类别
# 但是，实际使用中，并不一定如此

# example: 比如一个人，生物学上，需要新陈代谢
#          class people(object):
#              def eat(self):
#                  print('food')  
#          但是，社会学上的人还会有别的属性，比如衣服
#          class wearmixin(object):
#              def wear(self):
#                  print('T_shirt')

# 那么好了，一个社会学的人 man，首先要是生物学上的人，但是还要穿衣服。因此
# 要继承 people 这个父类，但是必须要添加 wear 的属性。实现上，是通过多重
# 继承 people wearmixin。但是编程规范上，people 是 man 的父类，但是添加了
# wear 的属性

# class man(wearmixin, people):
#     pass

# 一个实例 adam = man()
#
# 为什么说是一个编程规范而不是语法要求呢？
# issubclass(man, people)    -> True
# issubclass(man, wearmixin) -> True
# 可见语法上还是多重继承

# 那么 mixin 的规范有哪些？

#1. 首先它必须表示某一种功能，而不是某个物品，-able 而不是 is-a
#2. 其次它必须责任单一，如果有多个功能，那就写多个Mixin类
#3. 然后，它不依赖于子类的实现
#4. 最后，子类即便没有继承这个Mixin类，也照样可以工作，就是缺少了某个功能。

#----------------------------------------------------------------------------
# base 中调用这个 mixin 在编程规范上似乎是有些不太合适的

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

class RegressorMixin(object):
    
    """Mixin class for all regression estimators in scikit-learn."""
    
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')