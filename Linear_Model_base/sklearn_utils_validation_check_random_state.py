# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:42:33 2018

@author: USER
"""
"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 check_random_state
# 根据输入的随机数种子，生成并返回一个伪随机数生成器

#---------------------------------------------------------------
# numpy 中的伪随机数生成
# 几乎查看的文章都是强调不要用 np.random.seed 而是使用 np.random.RandomState

# pseudorandom number generator 伪随机数产生器 PRNG or
# deterministic random bit generator 确定性随机比特产生器 DRBG
# 是一套用来产生随机数的算法。由于是由算法产生的，因此实际上是具有确定性的
# 产生器有两个部分组成：1. random seed 随机数种 用来初始化一个随机数产生器
#                       2. pseudorandom number generator 伪随机数产生器

# random seed 与 PRNG 之间的关系

# example： seed = 1
#           test0 = np.random.RandomState(seed)
#           test1 = np.random.RandomState(seed)

# test0 == test1 --> False
# test0.randn(1) --> array([1.62434536])
# test0.randn(1) --> array([-0.61175641])

# 但是！
# test1.randn(1) --> array([1.62434536])
# test1.randn(1) --> array([-0.61175641])

# 也就是说由同一个 seed 生成的随机数产生器，其产生的随机数列是相同的。
# 当然这不是说一个随机数产生器会持续产生一个相同的随机数，这就不叫产生
# 随机数了

# random seed 可以是 0 ~ 2^32-1 的整数 或者是 None （default）
# 如果 seed = None，不能被视为同一个 seed，而是根本就没有 seed

import numpy as np
import numbers

def check_random_state(seed):
    
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    
    #-----------------------------------------------
    # 其实 RandomState 产生的随机数是可以接受 None 为 seed 的
    # 这时，RandomState 会使用系统自带的随机数产生其来产生随机数
    # 比如 /dev/random unix 系统 windows analogue windows 系统
    # 这里显然没有这么做，而是调用了 np.random 自带的随机数产生器
    # np.random.mtrand._rand
    
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    
    # 如果输入的是一个整数，python 整型或者是 numpy 整型
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    
    # 如果就是一个随机数产生器，直接返回
    if isinstance(seed, np.random.RandomState):
        return seed
    
    # 所有的情况都 return 之后，别的情况进行报错
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)