# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:50:22 2018

@author: USER
"""

"""
scikit-learn/sklearn/utils/validation.py
Utilities for input validation
"""
# validation.py 用于输入数据的验证以及不同系统和py2，py3版本的兼容性
# 函数 _check_large_sparse
# 检查较大的稀疏矩阵。有两种判据，一种是参数输入决定要不要接收，以及scipy的版本要求

from distutils.version import LooseVersion

from scipy import __version__ as scipy_version # 如何获得函数库的版本号

LARGE_SPARSE_SUPPORTED = LooseVersion(scipy_version) >= '0.14.0'

# 版本号管理

# 方法一
# distutils.version 中的两个函数 LooseVersion 和 StrictVersion

# 顾名思义，两者对版本号的格式标准要求的严格程度不同
# LooseVersion
# 合理的格式标准
# 1.5.1 1.5.2b2 161 3.10a 8.02 3.4j 1996.07.12 3.2.pl0 3.1.1.6
# 2g6 11g 0.960923 2.2beta29 1.13++ 5.5.kw 2.0b1pl0
# StrictVersion
# 合理的格式标准
# 0.4 0.4.0  (these two are equivalent) 0.4.1 0.5a1 0.5b3 0.5 0.9.6
# 1.0 1.0.4a3 1.0.4b1 1.0.4
#
# 方法二
# packageing.version.parse
# version.parse('2.3.1') < version.parse('10.1.2') 



def _check_large_sparse(X, accept_large_sparse=False):
    
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False
    """
    if not (accept_large_sparse and LARGE_SPARSE_SUPPORTED):
        
        # not (accept_large_sparse and LARGE_SPARSE_SUPPORTED)
        # = (not accept_large_sparse) or (not LARGE_SPARSE_SUPPORTED)
        # 如果参数设置不接受大稀疏矩阵 64bit indices 或者scipy版本不支持大稀疏矩阵
        
        supported_indices = ["int32"] # 接受 32bit indices size sparse matrix
        
        # scipy 稀疏矩阵的压缩方法有: coo csc csr bsr  等
        # 压缩方法使用：X = scipy.sparse.coo_matrix(test)
        # X.getformat() 得到 X 使用的是什么压缩方法
        
        if X.getformat() == "coo":
            
            # coo 压缩根据矩阵中非零元素的坐标，因此index_keys 是 col 和 row
            index_keys = ['col', 'row']
            
        elif X.getformat() in ["csr", "csc", "bsr"]:
            
            # csr,csc,bsr 的压缩方法除了根据坐标之外，还根据另一个维度上非零元素的指针 indptr - indice pointer
            # 参考 https://my.oschina.net/u/2362565/blog/2239727
            index_keys = ['indices', 'indptr']
        else:
            return
        
        # index keys 是 sparse matrix 稀疏矩阵的维度，或者是 col 和 row，或者是 row/col indice-pointer
        # 这个循环判断 sparse matrix 的维度是否满足 supported_indices 'int32'
        for key in index_keys:
            
            indices_datatype = getattr(X, key).dtype
            
            # 简单的理解，getattr(object, name) 从结果上来说 = object.name 
            # 那它有什么用呢？
            # 这里就可以看出，如果想循环或者条件 查询/调用 object.name， 即 name
            # 作为一个变量，就没有办法使用 object.name 的方法了， 使用 getattr(object,name)
            # 更进一步的用法 getattr(object, name, default)
            # 如果 object 没有属性 name，那么会返回默认值 default
            # default 可以是 number str list tuple dict boolen 等类型的值
            # 如果 defalut = None， 那么不会报错而是不返回任何值，所以默认 default 总可以为 None
            
            
            # 这里获得 sparse matrix 的维度数据形式，判断是否满足 int32
            
            if (indices_datatype not in supported_indices):
                # 不满足 supported_indices 的条件
                
                if not LARGE_SPARSE_SUPPORTED:
                    # 先判断scipy 的版本满足不满足
                    
                    raise ValueError("Scipy version %s does not support large"
                                     " indices, please upgrade your scipy"
                                     " to 0.14.0 or above" % scipy_version)
                
                # 再提示 sparse matrix 的格式条件满足不满足 
                raise ValueError("Only sparse matrices with 32-bit integer"
                                 " indices are accepted. Got %s indices."
                                 % indices_datatype)
