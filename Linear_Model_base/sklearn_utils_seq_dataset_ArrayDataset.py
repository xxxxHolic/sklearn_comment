# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:55:52 2018

@author: USER
"""
"""
scikit-learn/sklearn/utils/seq_dataset.pyx
sequential data access
"""
# seq_dataset.pyx 用于数据集的排序
# class ArrayDataset
# 用一个二维的 numpy 数组来备份数据集

cdef class ArrayDataset(SequentialDataset):
    
    """Dataset backed by a two-dimensional numpy array.
    The dtype of the numpy array is expected to be ``np.float64`` (double)
    and C-style memory layout.
    """

    #-------------------------------------------------------------
    # X 二维 numpy array 存储 sample 和 feature
    # Y 一维 numpy array 存储 target
    # sample_weight 一维 numpy array，存储数据的权重
    
    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] X,
                  np.ndarray[double, ndim=1, mode='c'] Y,
                  np.ndarray[double, ndim=1, mode='c'] sample_weights,
                  np.uint32_t seed=1):
        
        """A ``SequentialDataset`` backed by a two-dimensional numpy array.
        Parameters
        ----------
        X : ndarray, dtype=double, ndim=2, mode='c'
            The sample array, of shape(n_samples, n_features)
        Y : ndarray, dtype=double, ndim=1, mode='c'
            The target array, of shape(n_samples, )
        sample_weights : ndarray, dtype=double, ndim=1, mode='c'
            The weight of each sample, of shape(n_samples,)
        """
        
        # 先检查数据的长度维度是不是超过C库的limit （这是用 cython 改写的）
        # from libc.limits import INT_MAX
        
        if X.shape[0] > INT_MAX or X.shape[1] > INT_MAX:
            raise ValueError("More than %d samples or features not supported;"
                             " got (%d, %d)."
                             % (INT_MAX, X.shape[0], X.shape[1]))

        # keep a reference to the data to prevent garbage collection
        self.X = X
        self.Y = Y
        self.sample_weights = sample_weights
        
        # sample 个数
        self.n_samples = X.shape[0]
        # feature 的个数
        self.n_features = X.shape[1]

        cdef np.ndarray[int, ndim=1, mode='c'] feature_indices = \
        
            # np.arange 返回一维数组，长度=n_features
            # 用于存储 feature_indices
            np.arange(0, self.n_features, dtype=np.intc)
        self.feature_indices = feature_indices
        self.feature_indices_ptr = <int *> feature_indices.data

        self.current_index = -1
        self.X_stride = X.strides[0] / X.itemsize
        self.X_data_ptr = <double *>X.data
        self.Y_data_ptr = <double *>Y.data
        self.sample_weight_data = <double *>sample_weights.data

        # Use index array for fast shuffling
        cdef np.ndarray[int, ndim=1, mode='c'] index = \
            np.arange(0, self.n_samples, dtype=np.intc)
        self.index = index
        self.index_data_ptr = <int *>index.data
        # seed should not be 0 for our_rand_r
        self.seed = max(seed, 1)

    cdef void _sample(self, double **x_data_ptr, int **x_ind_ptr,
                      int *nnz, double *y, double *sample_weight,
                      int current_index) nogil:
        
        cdef long long sample_idx = self.index_data_ptr[current_index]
        cdef long long offset = sample_idx * self.X_stride

        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.feature_indices_ptr
        nnz[0] = self.n_features
        sample_weight[0] = self.sample_weight_data[sample_idx]