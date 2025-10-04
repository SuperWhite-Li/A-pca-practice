# pca.py

import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        PCA类的构造函数.
        参数:
        n_components (int): 我们想要保留的主成分数量 (即降维后的维度).
        """
        self.n_components = n_components
        self.components_ = None  # 将用来存储主成分 (特征向量)
        self.mean_ = None  # 将用来存储训练数据的均值
        self.eigenvalues_ = None  # 初始化特征值属性

    def fit(self, X):
        """
        核心训练方法. 它将学习数据的主成分.
        参数:
        X (np.ndarray): 训练数据, 形状为 (n_samples, n_features)
        """
        # 步骤1: 数据中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 步骤2: 计算协方差矩阵
        # rowvar=False 告诉 np.cov() 每一列是一个变量(特征), 每一行是一个观测(样本)
        # 这正是我们数据的组织方式
        C = np.cov(X_centered, rowvar=False)

        # 步骤3: 特征分解协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # 步骤4: 存储所需的主成分和均值
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]  # noqa: F401
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        self.components_ = sorted_eigenvectors[:, : self.n_components]

        # fit方法通常返回self, 这是一个惯例
        return self

    def transform(self, X):
        """
        使用已学习的主成分来对新数据进行降维.
        参数:
        X (np.ndarray): 需要被转换的数据, 形状为 (n_samples, n_features)
        """
        # 步骤5: 数据中心化
        X_centered = X - self.mean_

        # 步骤6: 投影到主成分上
        X_PCA = X_centered @ self.components_

        # 返回降维后的数据
        return X_PCA

    def fit_transform(self, X):
        """
        方便的方法, 结合了fit和transform
        """
        self.fit(X)
        return self.transform(X)
