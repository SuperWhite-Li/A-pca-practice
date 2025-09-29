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
        C = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

        # 步骤3: 特征分解协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # 步骤4: 存储所需的主成分和均值
        # TODO: 在这里编写存储结果的代码

        # fit方法通常返回self, 这是一个惯例
        return self

    def transform(self, X):
        """
        使用已学习的主成分来对新数据进行降维.
        参数:
        X (np.ndarray): 需要被转换的数据, 形状为 (n_samples, n_features)
        """
        # 步骤5: 数据中心化
        # TODO: 用训练时计算的均值来中心化数据

        # 步骤6: 投影到主成分上
        # TODO: 编写数据投影的代码

        # 返回降维后的数据
        pass  # 暂时用pass占位

    # --- 测试我们的类 ---
    if __name__ == "__main__":
        # 我们将在后续步骤中在这里编写测试代码
        print("PCA class structure created.")
