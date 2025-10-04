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


# --- 测试我们的类 ---
if __name__ == "__main__":
    # 1. 创建一个我们"知道"答案的玩具数据集
    # 我们将从一个标准正态分布的点云开始
    np.random.seed(42)
    X_raw = np.random.randn(100, 2)  # 100个样本, 2个特征

    # 然后, 我们对它进行一次"拉伸"和"旋转"
    # 这会创造出一个方差最大的方向(主成分)
    # 这个方向应该是 [1, 1] 经过旋转后的方向
    theta = np.radians(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S = np.array([[5, 0], [0, 1]])  # 延x轴拉伸5倍, y轴不变
    X = X_raw @ S @ R

    print("--- 自动化测试开始 ---")
    print(f"原始数据形状: {X.shape}")

    # 2. 初始化并训练我们的PCA模型
    # 我们想把它降到1维
    pca = PCA(n_components=1)
    pca.fit(X)

    # 3. 检查学习到的属性
    print(f"\n学习到的均值 (self.mean_):\n{pca.mean_}")
    print(f"\n学习到的主成分 (self.components_):\n{pca.components_}")
    # 理论上, 第一个主成分应该是一个指向30度方向的向量 [cos(30), sin(30)]
    # 即大约 [0.865, 0.5]
    print("(理论上的第一主成分方向应接近 [0.865, 0.5] 或其相反方向)")

    # 4. 对数据进行降维 (transform)
    X_transformed = pca.transform(X)

    print(f"\n降维后的数据形状: {X_transformed.shape}")
    print(f"降维后数据的前5行:\n{X_transformed[:5]}")

    # 5. 可视化结果，这是最有说服力的验证!
    import matplotlib

    matplotlib.use("TkAgg")  # 或者 'Qt5Agg'
    import matplotlib.pyplot as plt

    # 5. 可视化结果，这是最有说服力的验证!
    import matplotlib

    matplotlib.use("TkAgg")  # 保留后端设置

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 子图1: 原始数据和主成分方向 ---
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

    # 检查主成分、特征值和均值是否已计算
    assert (
        pca.components_ is not None
        and pca.eigenvalues_ is not None
        and pca.mean_ is not None
    )

    # --- 【推荐的修改】使用 ax1.quiver 绘制箭头 ---
    # quiver 是专门用来绘制向量/箭头的函数

    # 箭头的起点 (数据的均值)
    start_point = pca.mean_

    # 箭头的方向和长度 (主成分向量)
    # 我们仍然用特征值来缩放它，以反映其重要性，但不再需要乘以一个可能过大的常数
    arrow_vec = (
        pca.components_[:, 0] * np.sqrt(pca.eigenvalues_[0]) * 2.5
    )  # 将缩放系数从3减小到2.5

    # 使用 quiver 绘制
    # *start_point 解包成 x, y 起点
    # *arrow_vec 解包成 x, y 方向
    # angles, scale_units, scale 是为了让箭头按数据坐标1:1绘制
    # label 参数可以直接被 legend() 使用，不再需要代理plot
    ax1.quiver(
        *start_point,
        *arrow_vec,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="First Principal Component",
    )

    ax1.set_title("Original Data with Principal Component")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True)
    ax1.legend()  # 现在可以直接调用 legend

    # --- 子图2: 降维后的一维数据 ---
    ax2.hist(X_transformed, bins=20, alpha=0.7)
    ax2.set_title("Transformed Data (1 Dimension)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
