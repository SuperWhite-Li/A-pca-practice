# test_pca.py

import numpy as np
from pca import PCA  # 从我们的pca.py文件中导入PCA类

# --- 准备工作：创建一个共享的、可复现的测试数据集 ---
# 将数据生成部分放在所有测试函数的外面，这样可以被多个测试共享
np.random.seed(42)
X_raw = np.random.randn(100, 2)
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = np.array([[5, 0], [0, 1]])
X = X_raw @ S @ R


# --- 测试用例 1: 测试类的初始化 ---
def test_pca_initialization():
    """
    测试: PCA类在初始化时是否正确设置了属性。
    """
    pca = PCA(n_components=2)
    assert pca.n_components == 2
    assert pca.components_ is None
    assert pca.mean_ is None
    assert pca.eigenvalues_ is None
    print("\n[PASS] test_pca_initialization")


# --- 测试用例 2: 测试 fit 方法的核心逻辑 ---
def test_pca_fit():
    """
    测试: fit方法是否学习到了正确形状和方向的主成分。
    """
    n_components = 1
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # 检查均值是否被正确计算 (应该是一个2元素的向量)
    assert pca.mean_ is not None
    assert pca.mean_.shape == (2,)

    # 检查特征值是否被存储 (应该是一个2元素的向量)
    assert pca.eigenvalues_ is not None
    assert pca.eigenvalues_.shape == (2,)

    # 检查主成分是否被正确存储 (形状应该是 n_features * n_components)
    assert pca.components_ is not None
    assert pca.components_.shape == (2, n_components)

    # 这是一个更高级的检查：验证找到的主成分方向是否正确
    # 理论上的主成分方向是30度角
    expected_direction = np.array([1, 0]) @ R
    found_direction = pca.components_[:, 0]

    # 特征向量的方向可以是正也可以是负，所以我们检查它们是否平行
    # 方法是计算它们之间的点积，其绝对值应该非常接近1
    cosine_similarity = np.abs(np.dot(expected_direction, found_direction))
    assert np.isclose(cosine_similarity, 1.0, atol=1e-4)
    print("[PASS] test_pca_fit")


# --- 测试用例 3: 测试 transform 方法的输出形状 ---
def test_pca_transform_shape():
    """
    测试: transform方法是否返回了正确形状的降维后数据。
    """
    n_components = 1
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_transformed = pca.transform(X)

    # 原始数据是 (100, 2)，降到1维后，形状应该是 (100, 1)
    # 我们的实现返回的是 (n_components, n_samples)，所以是 (1, 100)
    # 更好的实践是返回 (n_samples, n_components)，即 (100, 1)
    # 我们先根据你当前的代码来测试
    # 在你的transform代码中: X_PCA = X_centered @ self.components_
    # (100, 2) @ (2, 1) -> (100, 1)
    assert X_transformed.shape == (100, n_components)
    print("[PASS] test_pca_transform_shape")


# --- 测试用例 4: 测试 fit_transform 方法的等价性 ---
def test_fit_transform_equivalence():
    """
    测试: pca.fit_transform(X) 的结果是否与 pca.fit(X) 后再 pca.transform(X) 的结果相同。
    """
    n_components = 1
    pca1 = PCA(n_components=n_components)
    pca2 = PCA(n_components=n_components)

    # 路径1
    result1 = pca1.fit_transform(X)

    # 路径2
    pca2.fit(X)
    result2 = pca2.transform(X)

    # 两个结果应该非常接近
    assert np.allclose(result1, result2)
    print("[PASS] test_fit_transform_equivalence")
