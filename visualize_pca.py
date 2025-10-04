# visualize_pca.py

import numpy as np
import matplotlib.pyplot as plt
from pca import PCA

print("--- 开始可视化验证 ---")

# 1. 创建和我们测试时一样的玩具数据集
np.random.seed(42)
X_raw = np.random.randn(100, 2)
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = np.array([[5, 0], [0, 1]])
X = X_raw @ S @ R

# 2. 初始化PCA，训练并转换数据
pca = PCA(n_components=1)
X_transformed = pca.fit_transform(X)

# 3. 开始绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 子图1: 原始数据和主成分方向
ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

# 绘制主成分箭头
start_point = pca.mean_
arrow_vec = pca.components_[:, 0] * np.sqrt(pca.eigenvalues_[0]) * 3
ax1.quiver(
    *start_point,
    *arrow_vec,
    color="red",
    angles="xy",
    scale_units="xy",
    scale=1,
    label="First Principal Component"
)

ax1.set_title("Original Data with Principal Component")
ax1.set_aspect("equal", adjustable="box")
ax1.grid(True)
ax1.legend()

# 子图2: 降维后的一维数据
ax2.hist(
    X_transformed.flatten(), bins=20, alpha=0.7
)  # .flatten()将(100,1)数组变为(100,)
ax2.set_title("Transformed Data (1 Dimension)")
ax2.grid(True)

plt.tight_layout()
plt.show()

print("--- 可视化完成 ---")
