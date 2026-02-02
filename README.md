# VideoGen_Metric

## Evaluation Metric

| 类别 | 包含指标 | 评估重点 |
|---|---|---|
| 生成质量评估 | FID, FVD, CLIP Score | 衡量生成内容的分布质量、多样性、语义对齐 |
| 重建保真度评估 | EPE, PSNR, SSIM, LPIPS | 衡量生成结果与目标内容的像素级或感知相似度 |

final_scores =  {
      "clip": 0.,
      "epe": 0.,
      "lpips": 0.,
      "ssim": 0.,
      "psnr": 0.,
      "fid": 0.,
      "fvd": 0.
}

## Info


| 指标 | 全称 | 主要用途 | 输入要求 | 理想值 | 方向 | 值为 0 的实际含义 |
|---|---|---|---|---|---|---|
| CLIP | Contrastive Language-Image Pre-training Score | 图文语义对齐 | 图像 + 文本 | 越高越好 | ↑ | 图文特征完全不相关 |
| EPE | End-Point Error | 光流/视差精度 | 预测场 + 真实场 | 0 | ↓ | 预测完全准确 |
| LPIPS | Learned Perceptual Image Patch Similarity | 感知相似度 | 预测图 + 真实图 | 0 | ↓ | 感知上完全相同 |
| SSIM | Structural Similarity Index | 结构相似度 | 预测图 + 真实图 | 1 | ↑ | 毫无结构相似性 |
| PSNR | Peak Signal-to-Noise Ratio | 像素级保真度 | 预测图 + 真实图 | \(\infty\) | ↑ | 极端情况（完全失真） |
| FID | Fréchet Inception Distance | 生成图像质量 | 生成集 + 真实集 | 0 | ↓ | 特征分布完全一致 |
| FVD | Fréchet Video Distance | 生成视频质量 | 生成集 + 真实集 | 0 | ↓ | 时空特征分布完全一致 |

---

## 一、生成质量评估指标

### 1. CLIP Score —— 图文语义对齐度

- **作用**：评估生成图像与对应文本描述在语义空间的对齐程度。
- **特点**：值越高越好，理论范围 $[0, \sim 100]$（实际通常为 0–50）。
- **计算公式**：

$$
\text{CLIP Score} = \max(\cos(E_I(I), E_T(T)), 0) \times w
$$

- **符号说明**：
  - $I$：生成图像
  - $T$：文本提示
  - $E_I, E_T$：CLIP 模型的图像编码器与文本编码器
  - $\cos(\cdot,\cdot)$：余弦相似度
  - $w$：缩放系数（通常为 2.5）
- **值为 0 的含义**：图像特征与文本特征完全正交，无任何语义关联。

### 2. FID (Fréchet Inception Distance) —— 弗雷歇起始距离

- **作用**：衡量生成图像与真实图像在特征空间分布的距离。
- **特点**：值越低越好，完美匹配时趋近于 0（实际 \(>0\)）。
- **计算公式**：

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)
$$

- **符号说明**：
  - $\mu_r, \Sigma_r$：真实图像 Inception-v3 特征分布的均值向量与协方差矩阵
  - $\mu_g, \Sigma_g$：生成图像 Inception-v3 特征分布的均值向量与协方差矩阵
  - $\text{Tr}(\cdot)$：矩阵的迹
- **值为 0 的含义**：生成图像与真实图像的特征分布完全一致（理论极限）。

### 3. FVD (Fréchet Video Distance) —— 弗雷歇视频距离

- **作用**：FID 的视频扩展版本，用于评估生成视频的质量。
- **特点**：值越低越好。
- **计算公式**：

$$
\text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)
$$

- **备注**：公式形式与 FID 相同，但特征提取器换为 I3D 等视频网络。
- **值为 0 的含义**：生成视频与真实视频的时空特征分布完全一致。

---

## 二、重建保真度评估指标

### 1. EPE (End-Point Error) —— 端点误差

- **作用**：主要用于光流、视差估计等任务，衡量预测向量场与真实场的平均欧氏距离。
- **特点**：值越低越好，完美时为 0。
- **计算公式**：

$$
\text{EPE} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{v}_i - v_i\|_2
$$

- **符号说明**：
  - $N$：像素总数
  - $\hat{v}_i$：第 $i$ 像素的预测向量（如 $(u,v)$ 光流）
  - $v_i$：对应的真实向量
- **值为 0 的含义**：预测的光流/视差图与真实值完全一致。

### 2. PSNR (Peak Signal-to-Noise Ratio) —— 峰值信噪比

- **作用**：基于均方误差（MSE）的传统图像保真度指标。
- **特点**：值越高越好，单位 dB；完美重建时为无穷大（实践中 MSE=0 时定义为 \(\infty\)）。
- **计算公式**：

$$
\text{MSE} = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}[y(i,j)-\hat{y}(i,j)]^2
$$

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

- **符号说明**：
  - $H, W$：图像高度与宽度
  - $\text{MAX}$：像素最大值（如 255）
- **值为 0 的含义**：特殊情况，表示 MSE 极大（信号完全被噪声淹没），或 $\text{MSE} > \text{MAX}^2$。

### 3. SSIM (Structural Similarity Index) —— 结构相似性指数

- **作用**：从亮度、对比度、结构三方面评估图像相似度，更符合人眼感知。
- **特点**：范围 \([-1, 1]\)，1 表示完全相同。
- **计算公式（局部窗口 \(x, y\)）**：

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

- **符号说明**：
  - $\mu_x, \mu_y$：窗口均值（亮度）
  - $\sigma_x^2, \sigma_y^2$：窗口方差（对比度）
  - $\sigma_{xy}$：协方差（结构相似性）
  - $C_1, C_2$：稳定常数（通常 $C_1=(0.01L)^2$, $C_2=(0.03L)^2$，$L$ 为动态范围）
- **备注**：全局 SSIM 为所有窗口结果的平均值。
- **值为 0 的含义**：两幅图像在这些局部窗口中毫无结构相似性。

### 4. LPIPS (Learned Perceptual Image Patch Similarity) —— 学习感知图像块相似度

- **作用**：使用深度网络特征计算感知相似度，与人类判断高度相关。
- **特点**：值越低越好，0 表示感知上完全相同。
- **计算公式**：

$$
\text{LPIPS}(y, \hat{y}) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(y)_{h,w} - \phi_l(\hat{y})_{h,w}) \|_2^2
$$

- **符号说明**：
  - $\phi_l(\cdot)$：预训练网络（如 VGG/AlexNet）第 \(l\) 层的激活特征
  - $w_l$：该层可学习的权重向量
  - $\odot$：逐元素乘法
- **值为 0 的含义**：两幅图像在所有网络层的特征表达完全一致。




