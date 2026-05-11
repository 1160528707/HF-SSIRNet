import numpy as np
import cv2
import torch


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, attn_weights, img_size=224, alpha=0.6):
    """生成热力图并叠加到原图上"""
    # 将一维注意力权重转换为二维网格
    grid_size = int(np.sqrt(attn_weights.size))
    attn_2d = attn_weights.reshape(grid_size, grid_size)

    # 调整到图像尺寸并归一化
    heatmap = cv2.resize(attn_2d, (img_size, img_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # 转换为伪彩色热力图
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 确保原图尺寸与热力图一致
    if image.shape[:2] != (img_size, img_size):
        image = cv2.resize(image, (img_size, img_size))

    # 将热力图叠加到原图上
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return superimposed_img

def generate_dynamic_nodes(image, attn_weights, grid_coords, img_size=224):
    """生成基于热力值的动态节点图"""
    # 深拷贝原始图像
    node_image = image.copy()

    # 归一化注意力权重到 [0, 1]
    attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)

    # 为每个分块绘制动态节点
    for idx, (x, y) in enumerate(grid_coords):
        # 根据权重设置颜色（红: 高权重，蓝: 低权重）
        red = int(255 * attn_norm[idx])
        blue = int(255 * (1 - attn_norm[idx]))
        color = (blue, 0, red)  # OpenCV 使用 BGR 格式

        # 根据权重设置节点大小（3~8像素）
        radius = int(3 + 5 * attn_norm[idx])

        # 绘制节点
        cv2.circle(node_image, (x, y), radius, color, thickness=-1)

    return node_image