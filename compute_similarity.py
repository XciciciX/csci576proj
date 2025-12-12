
import cv2
import numpy as np
# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 3        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 25              # HSV 每个通道的 bin 数
GRAD_BINS = 25               # 梯度方向直方图 bin 数
ALPHA = 0.7                 # 颜色差权重
BETAB = 0.3                 # 梯度差权重
MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域
def compute_color_hist(strip, bins=COLOR_BINS):
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2],
                        None,
                        [bins, bins, bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0,
                         norm_type=cv2.NORM_L1)
    return hist.flatten()



def compute_grad_hist(gray_strip, mag_strip, ang_strip, bins=GRAD_BINS):
    """
    梯度方向直方图，angle 在 [0, 2pi)，以 mag 为权重。
    """
    # 拉平成 1D
    angles = ang_strip.flatten()
    mags = mag_strip.flatten()

    # 映射到 [0, 2pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    hist = np.zeros(bins, dtype=np.float32)
    bin_width = 2 * np.pi / bins

    for a, m in zip(angles, mags):
        b = int(a // bin_width)
        if 0 <= b < bins:
            hist[b] += m

    # 归一化
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def compute_piece_edge_descriptors(piece_img, edge_strip_width=EDGE_STRIP_WIDTH, extra_strip=0):
    """
    对一个 piece（已经是某个固定旋转）的四条边，计算：
    - 颜色直方图
    - 梯度方向直方图
    返回 dict: {"top": desc, "right": desc, "bottom": desc, "left": desc}
    其中 desc = {"color": color_vec, "grad": grad_vec}
    """
    h, w, _ = piece_img.shape
    k = min(edge_strip_width + extra_strip, h // 3, w // 3)  # 防止太大

    # 获取边缘 strip
    top_strip = piece_img[0:k, :, :]
    bottom_strip = piece_img[h - k:h, :, :]
    left_strip = piece_img[:, 0:k, :]
    right_strip = piece_img[:, w - k:w, :]

    # 梯度在整张 piece 上算一次，然后取对应 strip
    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)

    top_mag = mag[0:k, :]
    top_ang = ang[0:k, :]
    bottom_mag = mag[h - k:h, :]
    bottom_ang = ang[h - k:h, :]
    left_mag = mag[:, 0:k]
    left_ang = ang[:, 0:k]
    right_mag = mag[:, w - k:w]
    right_ang = ang[:, w - k:w]

    edges = {}

    # top
    edges[0] = [compute_color_hist(top_strip), compute_grad_hist(gray[0:k, :], top_mag, top_ang)]
    # right
    edges[1] = [compute_color_hist(right_strip), compute_grad_hist(gray[:, w - k:w], right_mag, right_ang)] 
    # bottom
    edges[2] = [compute_color_hist(bottom_strip), compute_grad_hist(gray[h - k:h, :], bottom_mag, bottom_ang)]
    # left
    edges[3] = [compute_color_hist(left_strip), compute_grad_hist(gray[:, 0:k], left_mag, left_ang)]    


    return edges

def color_distance(c1, c2):
    # 直方图已经 normalize 了
    inter = np.minimum(c1, c2).sum()
    # 交集越大，相似度越高 → 距离越小
    return 1.0 - inter

def grad_distance(g1, g2):
    inter = np.minimum(g1, g2).sum()
    return 1.0 - inter



def edge_distance(descA, descB, alpha=ALPHA, beta=BETAB):
    color_diff = color_distance(descA[0], descB[0])
    grad_diff  = grad_distance(descA[1],  descB[1])
    return alpha * color_diff + beta * grad_diff
