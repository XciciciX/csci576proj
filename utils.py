import numpy as np
def rotate_piece(img, rot_idx):
    """
    rot_idx = 0,1,2,3 分别对应 0°,270°,180°,90°
    顺时针旋转
    """
    return np.rot90(img, rot_idx, axes=(0, 1))


def crop_background(img: np.ndarray, bg_color=(0, 0, 0), tol: int = 0) -> np.ndarray:
    """
    裁掉四周与背景相同的区域（默认黑边）。
    tol: 容差，>0 时允许一点点噪声（比如压缩/插值导致的非0边缘）
    """
    if img is None or img.size == 0:
        return img

    bg = np.array(bg_color, dtype=img.dtype).reshape(1, 1, 3)
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16))  # 防止 uint8 下溢
    mask = (diff.max(axis=2) > tol)  # True 表示“不是背景”的像素

    if not mask.any():
        return img  # 全是背景，啥也裁不了

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy()