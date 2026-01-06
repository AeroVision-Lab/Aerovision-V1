"""
图片质量评估模块

评估航空摄影图片的质量，包括清晰度、曝光、构图等指标
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
from PIL import Image
import cv2

from .config import get_config, InferConfig

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """
    图片质量评估器

    基于传统图像处理算法评估图片质量
    """

    def __init__(self, config: Optional[InferConfig] = None):
        """
        初始化评估器

        Args:
            config: 推理配置
        """
        self.config = config or get_config()
        self.quality_config = self.config.quality

    def _load_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """加载图片为 numpy 数组 (BGR)"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图片: {image}")
            return img
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            # 假设输入是 RGB
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")

    def assess_sharpness(self, image: np.ndarray) -> float:
        """
        评估图片清晰度

        使用 Laplacian 算子的方差作为清晰度指标

        Args:
            image: BGR 图片

        Returns:
            清晰度分数 (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # 归一化到 0-1（经验阈值）
        # variance < 100: 模糊, variance > 1000: 清晰
        score = min(variance / 1000.0, 1.0)
        return score

    def assess_exposure(self, image: np.ndarray) -> float:
        """
        评估图片曝光

        分析直方图分布评估曝光是否正确

        Args:
            image: BGR 图片

        Returns:
            曝光分数 (0-1)
        """
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # 计算亮度直方图
        hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # 计算亮度均值和标准差
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)

        # 理想亮度在 100-150 之间
        brightness_score = 1.0 - abs(mean_brightness - 127) / 127.0
        brightness_score = max(0, brightness_score)

        # 检查过曝和欠曝
        overexposed = np.sum(l_channel > 250) / l_channel.size
        underexposed = np.sum(l_channel < 5) / l_channel.size

        # 惩罚过曝和欠曝
        exposure_penalty = overexposed * 0.5 + underexposed * 0.5
        score = brightness_score * (1 - exposure_penalty)

        return max(0, min(1, score))

    def assess_composition(self, image: np.ndarray) -> float:
        """
        评估图片构图

        检查主体位置是否符合三分法则

        Args:
            image: BGR 图片

        Returns:
            构图分数 (0-1)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用边缘检测找主体
        edges = cv2.Canny(gray, 50, 150)

        # 计算边缘质心
        moments = cv2.moments(edges)
        if moments["m00"] == 0:
            return 0.5  # 无法检测到明显主体

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # 计算到三分点的距离
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]

        min_dist_x = min(abs(cx - tx) for tx in thirds_x) / (w / 3)
        min_dist_y = min(abs(cy - ty) for ty in thirds_y) / (h / 3)

        # 距离三分点越近，分数越高
        score = 1.0 - (min_dist_x + min_dist_y) / 2
        return max(0, min(1, score))

    def assess_noise(self, image: np.ndarray) -> float:
        """
        评估图片噪点水平

        Args:
            image: BGR 图片

        Returns:
            噪点分数 (0-1)，分数越高表示噪点越少
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 使用高斯滤波估计噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        noise_level = np.std(noise)

        # 归一化（经验值：noise_level < 5 好，> 20 差）
        score = 1.0 - min(noise_level / 20.0, 1.0)
        return max(0, score)

    def assess_color(self, image: np.ndarray) -> float:
        """
        评估色彩还原

        检查白平衡和色彩饱和度

        Args:
            image: BGR 图片

        Returns:
            色彩分数 (0-1)
        """
        # 转换到 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # 计算饱和度
        mean_sat = np.mean(saturation)
        std_sat = np.std(saturation)

        # 理想饱和度在 60-150 之间
        sat_score = 1.0 - abs(mean_sat - 100) / 100.0
        sat_score = max(0, sat_score)

        # 检查白平衡（通过计算各通道均值差异）
        b, g, r = cv2.split(image)
        mean_diff = abs(np.mean(r) - np.mean(g)) + abs(np.mean(g) - np.mean(b))
        wb_score = 1.0 - min(mean_diff / 50.0, 1.0)

        score = (sat_score + wb_score) / 2
        return max(0, min(1, score))

    def assess(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        综合评估图片质量

        Args:
            image: 输入图片

        Returns:
            {
                "success": True,
                "pass": True/False,
                "score": 0.85,
                "details": {
                    "sharpness": 0.90,
                    "exposure": 0.80,
                    "composition": 0.85,
                    "noise": 0.88,
                    "color": 0.82
                }
            }
        """
        try:
            img = self._load_image(image)

            # 评估各指标
            sharpness = self.assess_sharpness(img)
            exposure = self.assess_exposure(img)
            composition = self.assess_composition(img)
            noise = self.assess_noise(img)
            color = self.assess_color(img)

            # 加权计算总分
            weights = self.quality_config
            total_score = (
                sharpness * weights.sharpness_weight +
                exposure * weights.exposure_weight +
                composition * weights.composition_weight +
                noise * weights.noise_weight +
                color * weights.color_weight
            )

            # 判断是否通过
            is_pass = total_score >= weights.pass_threshold

            return {
                "success": True,
                "pass": is_pass,
                "score": round(total_score, 4),
                "details": {
                    "sharpness": round(sharpness, 4),
                    "exposure": round(exposure, 4),
                    "composition": round(composition, 4),
                    "noise": round(noise, 4),
                    "color": round(color, 4)
                }
            }

        except Exception as e:
            logger.error(f"质量评估失败: {e}", exc_info=True)
            return {
                "success": False,
                "pass": False,
                "score": 0.0,
                "error": str(e)
            }

    def quick_assess(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        快速评估（仅评估清晰度）

        Args:
            image: 输入图片

        Returns:
            {
                "pass": True/False,
                "sharpness": 0.90
            }
        """
        try:
            img = self._load_image(image)
            sharpness = self.assess_sharpness(img)

            return {
                "pass": sharpness >= 0.5,
                "sharpness": round(sharpness, 4)
            }

        except Exception as e:
            logger.error(f"快速评估失败: {e}")
            return {
                "pass": False,
                "sharpness": 0.0,
                "error": str(e)
            }


# 全局实例
_assessor: Optional[ImageQualityAssessor] = None


def get_quality_assessor() -> ImageQualityAssessor:
    """获取全局质量评估器"""
    global _assessor
    if _assessor is None:
        _assessor = ImageQualityAssessor()
    return _assessor
