#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Pipeline：注册号检测和识别
使用 YOLOv8 检测注册号区域，使用 PaddleOCR 识别注册号文字
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"错误：缺少必要的依赖库: {e}")
    print("请安装: pip install ultralytics opencv-python pillow")
    sys.exit(1)

# 导入OCR模块
try:
    from training.src.ocr.paddle_ocr import PaddleOCRWrapper, create_ocr
except ImportError:
    # 如果导入失败，尝试直接导入
    sys.path.insert(0, str(PROJECT_ROOT / 'training' / 'src'))
    from ocr.paddle_ocr import PaddleOCRWrapper, create_ocr


class RegistrationOCRPipeline:
    """
    注册号OCR Pipeline
    
    完整的端到端流程：
    1. 使用YOLOv8检测注册号区域
    2. 使用PaddleOCR识别注册号文字
    """
    
    def __init__(
        self,
        detector_model_path: str,
        ocr_lang: str = 'ch',
        ocr_use_gpu: bool = False,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        初始化OCR Pipeline
        
        Args:
            detector_model_path: YOLOv8检测模型路径
            ocr_lang: OCR语言 ('ch': 中文, 'en': 英文)
            ocr_use_gpu: 是否使用GPU进行OCR
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IoU阈值
        """
        print("初始化OCR Pipeline...")
        
        # 加载YOLOv8检测模型
        print(f"加载检测模型: {detector_model_path}")
        if not os.path.exists(detector_model_path):
            raise FileNotFoundError(f"检测模型不存在: {detector_model_path}")
        self.detector = YOLO(detector_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 初始化OCR
        print(f"初始化OCR (语言: {ocr_lang}, GPU: {ocr_use_gpu})")
        self.ocr = create_ocr(lang=ocr_lang, use_gpu=ocr_use_gpu)
        
        print("OCR Pipeline 初始化完成!\n")
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_path: 输出结果路径（可选）
            visualize: 是否可视化结果
        
        Returns:
            包含识别结果的字典
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 步骤1: 使用YOLOv8检测注册号区域
        print(f"检测注册号区域: {image_path}")
        detection_results = self.detector(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 获取检测结果
        boxes = []
        if detection_results and len(detection_results) > 0:
            boxes = detection_results[0].boxes
        
        # 步骤2: 对每个检测到的区域进行OCR识别
        results = []
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            
            # 裁剪注册号区域
            crop = image_rgb[y1:y2, x1:x2]
            
            # OCR识别
            ocr_result = self.ocr.ocr_text_with_boxes(crop)
            
            # 提取识别的文字
            registration_text = ""
            if ocr_result:
                # 选择置信度最高的结果
                best_result = max(ocr_result, key=lambda x: x['confidence'])
                registration_text = best_result['text']
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'registration': registration_text,
                'ocr_details': ocr_result
            })
        
        # 可视化
        if visualize:
            vis_image = self._visualize_results(image_rgb, results)
            
            if output_path:
                # 保存可视化结果
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存带标注的图片
                vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, vis_bgr)
                print(f"可视化结果已保存到: {output_path}")
        
        # 返回结果
        return {
            'image_path': image_path,
            'detections': len(results),
            'results': results,
            'best_registration': results[0]['registration'] if results else ""
        }
    
    def _visualize_results(
        self,
        image: np.ndarray,
        results: List[Dict]
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图片
            results: 检测结果列表
        
        Returns:
            可视化后的图片
        """
        vis_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            registration = result['registration']
            confidence = result['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制文字标签
            label = f"{registration} ({confidence:.2f})"
            
            # 计算文字背景框大小
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # 绘制文字背景
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1
            )
            
            # 绘制文字
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return vis_image
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        visualize: bool = True
    ) -> List[Dict]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录（可选）
            visualize: 是否可视化结果
        
        Returns:
            识别结果列表
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n处理图片 {i+1}/{len(image_paths)}: {image_path}")
            
            # 确定输出路径
            output_path = None
            if output_dir:
                output_filename = f"result_{Path(image_path).stem}.jpg"
                output_path = os.path.join(output_dir, output_filename)
            
            # 处理图片
            result = self.process_image(
                image_path,
                output_path=output_path,
                visualize=visualize
            )
            results.append(result)
        
        return results
    
    def save_results_json(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        保存结果到JSON文件
        
        Args:
            results: 识别结果列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='OCR Pipeline：注册号检测和识别',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单张图片
  python ocr_pipeline.py --detector training/checkpoints/stage6/registration_detector/weights/best.pt --image test.jpg --output result.jpg
  
  # 批量处理
  python ocr_pipeline.py --detector training/checkpoints/stage6/registration_detector/weights/best.pt --image images/ --output results/
  
  # 使用英文OCR
  python ocr_pipeline.py --detector training/checkpoints/stage6/registration_detector/weights/best.pt --image test.jpg --ocr-lang en
        """
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        required=True,
        help='检测模型路径'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='输入图片路径或目录'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出结果路径（单张图片）或目录（批量处理）'
    )
    
    parser.add_argument(
        '--ocr-lang',
        type=str,
        default='ch',
        choices=['ch', 'en', 'japan', 'korean', 'fr', 'german'],
        help='OCR语言 (默认: ch)'
    )
    
    parser.add_argument(
        '--ocr-gpu',
        action='store_true',
        help='使用GPU进行OCR'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='检测置信度阈值 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='NMS IoU阈值 (默认: 0.45)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='不保存可视化结果'
    )
    
    args = parser.parse_args()
    
    # 检查检测模型是否存在
    if not os.path.exists(args.detector):
        print(f"错误：检测模型不存在: {args.detector}")
        sys.exit(1)
    
    # 初始化Pipeline
    pipeline = RegistrationOCRPipeline(
        detector_model_path=args.detector,
        ocr_lang=args.ocr_lang,
        ocr_use_gpu=args.ocr_gpu,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 确定输入图片
    image_path = args.image
    if os.path.isdir(image_path):
        # 批量处理
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_path).glob(f'*{ext}'))
            image_files.extend(Path(image_path).glob(f'*{ext.upper()}'))
        
        image_paths = [str(f) for f in image_files]
        
        if not image_paths:
            print(f"错误：在目录 {image_path} 中未找到图片")
            sys.exit(1)
        
        print(f"找到 {len(image_paths)} 张图片\n")
        
        # 批量处理
        results = pipeline.process_batch(
            image_paths,
            output_dir=args.output,
            visualize=not args.no_visualize
        )
        
        # 保存结果到JSON
        if args.output:
            json_path = os.path.join(args.output, 'results.json')
            pipeline.save_results_json(results, json_path)
        
    else:
        # 单张图片处理
        if not os.path.exists(image_path):
            print(f"错误：图片不存在: {image_path}")
            sys.exit(1)
        
        result = pipeline.process_image(
            image_path,
            output_path=args.output,
            visualize=not args.no_visualize
        )
        
        # 打印结果
        print("\n" + "="*60)
        print("识别结果:")
        print("="*60)
        print(f"图片: {result['image_path']}")
        print(f"检测数量: {result['detections']}")
        print(f"注册号: {result['best_registration']}")
        
        if result['results']:
            print("\n详细信息:")
            for i, r in enumerate(result['results'], 1):
                print(f"  {i}. 位置: {r['bbox']}, 置信度: {r['confidence']:.4f}, 注册号: {r['registration']}")
        
        print("="*60)


if __name__ == '__main__':
    main()
