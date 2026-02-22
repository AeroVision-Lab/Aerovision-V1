# training/scripts/crop_aircraft.py
"""
使用 YOLOv8 检测并裁剪飞机

配置说明：
-----------
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- yolo.yaml: YOLO检测配置 (detection.*, model.*, inference.*)
- crop.yaml: 裁剪配置 (crop.*, output.*, image_extensions)
- paths.yaml: 路径配置 (data.*, models.*, logs.*)

配置键映射：
-----------
- 输入目录: data.raw 或 paths.raw_images
- 输出目录: data.processed.aircraft_crop.unsorted 或 paths.aircraft_crop_unsorted
- 置信度阈值: detection.conf_threshold
- 裁剪padding: crop.padding
- 最小尺寸: crop.min_size
- YOLO模型: model.weights 或 paths.yolo_model
- 推理设备: inference.device 或 device.default
- 飞机类别ID: detection.airplane_class_id
- 图片格式: image_extensions 或 image.extensions
- 输出质量: output.quality
- 日志目录: logs.root 或 paths.logs_root

使用方法：
-----------
1. 使用默认配置:
   python crop_airplane.py

2. 指定输入输出目录:
   python crop_airplane.py --input data/raw --output data/processed/aircraft_crop/unsorted

3. 修改检测参数:
   python crop_airplane.py --conf-threshold 0.7 --padding 0.15

4. 使用自定义配置:
   python crop_airplane.py --config path/to/custom_config.yaml
"""

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import sys

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def crop_aircraft(
        input_dir: str = None,
        output_dir: str = None,
        conf_threshold: float = None,
        padding: float = None,
        min_size: int = None,
        config_path: str = None
):
    """
    检测并裁剪飞机

    Args:
        input_dir: 原始图片目录（如果为None则从配置读取）
        output_dir: 输出目录（如果为None则从配置读取）
        conf_threshold: 检测置信度阈值（如果为None则从配置读取）
        padding: 边界框扩展比例（如果为None则从配置读取）
        min_size: 最小输出尺寸（如果为None则从配置读取）
        config_path: 自定义配置文件路径
    """
    # 加载配置（加载需要的模块）
    if config_path:
        # 使用自定义配置文件
        config = load_config(config_path)
    else:
        # 加载默认配置（只加载需要的模块以提高性能）
        config = load_config(modules=['yolo', 'crop', 'paths'], load_all_modules=False)

    # 使用参数或配置值
    # 优先使用 paths.yaml 中的配置，如果没有则回退到 base.yaml
    input_dir = input_dir or config.get('data.raw') or config.get('paths.raw_images')
    output_dir = output_dir or config.get('data.processed.aircraft_crop.unsorted') or config.get('paths.aircraft_crop_unsorted')
    conf_threshold = conf_threshold if conf_threshold is not None else config.get('detection.conf_threshold')
    padding = padding if padding is not None else config.get('crop.padding')
    min_size = min_size if min_size is not None else config.get('crop.min_size')

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")

    # 加载 YOLOv8（COCO 预训练，包含 airplane 类别）
    model_path = config.get('model.weights') or config.get('paths.yolo_model')
    device = config.get('inference.device') or config.get('device.default', 'cuda')
    model = YOLO(model_path)

    # 显式指定设备
    print(f"使用设备: {device}")
    # YOLO会自动使用可用的GPU，这里我们确认一下
    import torch
    if device == 'cuda' and torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ 配置为使用GPU，但GPU不可用，将使用CPU")
        device = 'cpu'
    else:
        print("ℹ️ 使用CPU进行推理")

    # COCO 数据集中 airplane 的类别 ID
    AIRPLANE_CLASS = config.get('detection.airplane_class_id')

    # 统计
    total = 0
    success = 0
    no_detection = 0
    too_small = 0

    # 记录失败的文件
    failed_files = {
        'no_detection': [],  # 未检测到飞机
        'too_small': [],     # 尺寸太小
        'error': []          # 处理出错
    }

    # 获取所有图片
    image_extensions = config.get('image_extensions') or config.get('image.extensions')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))

    # 去重：Windows系统下文件名不区分大小写，同一文件可能被多次匹配
    # Linux系统下文件名区分大小写，不会有此问题，但去重也不影响
    image_files = list(set(image_files))
    image_files.sort()  # 排序以保持一致性

    print(f"找到 {len(image_files)} 张图片")

    # 显示文件扩展名统计
    from collections import Counter
    ext_counter = Counter([f.suffix.lower() for f in image_files])
    if ext_counter:
        print(f"文件类型分布: {dict(ext_counter)}")

    for img_file in tqdm(image_files, desc="裁剪飞机"):
        total += 1

        try:
            # 检测
            results = model(str(img_file), verbose=config.get('inference.verbose', False))[0]

            # 筛选飞机检测结果
            boxes = results.boxes
            airplane_boxes = []

            for i, cls in enumerate(boxes.cls):
                if int(cls) == AIRPLANE_CLASS and boxes.conf[i] >= conf_threshold:
                    airplane_boxes.append({
                        'box': boxes.xyxy[i].cpu().numpy(),
                        'conf': boxes.conf[i].cpu().item()
                    })

            if not airplane_boxes:
                no_detection += 1
                failed_files['no_detection'].append(img_file.name)
                continue

            # 选择置信度最高的（或最大的）
            best_box = max(airplane_boxes, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_box['box']

            # 打开原图
            img = Image.open(img_file)
            img_w, img_h = img.size

            # 添加 padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = box_w * padding
            pad_h = box_h * padding

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(img_w, x2 + pad_w)
            y2 = min(img_h, y2 + pad_h)

            # 检查尺寸
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                too_small += 1
                failed_files['too_small'].append(img_file.name)
                continue

            # 裁剪并保存
            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            output_file = output_path / img_file.name
            quality = config.get('output.quality', 95)
            cropped.save(output_file, quality=quality)
            success += 1

        except Exception as e:
            print(f"处理 {img_file.name} 时出错: {e}")
            failed_files['error'].append((img_file.name, str(e)))
            continue

    # 打印统计
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"  总数: {total}")
    print(f"  成功: {success}")
    print(f"  未检测到飞机: {no_detection}")
    print(f"  太小跳过: {too_small}")
    print(f"  处理出错: {len(failed_files['error'])}")
    print(f"  输出目录: {output_path}")
    print("=" * 50)

    # 保存失败文件列表到日志
    if no_detection > 0 or too_small > 0 or len(failed_files['error']) > 0:
        log_path = config.get_path('logs.root', create=True) or config.get_path('paths.logs_root', create=True)
        failed_log = log_path / "crop_failed_files.txt"

        with open(failed_log, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("裁剪失败文件详细列表\n")
            f.write("=" * 70 + "\n\n")

            if failed_files['no_detection']:
                f.write(f"未检测到飞机 ({len(failed_files['no_detection'])} 个文件):\n")
                f.write("-" * 70 + "\n")
                for fname in failed_files['no_detection']:
                    f.write(f"  - {fname}\n")
                f.write("\n")

            if failed_files['too_small']:
                f.write(f"检测框太小 ({len(failed_files['too_small'])} 个文件):\n")
                f.write("-" * 70 + "\n")
                for fname in failed_files['too_small']:
                    f.write(f"  - {fname}\n")
                f.write("\n")

            if failed_files['error']:
                f.write(f"处理出错 ({len(failed_files['error'])} 个文件):\n")
                f.write("-" * 70 + "\n")
                for fname, error in failed_files['error']:
                    f.write(f"  - {fname}\n    错误: {error}\n")
                f.write("\n")

        print(f"\n详细失败列表已保存到: {failed_log}")

        # 在控制台显示前10个失败文件作为示例
        if failed_files['no_detection']:
            print(f"\n未检测到飞机的前10个文件:")
            for fname in failed_files['no_detection'][:10]:
                print(f"  - {fname}")
            if len(failed_files['no_detection']) > 10:
                print(f"  ... 还有 {len(failed_files['no_detection']) - 10} 个文件")


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="使用YOLOv8检测并裁剪飞机图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python crop_airplane.py

  # 指定输入输出目录
  python crop_airplane.py --input data/raw --output data/processed/aircraft_crop/unsorted

  # 修改检测参数
  python crop_airplane.py --conf-threshold 0.7 --padding 0.15 --min-size 256

  # 使用自定义配置文件
  python crop_airplane.py --config my_config.yaml

配置说明:
  本脚本使用模块化配置系统，自动加载 yolo.yaml, crop.yaml, paths.yaml
  可以通过命令行参数覆盖配置文件中的值
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='输入图片目录（默认从配置文件读取）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录（默认从配置文件读取）'
    )
    parser.add_argument(
        '--conf-threshold', '-c',
        type=float,
        default=None,
        help='YOLO检测置信度阈值 (0-1)（默认从配置文件读取）'
    )
    parser.add_argument(
        '--padding', '-p',
        type=float,
        default=None,
        help='边界框扩展比例（默认从配置文件读取）'
    )
    parser.add_argument(
        '--min-size', '-m',
        type=int,
        default=None,
        help='最小输出尺寸（像素）（默认从配置文件读取）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径'
    )

    args = parser.parse_args()

    # 调用裁剪函数
    crop_aircraft(
        input_dir=args.input,
        output_dir=args.output,
        conf_threshold=args.conf_threshold,
        padding=args.padding,
        min_size=args.min_size,
        config_path=args.config
    )


if __name__ == '__main__':
    main()
