# training/scripts/review_crops.py
"""
简单的图片浏览脚本，用于检查裁剪结果

配置说明：
本脚本使用模块化配置系统，自动加载以下配置模块：
- review.yaml: 审查配置 (review.*, output.*)
- paths.yaml: 路径配置 (data.*, logs.*)

使用方法：
  # 使用默认配置
  python review_crops.py

  # 指定图片目录
  python review_crops.py --image-dir path/to/images

  # 指定样本数量
  python review_crops.py --n-samples 30
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import sys
from datetime import datetime

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def review_random_samples(
        image_dir: str = None,
        n_samples: int = None,
        config_path: str = None
):
    """
    随机查看一些裁剪结果

    Args:
        image_dir: 图片目录（如果为None则从配置读取）
        n_samples: 样本数量（如果为None则从配置读取）
        config_path: 自定义配置文件路径
    """
    # 加载配置
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config(modules=['review', 'paths'], load_all_modules=False)

    # 使用参数或配置值
    image_dir = image_dir or config.get('data.processed.aircraft_crop.unsorted') or config.get('paths.aircraft_crop')
    n_samples = n_samples if n_samples is not None else config.get('review.n_samples', 20)
    cols = config.get('review.grid_cols', 5)
    fig_width = config.get('review.fig_width', 15)
    row_height = config.get('review.row_height', 3)
    dpi = config.get('review.dpi', 150)
    title_length = config.get('review.title_length', 15)

    # 解析图片目录路径
    if image_dir and not Path(image_dir).is_absolute():
        image_dir = config.get_path('data.processed.aircraft_crop.unsorted') or config.get_path('paths.aircraft_crop')

    image_path = Path(image_dir)

    print("=" * 60)
    print("裁剪结果审查")
    print("=" * 60)
    print(f"图片目录: {image_path}")
    print(f"样本数量: {n_samples}")
    print(f"网格列数: {cols}")
    print("=" * 60)

    images = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if len(images) == 0:
        print("未找到图片！")
        return

    print(f"找到 {len(images)} 张图片")

    samples = random.sample(images, min(n_samples, len(images)))

    # 显示图片网格
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, row_height * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_path.name[:title_length] + "...", fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图
    for ax in axes[len(samples):]:
        ax.axis('off')

    plt.tight_layout()

    # 获取保存路径并替换时间戳占位符
    save_path_template = config.get('logs.crop_review') or config.get('output.save_path') or config.get('paths.crop_review')

    if save_path_template:
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 替换{timestamp}占位符
        save_path_str = save_path_template.replace('{timestamp}', timestamp)

        # 转换为Path对象并解析相对路径
        save_path = Path(save_path_str)
        if not save_path.is_absolute():
            config_base = Path(config._config_base_path)
            save_path = (config_base / save_path).resolve()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=dpi)
        print(f"已保存到 {save_path}")

    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="裁剪结果审查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python review_crops.py

  # 指定图片目录
  python review_crops.py --image-dir path/to/images

  # 指定样本数量
  python review_crops.py --n-samples 30

  # 使用自定义配置文件
  python review_crops.py --config my_config.yaml
        """
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='图片目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='随机抽样数量（默认从配置文件读取）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径'
    )

    args = parser.parse_args()

    review_random_samples(
        image_dir=args.image_dir,
        n_samples=args.n_samples,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
