#!/usr/bin/env python3
"""
准备航司识别测试数据
从labels.csv中提取航司数据并创建测试集目录结构
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

def prepare_airline_test_data():
    """准备航司测试数据"""
    labels_file = Path('data/labels.csv')
    output_dir = Path('data/splits/latest/aerovision/airline/test')
    labeled_dir = Path('data/labeled')

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取labels.csv并按航司分组
    airline_images = defaultdict(list)

    with open(labels_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        print(f'CSV Columns: {reader.fieldnames}')
        count = 0
        for row in reader:
            count += 1
            if count <= 5:
                print(f"Row {count}: filename={row.get('filename')}, airlinename={row.get('airlinename')}")
            
            # Handle BOM in column names
            filename = row.get('\ufefffilename') or row.get('filename')
            airlinename = row.get('airlinename')

            if filename and airlinename:
                src_path = labeled_dir / filename
                if src_path.exists():
                    airline_images[airlinename].append(src_path)
                else:
                    if count <= 5:
                        print(f"  File not found: {src_path}")

    # 为每个航司创建目录并复制图片（每个航司最多3张）
    for airline, images in airline_images.items():
        airline_dir = output_dir / airline
        airline_dir.mkdir(parents=True, exist_ok=True)

        # 复制最多3张图片
        for img in images[:3]:
            shutil.copy(img, airline_dir / img.name)

    print(f"Created {len(airline_images)} airline directories")
    print(f"Total images copied: {sum(len(images[:3]) for images in airline_images.values())}")

if __name__ == '__main__':
    prepare_airline_test_data()
