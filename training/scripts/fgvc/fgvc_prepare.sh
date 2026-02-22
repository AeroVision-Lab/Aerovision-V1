#!/bin/bash
# FGVC_Aircraft 数据集准备脚本

# 1. 下载数据集
echo "下载 FGVC_Aircraft 数据集..."
modelscope download --dataset OmniData/FGVC_Aircraft --local_dir /home/wlx/Aerovision-V1/FGVC_Aircraft

# 2. 解压数据集
echo "解压数据集..."
cd /home/wlx/Aerovision-V1/FGVC_Aircraft/raw
tar -zxvf FGVC_Aircraft.tar.gz

# 3. 列出可用的机型（可选，帮助选择正确的类型名称）
echo ""
echo "=================================="
echo "可用机型列表："
echo "=================================="
cd /home/wlx/Aerovision-V1
python training/scripts/list_fgvc_types.py --fgvc-dir /home/wlx/Aerovision-V1/FGVC_Aircraft/raw

# 4. 准备数据集（根据需要修改 --types 参数）
echo ""
echo "=================================="
echo "准备数据集..."
echo "=================================="
# 注意：FGVC数据集使用简单名称，如 '747-100', '747-300'（无 'Boeing_' 前缀）
# 示例：
#   python training/scripts/fgvc_workflow.py --types 747-100,747-300
#   python training/scripts/fgvc_workflow.py --types all  # 使用所有机型
python training/scripts/fgvc_workflow.py --fgvc-dir /home/wlx/Aerovision-V1/FGVC_Aircraft/raw --output-base /home/wlx/Aerovision-V1/data --types all

echo ""
echo "准备完成！"
echo "输出目录: /home/wlx/Aerovision-V1/data/fgvc_splits/"