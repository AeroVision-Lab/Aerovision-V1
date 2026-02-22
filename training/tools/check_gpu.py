# training/scripts/check_gpu.py
"""检查GPU和CUDA环境"""

import torch
from ultralytics import YOLO
import sys
from pathlib import Path

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def check_gpu_environment():
    """检查GPU和CUDA环境"""

    print("=" * 70)
    print("GPU和CUDA环境检查")
    print("=" * 70)

    # 1. PyTorch CUDA检查
    print("\n1. PyTorch CUDA支持:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     名称: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"     计算能力: {props.major}.{props.minor}")
            print(f"     多处理器数量: {props.multi_processor_count}")
    else:
        print("   ⚠️ CUDA不可用，可能原因：")
        print("      - 没有安装NVIDIA驱动")
        print("      - PyTorch版本不支持CUDA")
        print("      - 没有NVIDIA GPU")

    # 2. 测试简单的GPU操作
    print("\n2. GPU运算测试:")
    if torch.cuda.is_available():
        try:
            # 创建一个简单的张量并移到GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   ✅ GPU运算正常")
            print(f"   测试张量设备: {z.device}")

            # 显存使用情况
            print(f"   已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"   缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"   ❌ GPU运算失败: {e}")
    else:
        print("   ⏭️ 跳过（CUDA不可用）")

    # 3. YOLOv8 GPU支持检查
    print("\n3. YOLOv8 GPU支持:")
    try:
        config = load_config()
        model_path = config.get('paths.yolo_model')

        if Path(model_path).exists():
            print(f"   加载模型: {model_path}")
            model = YOLO(model_path)

            # 检查模型设备
            # YOLOv8会自动使用GPU（如果可用）
            if torch.cuda.is_available():
                print(f"   ✅ YOLOv8将自动使用GPU进行推理")
            else:
                print(f"   ℹ️ YOLOv8将使用CPU进行推理")
        else:
            print(f"   ⚠️ 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"   ❌ YOLOv8检查失败: {e}")

    # 4. 配置文件设置
    print("\n4. 当前配置:")
    try:
        config = load_config()
        device = config.get('yolo.device', 'cuda')
        print(f"   配置的设备: {device}")

        if device == 'cuda' and not torch.cuda.is_available():
            print(f"   ⚠️ 配置为使用GPU，但GPU不可用")
            print(f"   建议: 修改config/default.yaml中的device为'cpu'")
        elif device == 'cuda' and torch.cuda.is_available():
            print(f"   ✅ 配置正确，将使用GPU")
        else:
            print(f"   ℹ️ 配置为使用CPU")
    except Exception as e:
        print(f"   ⚠️ 读取配置失败: {e}")

    # 5. 总结和建议
    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)

    if torch.cuda.is_available():
        print("✅ GPU环境正常，推理将使用GPU加速")
        print(f"✅ 推荐使用的设备: cuda")
        print(f"✅ 预期性能提升: 5-10倍（相比CPU）")
    else:
        print("⚠️ GPU不可用，推理将使用CPU")
        print("建议:")
        print("  1. 检查NVIDIA驱动是否正确安装")
        print("  2. 确认PyTorch安装的是CUDA版本")
        print("  3. 运行 'nvidia-smi' 查看GPU状态")
        print("  4. 重新安装PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 70)


if __name__ == "__main__":
    check_gpu_environment()

