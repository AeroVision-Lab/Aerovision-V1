#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AeroVision 环境验证脚本
检查训练环境是否满足要求
"""

import sys
import os
from pathlib import Path

# ANSI 颜色代码
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

# 检查结果
class CheckResult:
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'

def print_header(text):
    """打印标题"""
    print(f"\n{'=' * 40}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{'=' * 40}\n")

def print_check(status, message):
    """打印检查结果"""
    if status == CheckResult.SUCCESS:
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        color = Colors.GREEN
    elif status == CheckResult.WARNING:
        symbol = f"{Colors.YELLOW}⚠{Colors.END}"
        color = Colors.YELLOW
    else:
        symbol = f"{Colors.RED}✗{Colors.END}"
        color = Colors.RED
    
    print(f"[{symbol}] {color}{message}{Colors.END}")

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_check(CheckResult.SUCCESS, f"Python 版本: {version_str}")
        return True, version_str
    else:
        print_check(CheckResult.ERROR, f"Python 版本: {version_str} (需要 >= 3.8)")
        return False, version_str

def check_package(package_name, import_name=None):
    """检查 Python 包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print_check(CheckResult.SUCCESS, f"{package_name}: {version}")
        return True, version
    except ImportError:
        print_check(CheckResult.ERROR, f"{package_name}: 未安装")
        return False, None

def check_torch():
    """检查 PyTorch 和 CUDA"""
    try:
        import torch
        version = torch.__version__
        print_check(CheckResult.SUCCESS, f"PyTorch 版本: {version}")
        
        cuda_available = torch.cuda.is_available()
        print_check(CheckResult.SUCCESS, f"CUDA 可用: {cuda_available}")
        
        if cuda_available:
            cuda_version = torch.version.cuda
            print_check(CheckResult.SUCCESS, f"CUDA 版本: {cuda_version}")
            
            # 检查 GPU
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print_check(CheckResult.SUCCESS, f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                return True, version
            else:
                print_check(CheckResult.WARNING, "未检测到 GPU 设备")
                return True, version
        else:
            print_check(CheckResult.WARNING, "CUDA 不可用，将使用 CPU 训练")
            return True, version
            
    except ImportError:
        print_check(CheckResult.ERROR, "PyTorch: 未安装")
        return False, None

def check_directories():
    """检查必要的目录结构"""
    print("\n检查目录结构...")
    
    # 获取脚本所在目录的父目录（training 目录）
    script_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "data/raw",
        "data/processed/aircraft_crop/unsorted",
        "data/labels",
        "checkpoints",
        "logs",
        "configs",
        "scripts",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = script_dir / dir_path
        if full_path.exists():
            print_check(CheckResult.SUCCESS, f"{dir_path}/")
        else:
            print_check(CheckResult.WARNING, f"{dir_path}/ (不存在)")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0, missing_dirs

def print_summary(all_passed, issues):
    """打印环境摘要"""
    print_header("环境验证结果")
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}环境验证通过！可以开始训练。{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}环境验证完成，但存在一些问题。{Colors.END}")
        print(f"\n{Colors.YELLOW}修复建议:{Colors.END}")
        for issue in issues:
            print(f"  • {issue}")
    
    print(f"\n{'=' * 40}\n")

def main():
    """主函数"""
    print_header("AeroVision 环境验证")
    
    all_passed = True
    issues = []
    
    # 1. 检查 Python 版本
    python_ok, python_version = check_python_version()
    if not python_ok:
        all_passed = False
        issues.append(f"升级 Python 到 3.8 或更高版本（当前: {python_version}）")
    
    # 2. 检查 PyTorch 和 CUDA
    torch_ok, torch_version = check_torch()
    if not torch_ok:
        all_passed = False
        issues.append("安装 PyTorch: pip install torch torchvision")
    
    # 3. 检查必要的依赖包
    packages = [
        ("ultralytics", "ultralytics"),
        ("paddleocr", "paddleocr"),
        ("albumentations", "albumentations"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard"),
        ("pyyaml", "yaml"),
    ]
    
    missing_packages = []
    for package_name, import_name in packages:
        ok, version = check_package(package_name, import_name)
        if not ok:
            all_passed = False
            missing_packages.append(package_name)
    
    if missing_packages:
        install_cmd = f"pip install {' '.join(missing_packages)}"
        issues.append(f"安装缺失的包: {install_cmd}")
    
    # 4. 检查目录结构
    dirs_ok, missing_dirs = check_directories()
    if not dirs_ok:
        # 目录缺失不算错误，只是警告
        issues.append("创建缺失的目录: mkdir -p training/data/{raw,labels,processed/aircraft_crop/unsorted}")
    
    # 5. 打印摘要
    print_summary(all_passed, issues)
    
    # 返回退出码
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
