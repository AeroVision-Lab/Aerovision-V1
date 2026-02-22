#!/usr/bin/env python3
"""
配置系统验证脚本

验证所有配置文件和代码使用是否正确:
1. 配置文件能否正常加载
2. 所有配置键是否存在
3. 路径配置是否正确
4. 训练脚本能否正确读取配置
5. Logger 是否正常工作

Usage:
    cd training
    python verify_configs.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import load_config, setup_logger


class ConfigVerifier:
    """配置验证器"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def verify_all(self):
        """运行所有验证"""
        print("=" * 70)
        print("配置系统验证")
        print("=" * 70)
        print()

        # 1. 验证配置加载
        self.verify_config_loading()

        # 2. 验证配置键
        self.verify_config_keys()

        # 3. 验证路径配置
        self.verify_paths()

        # 4. 验证 Logger
        self.verify_logger()

        # 5. 验证训练脚本配置使用
        self.verify_training_scripts()

        # 输出结果
        self.print_results()

    def verify_config_loading(self):
        """验证配置文件能否正常加载"""
        print("1. 验证配置加载...")

        try:
            # 加载所有模块
            config = load_config()
            self.passed.append("[OK] 配置加载成功 (所有模块)")

            # 加载特定模块
            config_classify = load_config(modules=['training', 'paths'], load_all_modules=False)
            self.passed.append("[OK] 分类训练配置加载成功")

            config_airline = load_config(modules=['airline', 'paths'], load_all_modules=False)
            self.passed.append("[OK] 航司训练配置加载成功")

        except Exception as e:
            self.errors.append(f"[X] 配置加载失败: {e}")

        print()

    def verify_config_keys(self):
        """验证关键配置键是否存在"""
        print("2. 验证配置键...")

        try:
            config = load_config()

            # 基础配置
            required_keys = [
                ('device.default', '设备配置'),
                ('seed.random', '随机种子'),
                ('project.name', '项目名称'),
            ]

            # 训练配置
            training_keys = [
                ('training.epochs', '训练轮数'),
                ('training.batch_size', '批次大小'),
                ('training.image_size', '图片尺寸'),
                ('training.optimizer.lr0', '学习率'),
                ('training.optimizer.type', '优化器类型'),
                ('training.scheduler.cosine', '余弦学习率'),
                ('training.early_stopping.patience', '早停耐心值'),
                ('training.warmup.epochs', 'Warmup轮数'),
            ]

            # 路径配置
            path_keys = [
                ('data.raw', '原始数据路径'),
                ('data.prepared.root', '准备数据路径'),
                ('labels.main', '标注文件路径'),
                ('logs.root', '日志根目录'),
                ('checkpoints.classify', '分类检查点路径'),
                ('checkpoints.airline', '航司检查点路径'),
                ('checkpoints.detection', '检测检查点路径'),
            ]

            # YOLO 配置
            yolo_keys = [
                ('model.weights', 'YOLO模型权重'),
                ('yolo_detection.conf_threshold', '置信度阈值'),
                ('yolo_detection.airplane_class_id', '飞机类别ID'),
            ]

            # 日志配置
            logging_keys = [
                ('logging.level', '日志级别'),
                ('logging.format', '日志格式'),
                ('logging.console_level', '控制台日志级别'),
            ]

            all_keys = required_keys + training_keys + path_keys + yolo_keys + logging_keys

            for key, desc in all_keys:
                value = config.get(key)
                if value is not None:
                    self.passed.append(f"[OK] {desc}: {key}")
                else:
                    self.warnings.append(f"[WARN] {desc} 未配置: {key}")

        except Exception as e:
            self.errors.append(f"[X] 配置键验证失败: {e}")

        print()

    def verify_paths(self):
        """验证路径配置"""
        print("3. 验证路径配置...")

        try:
            config = load_config()

            # 检查路径是否使用正确的相对路径格式
            path_keys = [
                'data.raw',
                'data.prepared.root',
                'logs.root',
                'checkpoints.classify',
            ]

            for key in path_keys:
                path_str = config.get(key)
                if path_str:
                    if path_str.startswith('../'):
                        self.passed.append(f"[OK] 路径格式正确: {key} = {path_str}")
                    elif not Path(path_str).is_absolute():
                        self.warnings.append(f"[WARN] 路径格式可能不正确: {key} = {path_str} (应该以 ../ 开头)")
                    else:
                        self.passed.append(f"[OK] 绝对路径: {key} = {path_str}")

            # 测试 get_path 方法
            try:
                logs_path = config.get_path('logs.root')
                self.passed.append(f"[OK] get_path() 工作正常: logs.root -> {logs_path}")
            except Exception as e:
                self.errors.append(f"[X] get_path() 失败: {e}")

        except Exception as e:
            self.errors.append(f"[X] 路径验证失败: {e}")

        print()

    def verify_logger(self):
        """验证 Logger 功能"""
        print("4. 验证 Logger...")

        try:
            config = load_config()

            # 创建临时日志目录
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                log_dir = Path(tmpdir) / "test_logs"

                # 测试 setup_logger
                logger = setup_logger(
                    name="TestLogger",
                    log_dir=log_dir,
                    config=config
                )

                # 测试日志输出
                logger.info("测试日志消息")
                logger.debug("调试消息")

                # 关闭所有 handlers 以释放文件
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

                # 检查日志文件是否创建
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    self.passed.append(f"[OK] Logger 工作正常,日志文件已创建: {log_files[0].name}")
                else:
                    self.errors.append("[X] Logger 未创建日志文件")

        except Exception as e:
            self.errors.append(f"[X] Logger 验证失败: {e}")

        print()

    def verify_training_scripts(self):
        """验证训练脚本的配置使用"""
        print("5. 验证训练脚本配置使用...")

        scripts = [
            ('scripts/train/train_classify.py', 'training', ['training', 'paths']),
            ('scripts/train/train_airline.py', 'airline_training', ['airline', 'paths']),
            ('scripts/train/train_detection.py', 'training', ['training', 'paths']),
        ]

        for script_path, config_prefix, modules in scripts:
            script_file = Path(__file__).parent / script_path
            if not script_file.exists():
                self.warnings.append(f"[WARN] 脚本不存在: {script_path}")
                continue

            try:
                # 读取脚本内容
                content = script_file.read_text(encoding='utf-8')

                # 检查是否使用了统一的 logger
                if 'from configs import' in content and 'setup_logger' in content:
                    self.passed.append(f"[OK] {script_path} 使用统一 logger")
                else:
                    self.warnings.append(f"[WARN] {script_path} 未使用统一 logger")

                # 检查是否使用了 load_config
                if 'load_config' in content:
                    self.passed.append(f"[OK] {script_path} 使用 load_config")
                else:
                    self.errors.append(f"[X] {script_path} 未使用 load_config")

                # 检查是否有硬编码的 setup_logging 函数
                if 'def setup_logging(' in content:
                    self.warnings.append(f"[WARN] {script_path} 仍有 setup_logging 函数定义 (应该删除)")

            except Exception as e:
                self.errors.append(f"[X] 验证 {script_path} 失败: {e}")

        print()

    def print_results(self):
        """输出验证结果"""
        print("=" * 70)
        print("验证结果")
        print("=" * 70)
        print()

        if self.passed:
            print(f"[PASS] 通过 ({len(self.passed)} 项):")
            for msg in self.passed[:10]:  # 只显示前10项
                print(f"  {msg}")
            if len(self.passed) > 10:
                print(f"  ... 还有 {len(self.passed) - 10} 项通过")
            print()

        if self.warnings:
            print(f"[WARN] 警告 ({len(self.warnings)} 项):")
            for msg in self.warnings:
                print(f"  {msg}")
            print()

        if self.errors:
            print(f"[ERROR] 错误 ({len(self.errors)} 项):")
            for msg in self.errors:
                print(f"  {msg}")
            print()

        # 总结
        total = len(self.passed) + len(self.warnings) + len(self.errors)
        print("=" * 70)
        if self.errors:
            print(f"[FAIL] 验证失败: {len(self.errors)} 个错误, {len(self.warnings)} 个警告")
            print("请修复错误后重新运行验证")
            return False
        elif self.warnings:
            print(f"[WARN] 验证通过但有警告: {len(self.warnings)} 个警告")
            print("建议修复警告项以获得最佳体验")
            return True
        else:
            print(f"[SUCCESS] 验证完全通过! 所有 {len(self.passed)} 项检查通过")
            return True


def main():
    """主函数"""
    verifier = ConfigVerifier()
    success = verifier.verify_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
