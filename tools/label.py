"""
标注工具 - 启动 LabelImg 标注数据集
"""
import os
import sys
import pickle
import subprocess


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_root, "datasets", "delta_force")

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 检查 labelImg 是否安装
    try:
        import labelImg  # noqa
    except ImportError:
        print("正在安装 LabelImg (含 PyQt5 依赖，约 60MB，请耐心等待)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "labelImg"]
            )
            print("[i] 安装完成，请重新运行此脚本")
        except Exception as e:
            print(f"[!] 安装失败: {e}")
            print("    请手动执行: pip install labelImg")
        return

    # 创建类别文件
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_file):
        default_classes = [
            "operator_friendly",  # 队友
            "operator_enemy",  # 敌人
            "vehicle",  # 载具
            "soldier",  # 小兵
            "knocked",  # 倒地
        ]
        with open(classes_file, "w") as f:
            for cls in default_classes:
                f.write(f"{cls}\n")
        print(f"[i] 已创建类别文件: {classes_file}")
        print(f"    默认类别: {default_classes}")
        print("    请按需修改 classes.txt 后再标注")

    # 设置 Qt 插件路径 (修复 "Could not find the Qt platform plugin" 错误)
    import PyQt5
    qt_plugin_path = os.path.join(
        os.path.dirname(PyQt5.__file__), "Qt5", "plugins"
    )
    if os.path.isdir(qt_plugin_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

    # 预设 labelImg 配置: YOLO 格式 + 自动保存
    from libs.labelFile import LabelFileFormat
    # Qt 类型必须导入，否则 pickle 加载时反序列化 QSize/QPoint/QColor 等会失败
    from PyQt5 import QtCore, QtGui
    settings_path = os.path.join(os.path.expanduser("~"), ".labelImgSettings.pkl")
    try:
        with open(settings_path, "rb") as f:
            settings = pickle.load(f)
    except Exception:
        # 旧配置文件损坏或依赖缺失（如 sip 模块），重置
        settings = {}
        try:
            os.remove(settings_path)
        except OSError:
            pass
    # 固定配置（每次覆盖）
    settings["labelFileFormat"] = LabelFileFormat.YOLO
    settings["autosave"] = True
    settings["savedir"] = labels_dir
    # 保留上次的标注位置（如果已有则不动，没有则设空）
    settings.setdefault("filename", "")
    with open(settings_path, "wb") as f:
        pickle.dump(settings, f)
    print("[i] labelImg 配置: YOLO 格式 + 自动保存 + 记住上次标注位置")

    # 启动 LabelImg
    print(f"启动 LabelImg...")
    print(f"  图片目录: {images_dir}")
    print(f"  标签目录: {labels_dir}")
    print(f"  类别文件: {classes_file}")
    print("\nLabelImg 快捷键:")
    print("  [W] - 创建标注框")
    print("  [D] - 下一张图")
    print("  [A] - 上一张图")
    print("  [Ctrl+S] - 保存")
    print("  标注格式: YOLO\n")

    # 尝试多种方式启动 labelImg
    cmds = [
        ["labelImg"],
        ["labelimg"],
    ]
    success = False
    for cmd in cmds:
        try:
            ret = subprocess.call(
                cmd + [images_dir, classes_file, labels_dir],
                cwd=dataset_dir,
                env=os.environ.copy(),
            )
            success = True
            break
        except FileNotFoundError:
            continue
    if not success:
        print("[!] 无法启动 LabelImg，请检查:")
        print("    1. pip install labelImg 是否成功")
        print("    2. Python Scripts 目录是否在 PATH 中")
        print(f"    3. 手动运行: labelImg \"{images_dir}\" \"{classes_file}\" \"{labels_dir}\"")


if __name__ == "__main__":
    main()
