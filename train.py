"""
训练 YOLOv11 模型 - 三角洲行动角色识别
"""
import os
import sys
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args):
    # 数据集配置
    data_yaml = args.data or os.path.join(
        os.path.dirname(__file__), "datasets", "delta_force", "data.yaml"
    )

    if not os.path.exists(data_yaml):
        print(f"[!] 数据集配置文件不存在: {data_yaml}")
        print("请先截取图片并标注，或创建 data.yaml")
        sys.exit(1)

    # 模型选择
    model_name = args.model  # yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt
    print(f"[*] 加载预训练模型: {model_name}")
    model = YOLO(model_name)

    # 训练参数
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        augment=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        amp=True,
        val=True,
        save=True,
        verbose=True,
    )

    print(f"\n[*] 训练完成!")
    print(f"   模型保存: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 三角洲行动训练")
    parser.add_argument("--data", type=str, default=None, help="data.yaml 路径")
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="模型规模 (n=纳诺, s=小, m=中, l=大, x=超大)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心轮数")
    parser.add_argument("--project", type=str, default="runs/train", help="保存目录")
    parser.add_argument("--name", type=str, default="delta_force", help="实验名称")
    args = parser.parse_args()

    # GPU 检查
    if args.device != "cpu":
        import torch
        if not torch.cuda.is_available():
            print("[!] CUDA 不可用，使用 CPU 训练（会很慢）")
            args.device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[*] GPU: {gpu_name} ({vram:.1f} GB)")
            if vram < 6 and args.model != "yolo11n.pt":
                print(f"[!] 显存较低 ({vram:.1f}GB)，建议使用 yolo11n.pt")
            # 根据显存自动调整 batch size
            if vram < 4:
                args.batch = min(args.batch, 8)
            elif vram < 6:
                args.batch = min(args.batch, 16)
            elif vram < 8:
                args.batch = min(args.batch, 24)

    print("=" * 50)
    print("YOLOv11 三角洲行动 训练")
    print("=" * 50)
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.data or '默认'}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图片尺寸: {args.imgsz}")
    print(f"  设备: {args.device}")
    print("=" * 50)

    train(args)
