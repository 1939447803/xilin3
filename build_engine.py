"""
将 .pt 模型导出为 TensorRT FP16 engine（用于网页 Demo 的极速推理）。

用法：
    python build_engine.py                        # yolo11n.pt -> yolo11n.engine
    python build_engine.py --model best.pt        # 指定模型
    python build_engine.py --imgsz 320 --fp16     # 调整分辨率 / 精度

导出后，app.py / detect.py 会自动优先加载同名 .engine。
构建过程需要数分钟，属一次性成本。
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def _find_tensorrt_bin():
    env = os.environ.get("TENSORRT_BIN")
    if env:
        return env
    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for root in roots:
        for pat in ("TensorRT-*/**/bin", "TensorRT-*/**/lib"):
            for d in root.glob(pat):
                if (d / "nvinfer_10.dll").exists():
                    return str(d)
    return None


def main():
    parser = argparse.ArgumentParser(description="导出 YOLO 模型为 TensorRT engine")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--half", action="store_true", default=True, help="FP16")
    parser.add_argument("--dynamic", action="store_true", help="动态 batch")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] 模型不存在: {args.model}")
        sys.exit(1)

    trt_bin = _find_tensorrt_bin()
    if trt_bin:
        os.environ["PATH"] = trt_bin + os.pathsep + os.environ.get("PATH", "")
        print(f"[*] TensorRT bin: {trt_bin}")
    else:
        print("[!] 未找到 nvinfer_10.dll，请设置环境变量 TENSORRT_BIN 指向 TensorRT bin/lib 目录")

    print(f"[*] 导出 {args.model} -> .engine (imgsz={args.imgsz}, half={args.half})")
    model = YOLO(args.model)
    model.export(format="engine", device=0, imgsz=args.imgsz,
                 half=args.half, dynamic=args.dynamic, verbose=False)
    out = str(Path(args.model).with_suffix(".engine"))
    print(f"[✓] 导出完成: {out}")


if __name__ == "__main__":
    main()
