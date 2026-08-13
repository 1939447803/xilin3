"""
YOLO 视觉识别 Web Demo (Gradio)

功能：上传图片 → YOLO 目标检测 → 返回标注后的图片 + 检测结果表格。

仅做识别，不包含任何跟踪 / 瞄准逻辑。

推理后端自动选择（从快到慢）：
  1. TensorRT engine（*.engine，FP16，~10ms）—— 若存在同名 .engine 自动使用
  2. PyTorch GPU（.pt + CUDA，~30ms）
  3. PyTorch CPU

用法：
    python app.py                  # 自动选择最快后端
    python app.py --model best.pt  # 指定本地模型（GPU）
    python app.py --device cpu     # 强制 CPU
    python app.py --share          # 生成公网分享链接

构建 TensorRT engine（可选，一次性）：
    python build_engine.py         # 生成 yolo11n.engine
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO


# 每个类别的固定颜色（按类别 id 取模，保证同类别颜色稳定）
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


def _find_tensorrt_bin():
    """定位包含 nvinfer_10.dll 的 TensorRT bin/lib 目录（Windows）。

    优先使用环境变量 TENSORRT_BIN，其次在当前目录及其父目录下查找
    TensorRT-* 安装目录。找不到返回 None（此时回退到 GPU/CPU 推理）。
    """
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


def _setup_tensorrt_path():
    """把 TensorRT DLL 目录加入 PATH，使 `import tensorrt` 可用。"""
    trt_bin = _find_tensorrt_bin()
    if trt_bin:
        paths = os.environ.get("PATH", "").split(os.pathsep)
        if trt_bin not in paths:
            os.environ["PATH"] = trt_bin + os.pathsep + os.environ.get("PATH", "")
    return trt_bin


def _tensorrt_available():
    """返回 TensorRT 是否可用（DLL 就绪且能 import）。"""
    _setup_tensorrt_path()
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def resolve_model(model_arg):
    """选择实际加载的模型与后端。

    若存在与 --model 同名的 .engine，优先使用 TensorRT。
    返回 (模型路径, backend)，backend ∈ {"engine", "pt"}。
    """
    if model_arg.endswith(".engine"):
        return model_arg, "engine"
    engine = str(Path(model_arg).with_suffix(".engine"))
    if _tensorrt_available() and os.path.exists(engine):
        return engine, "engine"
    return model_arg, "pt"


def draw_boxes(image: np.ndarray, results) -> np.ndarray:
    """在 BGR 图上绘制检测框，返回 BGR 图。"""
    image = image.copy()
    names = results[0].names if results else {}
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
            color = COLORS[c % len(COLORS)]
            label = f"{names.get(c, c)} {cf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (int(x1), int(y1) - th - 4),
                          (int(x1) + tw + 4, int(y1)), color, -1)
            cv2.putText(image, label, (int(x1) + 2, int(y1) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def build_table(results):
    """从检测结果生成表格数据（Dataframe）。"""
    rows = []
    names = results[0].names if results else {}
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for i, (xy, c, cf) in enumerate(zip(xyxy, cls, conf), start=1):
            rows.append({
                "序号": i,
                "类别": names.get(c, c),
                "置信度": round(float(cf), 3),
                "x1": int(xy[0]), "y1": int(xy[1]),
                "x2": int(xy[2]), "y2": int(xy[3]),
            })
    if not rows:
        return "未检测到目标"
    return rows


def detect(image: np.ndarray, conf_threshold: float, iou_threshold: float):
    """Gradio 回调：接收上传图片，返回标注图 + 表格。"""
    if image is None:
        return None, "请先上传一张图片"
    kwargs = dict(conf=conf_threshold, iou=iou_threshold, verbose=False)
    if BACKEND == "pt":
        kwargs["device"] = DEVICE
    results = model(image, **kwargs)
    annotated = draw_boxes(image, results)
    # BGR -> RGB 供 Gradio 显示
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    table = build_table(results)
    return annotated, table


def main():
    parser = argparse.ArgumentParser(description="YOLO 视觉识别 Web Demo")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="模型文件（.pt 或 .engine；存在同名 .engine 时自动用 TensorRT）")
    parser.add_argument("--device", type=str, default="0",
                        help="PyTorch 推理设备 (0=GPU, cpu=CPU)；engine 后端忽略此参数")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()

    global model, BACKEND, DEVICE
    DEVICE = args.device

    model_path, BACKEND = resolve_model(args.model)
    if not os.path.exists(model_path):
        print(f"[!] 模型文件不存在: {model_path}")
        print("    可先运行 `python build_engine.py` 构建 engine，"
              "或指定 --model 到已有模型")
        sys.exit(1)

    if BACKEND == "engine":
        _setup_tensorrt_path()
        model = YOLO(model_path, task="detect")
    else:
        model = YOLO(model_path)

    backend_label = "TensorRT (engine)" if BACKEND == "engine" else f"PyTorch ({DEVICE})"
    print(f"[*] 模型: {model_path}")
    print(f"[*] 后端: {backend_label}")

    examples = [
        ["samples/delta_20260508_225635_0010.png", 0.25, 0.45],
        ["samples/delta_20260508_225637_0016.png", 0.25, 0.45],
    ]

    with gr.Blocks(title="YOLO 视觉识别 Demo") as demo:
        gr.Markdown(
            "# YOLO 视觉识别 Demo\n"
            f"模型：`{model_path}`  |  后端：`{backend_label}`\n\n"
            "上传图片，点击识别即可得到标注结果与检测列表。**仅识别，无跟踪。**"
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="输入图片")
                with gr.Row():
                    conf_slider = gr.Slider(0.05, 1.0, value=0.25, step=0.05,
                                            label="置信度阈值")
                    iou_slider = gr.Slider(0.1, 1.0, value=0.45, step=0.05,
                                           label="IoU 阈值")
                detect_btn = gr.Button("识别", variant="primary")
                gr.Examples(examples, inputs=[input_image, conf_slider, iou_slider])
            with gr.Column():
                output_image = gr.Image(type="numpy", label="检测结果")
                output_table = gr.Dataframe(
                    headers=["序号", "类别", "置信度", "x1", "y1", "x2", "y2"],
                    label="检测列表",
                )

        detect_btn.click(
            detect, inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_table],
        )

    demo.launch(server_name=args.server_name, server_port=args.port,
                share=args.share)


if __name__ == "__main__":
    main()
