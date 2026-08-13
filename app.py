"""
YOLO 视觉识别 Web Demo (Gradio)

功能：上传图片 → YOLO 目标检测 → 返回标注后的图片 + 检测结果表格。

仅做识别，不包含任何跟踪 / 瞄准逻辑。
默认使用 COCO 预训练模型 yolo11n.pt（首次运行会自动下载），
也可通过 --model 指定本地 .pt 模型（例如训练好的 best.pt）。

用法：
    python app.py                  # 默认 COCO yolo11n.pt，CPU
    python app.py --model best.pt  # 指定本地模型
    python app.py --share          # 生成公网分享链接
"""
import argparse

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
    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    annotated = draw_boxes(image, results)
    # BGR -> RGB 供 Gradio 显示
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    table = build_table(results)
    return annotated, table


def main():
    parser = argparse.ArgumentParser(description="YOLO 视觉识别 Web Demo")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="模型文件（默认 COCO yolo11n.pt，可指定本地 .pt）")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备 (cpu / 0 表示 GPU)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()

    global model
    model = YOLO(args.model)

    examples = [
        ["samples/delta_20260508_225635_0010.png", 0.25, 0.45],
        ["samples/delta_20260508_225637_0016.png", 0.25, 0.45],
    ]

    with gr.Blocks(title="YOLO 视觉识别 Demo") as demo:
        gr.Markdown(
            "# YOLO 视觉识别 Demo\n"
            f"模型：`{args.model}`  |  设备：`{args.device}`\n\n"
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
