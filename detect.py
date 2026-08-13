"""
实时检测 - 屏幕截图 + YOLO 目标检测

仅做识别，不包含任何跟踪 / 瞄准逻辑。

用法：
    python detect.py                      # 默认 COCO yolo11n.pt，主屏
    python detect.py --model best.pt      # 指定本地模型
    python detect.py --monitor 1          # 副屏
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import dxcam
from ultralytics import YOLO


COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


def detect_live(args):
    print("=" * 50)
    print("YOLO 实时检测")
    print("=" * 50)

    model = YOLO(args.model)
    class_names = model.names if hasattr(model, 'names') else {}

    print(f"  模型: {args.model}")
    print(f"  置信度: {args.confidence:.2f}")
    print(f"  跳帧: 每 {args.frame_skip} 帧推理一次")
    print(f"  [q] / [Esc] - 退出  [s] - 截图\n")

    camera = dxcam.create(output_idx=args.monitor, output_color="BGR")
    if camera is None:
        print(f"[!] 无法打开显示器 {args.monitor}")
        return

    # 预热：跑一次空推理让 CUDA 初始化
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=args.imgsz,
              half=True, device=args.device, verbose=False)

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 960, 540)

    frame_count = 0
    fps = 0
    fps_timer = time.time()
    save_dir = "detections"
    os.makedirs(save_dir, exist_ok=True)

    detected_boxes = []  # 缓存最近一次推理结果

    while True:
        frame = camera.grab()
        if frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1

        # 跳帧：只有每隔 frame_skip 帧才推理，其余帧复用上次结果
        if frame_count % args.frame_skip == 0:
            results = model(frame, imgsz=args.imgsz, conf=args.confidence,
                            iou=args.iou, device=args.device, half=True,
                            max_det=args.max_det, verbose=False)
            detected_boxes = []
            for r in results:
                if r.boxes is not None:
                    boxes_xyxy = r.boxes.xyxy.cpu().tolist()
                    boxes_cls = r.boxes.cls.cpu().tolist()
                    boxes_conf = r.boxes.conf.cpu().tolist()
                    for xyxy, cls_id, conf in zip(boxes_xyxy, boxes_cls, boxes_conf):
                        detected_boxes.append((xyxy, int(cls_id), float(conf)))

        # 缩放到显示尺寸
        h, w = frame.shape[:2]
        display_w = 960
        scale = display_w / w
        display_h = int(h * scale)
        display = cv2.resize(frame, (display_w, display_h))

        # 画框（使用缓存的结果）
        for xyxy, cls_id, conf in detected_boxes:
            if conf < args.confidence:
                continue
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            color = COLORS[cls_id % len(COLORS)]
            label = class_names.get(cls_id, f"cls_{cls_id}")
            text = f"{label} {conf:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS 计数
        if time.time() - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(display, f"FPS: {fps} | conf: {args.confidence:.2f} | skip: {args.frame_skip}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            path = os.path.join(save_dir, f"detect_{int(time.time())}.jpg")
            cv2.imwrite(path, display)
            print(f"[+] {path}")

    cv2.destroyAllWindows()
    print("\n[*] 检测结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 实时检测")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="模型文件（默认 COCO yolo11n.pt）")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="置信度阈值 (默认 0.2)")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--monitor", type=int, default=0,
                        help="显示器 (0=主屏, 1=副屏)")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="跳帧数 (默认 2, 每2帧推理1次)")
    parser.add_argument("--max-det", type=int, default=50,
                        help="最大检测数 (默认 50)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="推理分辨率 (默认 320)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] 模型文件不存在: {args.model}")
        sys.exit(1)

    detect_live(args)
