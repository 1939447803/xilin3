# xilin3 — YOLO 视觉识别

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的**视觉识别**项目：目标检测推理 + 网页 Demo + 数据采集/标注/训练完整流水线。

> 本项目**仅包含视觉识别**，不包含任何跟踪（tracking）或瞄准（aim）逻辑。

## 功能

- **网页 Demo（`app.py`）**：Gradio 界面，上传图片 → YOLO 目标检测 → 标注图 + 检测列表。
- **实时检测（`detect.py`）**：屏幕截图 + 实时目标检测（dxcam 高性能截图）。
- **训练（`train.py`）**：基于自有数据集训练 YOLOv11 模型。
- **数据流水线（`tools/`）**：截图采集、标注、XML→YOLO 格式转换。

## 安装

```bash
pip install -r requirements.txt
```

> 如需 GPU 加速，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch。

## 快速开始（网页 Demo）

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860 ，上传图片点击「识别」。

推理后端按速度自动选择：

| 后端 | 触发条件 | 单张耗时（约） |
|------|----------|----------------|
| TensorRT engine | 存在同名 `.engine` 且 TensorRT 可用 | ~10ms |
| PyTorch GPU | 有 CUDA 且无 engine | ~30ms |
| PyTorch CPU | 无 CUDA | ~300ms+ |

- 默认模型为 COCO `yolo11n.pt`（80 类通用目标，首次运行自动下载）。
- 构建 TensorRT engine（一次性，约 6 分钟，之后极速推理）：

  ```bash
  python build_engine.py          # 生成 yolo11n.engine
  ```

  构建前请确保 `tensorrt` 已安装且 `nvinfer_10.dll` 在 PATH 上，或设置
  环境变量 `TENSORRT_BIN` 指向 TensorRT 的 bin/lib 目录。
- 指定本地模型：`python app.py --model runs/train/delta_force/weights/best.pt`
- 强制 CPU：`python app.py --device cpu`
- 生成公网分享链接：`python app.py --share`

## 实时检测

```bash
python detect.py                          # 默认 COCO 模型，主屏
python detect.py --model best.pt          # 指定本地模型
python detect.py --monitor 1              # 副屏
python detect.py --device cpu             # CPU 推理
```

## 训练自有数据集

```bash
# 1. 采集数据（按 q 截图，Ctrl+C 退出）
python tools/screenshot.py

# 2. 标注数据（LabelImg，YOLO 格式）
python tools/label.py

# 3. 训练
python train.py --model yolo11n.pt --epochs 100 --batch 16

# 4. 用训练好的模型跑网页 Demo
python app.py --model runs/train/delta_force/weights/best.pt
```

## 项目结构

```
xilin3/
├── app.py                  # Gradio 网页 Demo
├── detect.py               # 实时屏幕检测
├── train.py                # 训练脚本
├── build_engine.py         # .pt → TensorRT engine（可选，极速推理）
├── tools/
│   ├── screenshot.py       # 截图采集
│   ├── label.py            # LabelImg 标注启动器
│   └── convert_xml2yolo.py # VOC XML → YOLO TXT
├── datasets/delta_force/   # 数据集（data.yaml + 样例）
└── samples/                # 网页 Demo 示例图片
```

## 说明

- 数据集 `datasets/delta_force/data.yaml` 使用**相对路径**，克隆后可直接使用。
- 仓库内只保留少量样例数据（图片 + 标签），完整数据集与模型权重因体积大不纳入版本控制。
