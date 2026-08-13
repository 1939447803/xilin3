"""Pascal VOC XML -> YOLO TXT 格式转换"""
import os
import sys
import io

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import xml.etree.ElementTree as ET
from glob import glob


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_root, "datasets", "delta_force")
    labels_dir = os.path.join(dataset_dir, "labels")
    labels_train = os.path.join(labels_dir, "train")

    # 读取类别
    classes_file = os.path.join(dataset_dir, "classes.txt")
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"类别映射: {class_map}")

    xml_files = glob(os.path.join(labels_dir, "*.xml"))
    print(f"找到 {len(xml_files)} 个 XML 文件")

    os.makedirs(labels_train, exist_ok=True)
    converted = 0

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        filename = root.find("filename").text
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(labels_train, txt_name)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_map:
                print(f"  [!] 未知类别: {name} in {filename}, 跳过")
                continue

            cls_id = class_map[name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 归一化
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        # 删除原 XML
        os.remove(xml_path)
        converted += 1

    print(f"[✓] 转换完成: {converted} 个文件 -> {labels_train}")

    # 验证
    txt_files = glob(os.path.join(labels_train, "*.txt"))
    print(f"labels/train/ 中现有 {len(txt_files)} 个 TXT 文件")


if __name__ == "__main__":
    main()
